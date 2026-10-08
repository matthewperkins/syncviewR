//! Open Ephys binary-format loading and video-frame <-> sample synchronisation
//! (port of syncview/core/oe.py).

use crate::npy;
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct Recording {
    pub rec_dir: PathBuf,
    pub stream: String,
    pub fs: f64,
    pub ch_names: Vec<String>,
    pub bit_volts: Vec<f64>,
    pub units: Vec<String>,
    pub ch_types: Vec<Option<i64>>, // Open Ephys: 0 ephys, 1 aux, 2 adc
    pub contiguous: bool,
    pub ttl_states: Vec<i64>,
    pub ttl_index: Vec<i64>, // TTL sample numbers -> row index into continuous.dat
    data: Mmap,
    n_ch: usize,
}

impl Recording {
    pub fn open(rec_dir: &Path, stream: &str) -> Result<Self> {
        let rec_dir = std::fs::canonicalize(rec_dir).with_context(|| format!("{}", rec_dir.display()))?;
        let meta: Value = serde_json::from_str(
            &std::fs::read_to_string(rec_dir.join("structure.oebin"))
                .with_context(|| format!("no structure.oebin in {}", rec_dir.display()))?,
        )?;
        let conts = meta["continuous"].as_array().context("structure.oebin has no continuous streams")?;
        let names: Vec<&str> = conts.iter().filter_map(|c| c["stream_name"].as_str()).collect();
        let Some(i) = names.iter().position(|n| *n == stream) else {
            bail!("No continuous stream {stream:?} in {}; available: {}", rec_dir.display(), names.join(", "));
        };
        let cont = &conts[i];
        let chans = cont["channels"].as_array().context("no channels")?;
        let ch_names = chans.iter().map(|c| c["channel_name"].as_str().unwrap_or("").to_string()).collect::<Vec<_>>();
        let bit_volts = chans.iter().map(|c| c["bit_volts"].as_f64().unwrap_or(1.0)).collect();
        let units = chans.iter().map(|c| c["units"].as_str().unwrap_or("").to_string()).collect();
        let ch_types = chans.iter().map(|c| c["type"].as_i64()).collect();
        let folder = rec_dir.join("continuous").join(cont["folder_name"].as_str().context("no folder_name")?);
        let file = File::open(folder.join("continuous.dat"))?;
        // SAFETY: the recording is treated as read-only.
        let data = unsafe { Mmap::map(&file)? };
        let sn = npy::read_i64(&folder.join("sample_numbers.npy"))?;
        let first_sample = *sn.first().context("empty recording")?;
        let contiguous = sn.last().unwrap() - first_sample + 1 == sn.len() as i64;
        if !contiguous {
            eprintln!("syncviewr: warning: continuous sample numbers are not contiguous (dropped samples?) - sync may be off.");
        }
        let ttl = meta["events"].as_array().into_iter().flatten().find(|e| {
            e["stream_name"].as_str() == Some(stream)
                && e["folder_name"].as_str().is_some_and(|f| f.trim_end_matches('/').ends_with("TTL"))
        });
        let (ttl_states, ttl_index) = match ttl {
            None => {
                eprintln!("syncviewr: warning: no TTL events for stream {stream:?} - video sync is impossible.");
                (vec![], vec![])
            }
            Some(e) => {
                let tdir = rec_dir.join("events").join(e["folder_name"].as_str().unwrap());
                let states = npy::read_i64(&tdir.join("states.npy"))?;
                let idx = npy::read_i64(&tdir.join("sample_numbers.npy"))?.into_iter().map(|s| s - first_sample).collect();
                (states, idx)
            }
        };
        let n_ch = ch_names.len();
        Ok(Self {
            rec_dir,
            stream: stream.to_string(),
            fs: cont["sample_rate"].as_f64().context("no sample_rate")?,
            ch_names,
            bit_volts,
            units,
            ch_types,
            contiguous,
            ttl_states,
            ttl_index,
            data,
            n_ch,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.data.len() / (2 * self.n_ch)
    }

    pub fn duration(&self) -> f64 {
        self.n_samples() as f64 / self.fs
    }

    pub fn ch(&self, name: &str) -> Option<usize> {
        self.ch_names.iter().position(|n| n == name)
    }

    pub fn rising_edges(&self, line: i64) -> Vec<i64> {
        self.ttl_states.iter().zip(&self.ttl_index).filter(|(s, _)| **s == line).map(|(_, i)| *i).collect()
    }

    /// {TTL line: number of rising edges} for every line with activity.
    pub fn edge_counts(&self) -> BTreeMap<i64, usize> {
        let mut out = BTreeMap::new();
        for &s in self.ttl_states.iter().filter(|s| **s > 0) {
            *out.entry(s).or_insert(0) += 1;
        }
        out
    }

    /// Rows [i0, i1) clipped to the recording: (first row, number of rows).
    pub fn clip(&self, i0: i64, i1: i64) -> (usize, usize) {
        let a = i0.max(0) as usize;
        let b = (i1.max(0) as usize).min(self.n_samples());
        (a, b.saturating_sub(a))
    }

    /// Physical-unit trace of `ch` (minus `reference`) for rows [first, first + n).
    pub fn trace(&self, ch: usize, reference: Option<usize>, first: usize, n: usize) -> Vec<f64> {
        let raw: &[i16] = bytemuck::cast_slice(&self.data[..self.n_samples() * self.n_ch * 2]);
        let (bv, nc) = (self.bit_volts[ch], self.n_ch);
        let rows = &raw[first * nc..(first + n) * nc];
        match reference {
            None => rows.chunks_exact(nc).map(|r| r[ch] as f64 * bv).collect(),
            Some(r) => {
                let bvr = self.bit_volts[r];
                rows.chunks_exact(nc).map(|row| row[ch] as f64 * bv - row[r] as f64 * bvr).collect()
            }
        }
    }
}

pub struct SyncInfo {
    pub frames: Vec<i64>, // row index of each frame's trigger
    pub issues: Vec<String>,
    pub n_triggers: usize,
    pub extra: i64,
    pub period: f64,
}

/// Frame k <-> k-th rising edge on `line`; extra triggers are expected only at the END.
/// Err when alignment is impossible (no / too few triggers).
pub fn check_sync(rec: &Recording, n_frames: usize, line: i64, video_duration: Option<f64>) -> Result<SyncInfo, String> {
    let edges = rec.rising_edges(line);
    let counts = rec.edge_counts();
    let others: Vec<String> = counts.iter().filter(|(l, _)| **l != line).map(|(l, c)| format!("line {l}: {c}")).collect();
    let others = if others.is_empty() { "none".to_string() } else { others.join(", ") };
    if edges.is_empty() {
        return Err(format!(
            "No triggers on TTL line {line}. Rising edges on other lines: {others}. Pick the camera trigger line with --trigger-line."
        ));
    }
    let extra = edges.len() as i64 - n_frames as i64;
    if extra < 0 {
        let hint: Vec<String> = counts
            .iter()
            .filter(|(l, c)| **l != line && (0..=2).contains(&(**c as i64 - n_frames as i64)))
            .map(|(l, _)| l.to_string())
            .collect();
        return Err(format!(
            "Fewer triggers ({}) on line {line} than video frames ({n_frames}) - wrong recording/video pair or trigger line?{}",
            edges.len(),
            if hint.is_empty() { format!(" Other lines: {others}.") } else { format!(" Line(s) {} have a matching count.", hint.join(", ")) }
        ));
    }
    let fr = edges[..n_frames].to_vec();
    let d: Vec<f64> = fr.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    let med = median(&d);
    let mut issues = vec![];
    if extra > 1 {
        issues.push(format!(
            "{extra} extra triggers after the last frame (expected 0 or 1): frames may be missing from the video, which shifts the alignment."
        ));
    }
    if extra >= 1 && ((edges[n_frames] - fr[n_frames - 1]) as f64) < 1.5 * med {
        issues.push("The first unused trigger follows the last frame by only one frame period, so it looks like a real frame rather than an end-of-recording trigger: the video may have lost a frame.".into());
    }
    let n_long = d.iter().filter(|v| **v > 1.5 * med).count();
    let n_short = d.iter().filter(|v| **v < 0.5 * med).count();
    if n_long > 0 {
        issues.push(format!("{n_long} trigger interval(s) > 1.5x the frame period inside the video (missed triggers or paused camera?) - frames around them may be misaligned."));
    }
    if n_short > 0 {
        issues.push(format!("{n_short} trigger interval(s) < 0.5x the frame period (double or spurious triggers?) - alignment after them is likely off."));
    }
    let span = (fr[n_frames - 1] - fr[0]) as f64 / rec.fs;
    if let Some(vd) = video_duration {
        if span > 0.0 && (vd - span).abs() > (0.01 * span).max(2.0) {
            issues.push(format!("Video duration {vd:.1} s differs from the trigger span {span:.1} s by more than 1% - check that this is the right video and trigger line."));
        }
    }
    if !rec.contiguous {
        issues.push("The recording's sample numbers have gaps (dropped samples), so trigger times may be off.".into());
    }
    Ok(SyncInfo { frames: fr, issues, n_triggers: edges.len(), extra, period: med / rec.fs })
}

/// numpy.median (mean of the two middle values for even lengths); NaN for empty input.
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    let n = s.len();
    let (_, hi, _) = s.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    let hi = *hi;
    if n % 2 == 1 {
        hi
    } else {
        let lo = s[..n / 2].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo + hi) / 2.0
    }
}
