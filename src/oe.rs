//! The recording as the viewer sees it: one continuous stream of an Open Ephys recording, its
//! camera TTL lines, and video-frame <-> sample synchronisation. Reading is done by the
//! `openephys` crate; this keeps the handful of things the viewer needs in one place.

use anyhow::Result;
use openephys::{Continuous, TtlSource};
use std::path::{Path, PathBuf};

pub use openephys::sync::median;

pub struct Recording {
    pub rec_dir: PathBuf,
    pub fs: f64,
    pub ch_names: Vec<String>,
    pub units: Vec<String>,
    pub ch_types: Vec<Option<i64>>, // Open Ephys: 0 ephys, 1 aux, 2 adc
    rec: openephys::Recording,
    stream: usize,
}

impl Recording {
    pub fn open(rec_dir: &Path, stream: &str) -> Result<Self> {
        let rec = openephys::Recording::open(rec_dir)?;
        let c = rec.stream(stream)?;
        let index = rec.continuous().iter().position(|x| std::ptr::eq(x, c)).unwrap();
        if !c.is_contiguous() {
            eprintln!("syncviewr: warning: continuous sample numbers are not contiguous (dropped samples?) - sync may be off.");
        }
        let s = Self {
            rec_dir: rec.dir().to_path_buf(),
            fs: c.sample_rate(),
            ch_names: c.channels().iter().map(|ch| ch.channel_name.clone()).collect(),
            units: c.channels().iter().map(|ch| ch.units.clone()).collect(),
            ch_types: c.channels().iter().map(|ch| ch.type_code).collect(),
            stream: index,
            rec,
        };
        if s.rec.ttl_for(s.cont()).next().is_none() {
            eprintln!("syncviewr: warning: no TTL events for stream {stream:?} - video sync is impossible.");
        }
        Ok(s)
    }

    fn cont(&self) -> &Continuous {
        &self.rec.continuous()[self.stream]
    }

    /// The stream's TTL source with events on `line` (the first source if none has).
    fn ttl(&self, line: i64) -> Option<&TtlSource> {
        let line = u16::try_from(line).ok();
        let mut sources = self.rec.ttl_for(self.cont()).peekable();
        let first = sources.peek().copied();
        sources.find(|t| line.is_some_and(|l| t.lines().contains_key(&l))).or(first)
    }

    /// Row of `continuous.dat` at a sample number (extrapolated outside the recording).
    fn row_of(&self, sample_number: i64) -> i64 {
        let c = self.cont();
        let first = c.first_sample_number().unwrap_or(0);
        if c.is_contiguous() {
            return sample_number - first;
        }
        match c.index_of(sample_number) {
            Ok(i) => i as i64,
            Err(0) => sample_number - first,
            Err(i) if i == c.n_samples() => i as i64 - 1 + (sample_number - c.sample_numbers()[i - 1]),
            Err(i) => i as i64,
        }
    }

    pub fn n_samples(&self) -> usize {
        self.cont().n_samples()
    }

    pub fn duration(&self) -> f64 {
        self.cont().duration()
    }

    pub fn ch(&self, name: &str) -> Option<usize> {
        self.cont().channel_index(name)
    }

    /// Rows of the rising edges on `line`.
    pub fn rising_edges(&self, line: i64) -> Vec<i64> {
        match (self.ttl(line), u16::try_from(line)) {
            (Some(t), Ok(l)) => t.rising(l).into_iter().map(|sn| self.row_of(sn)).collect(),
            _ => vec![],
        }
    }

    /// Rows [i0, i1) clipped to the recording: (first row, number of rows).
    pub fn clip(&self, i0: i64, i1: i64) -> (usize, usize) {
        let n = self.n_samples();
        let a = (i0.max(0) as usize).min(n);
        let b = (i1.max(0) as usize).min(n);
        (a, b.saturating_sub(a))
    }

    /// Physical-unit trace of `ch` (minus `reference`) for rows [first, first + n).
    pub fn trace(&self, ch: usize, reference: Option<usize>, first: usize, n: usize) -> Vec<f64> {
        match reference {
            None => self.cont().read(ch, first..first + n),
            Some(r) => self.cont().read_referenced(ch, r, first..first + n),
        }
    }
}

pub struct SyncInfo {
    pub frames: Vec<i64>, // row index of each frame's trigger
    pub issues: Vec<String>,
    pub extra: i64,
    pub period: f64,
}

/// Frame k <-> k-th rising edge on `line`; extra triggers are expected only at the END.
/// Err when alignment is impossible (no / too few triggers).
pub fn check_sync(rec: &Recording, n_frames: usize, line: i64, video_duration: Option<f64>) -> Result<SyncInfo, String> {
    let l = u16::try_from(line).map_err(|_| format!("bad trigger line {line}"))?;
    let Some(ttl) = rec.ttl(line) else {
        return Err(format!("No TTL events in this recording's stream, so the video can't be synced (trigger line {line})."));
    };
    let a = openephys::align_frames(rec.cont(), ttl, l, n_frames, video_duration).map_err(|e| {
        let e = e.to_string();
        if e.starts_with("No triggers") { format!("{e} Pick the camera trigger line with --trigger-line.") } else { e }
    })?;
    Ok(SyncInfo {
        frames: a.sample_numbers.iter().map(|sn| rec.row_of(*sn)).collect(),
        issues: a.issues,
        extra: a.extra as i64,
        period: a.period,
    })
}
