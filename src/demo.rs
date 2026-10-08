//! `--demo`: a synthetic Open Ephys recording, a cartoon video locked to its camera trigger, and a
//! preset for them, so the viewer can be tried without a rig.
//!
//! Five minutes at 10 kHz: a chewing jaw (masseter and digastric EMG alternating with the jaw
//! opening, a jaw-position sensor on ADC1, and movement artefact shared with the masseter's
//! reference wire), antral slow waves (5/min) that grow after each chewing bout, and duodenal slow
//! waves (~27/min), with 60 and 180 Hz mains hum on every electrode. The camera runs at 30 frames/s
//! from 2 s to 298 s with one trigger per frame plus one when it stops, as on the rig. Every video
//! frame is drawn from the same model at its trigger's time and prints its frame number and that
//! time, so the sync can be checked by eye.

use crate::npy;
use anyhow::{Context, Result};
use ffmpeg_next as ff;
use rayon::prelude::*;
use serde_json::json;
use std::f64::consts::TAU;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Bump when the generated data changes, so old demo folders are not reused.
const VERSION: u32 = 1;
const FS: f64 = 10_000.0;
const DURATION_S: f64 = 300.0;
const FIRST_SAMPLE: i64 = 4_812_330; // Open Ephys sample numbers don't start at 0
const FPS: f64 = 30.0;
const CAM_START_S: f64 = 2.0;
const CAM_STOP_S: f64 = 298.0;
const W: usize = 640;
const H: usize = 480;
/// Chewing bouts: (start s, end s, chewing rate Hz).
const BOUTS: [(f64, f64, f64); 5] = [(20.0, 34.0, 4.2), (62.0, 80.0, 4.8), (128.0, 138.0, 4.5), (185.0, 212.0, 4.0), (245.0, 258.0, 5.0)];
const CH_UV: f64 = 0.195; // bit_volts of the electrode channels
const ADC_V: f64 = 0.000_152_587_9; // bit_volts of the ADC channel

const STREAM_FOLDER: &str = "Acquisition_Board-100.acquisition_board";
const CHANNELS: [&str; 9] = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8", "ADC1"];

pub struct Demo {
    pub rec: PathBuf,
    pub video: PathBuf,
    pub preset: PathBuf,
}

fn paths(dir: &Path) -> Demo {
    Demo {
        rec: dir.join("Record Node 101/experiment1/recording1"),
        video: dir.join("demo_camera.mp4"),
        preset: dir.join("demo_preset.json"),
    }
}

/// The demo folder inside `parent`, generated on first use.
pub fn ensure(parent: &Path) -> Result<Demo> {
    let dir = parent.join(format!("syncviewr_demo_v{VERSION}"));
    if dir.is_dir() {
        return Ok(paths(&dir));
    }
    eprintln!("syncviewr: writing the demo recording and video to {} (once; ~100 MB)…", dir.display());
    let t0 = std::time::Instant::now();
    let tmp = parent.join(format!(".syncviewr_demo_v{VERSION}.{}.tmp", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    let out = paths(&tmp);
    let model = Model::new();
    write_recording(&model, &out.rec).context("writing the demo recording")?;
    write_video(&model, &out.video).context("writing the demo video")?;
    fs::write(&out.preset, serde_json::to_string_pretty(&preset())? + "\n")?;
    fs::write(tmp.join("README.txt"), README)?;
    fs::rename(&tmp, &dir).with_context(|| format!("moving the demo into {}", dir.display()))?;
    eprintln!("syncviewr: demo written in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(paths(&dir))
}

const README: &str = "\
Synthetic data made by `syncviewr --demo`: five minutes of a chewing jaw and a stomach, as an Open
Ephys binary recording plus a 30 frames/s video triggered on TTL line 1. Nothing here is real data.

  CH1, CH2  masseter EMG (bipolar pair: CH2 carries only the shared hum and movement artefact)
  CH3       digastric EMG (opens the jaw; 60 + 180 Hz hum, removed by the preset's notch)
  CH4, CH5  antrum slow waves (5/min; CH5 lags by 2 s; larger after each chewing bout)
  CH6, CH7  duodenum slow waves (~27/min; CH7 lags by 0.4 s)
  CH8       spare electrode (noise and hum only; hidden in the preset)
  ADC1      jaw-position sensor (V)
  TTL 1     camera trigger, one per video frame plus one after the last
  TTL 2     a pulse at the start of each chewing bout (\"food delivered\")

Open it again with
  syncviewr --rec \"Record Node 101/experiment1/recording1\" --video demo_camera.mp4 --preset demo_preset.json
or with Python syncview using the same three files.
";

fn preset() -> serde_json::Value {
    let gi = |ch: &str, label: &str| json!({"ch": ch, "mode": "slow", "label": label, "notch": [60.0]});
    json!({
        "time_base": 10.0,
        "channels": [
            {"ch": "ADC1", "mode": "slow", "label": "jaw sensor", "band": [null, 50.0]},
            {"ch": "CH1", "ref": "CH2", "mode": "hilo", "label": "masseter"},
            {"ch": "CH1", "ref": "CH2", "mode": "envelope", "label": "masseter env"},
            {"ch": "CH3", "mode": "hilo", "label": "digastric", "notch": [60.0, 180.0]},
            gi("CH4", "antrum 1"), gi("CH5", "antrum 2"), gi("CH6", "duodenum 1"), gi("CH7", "duodenum 2"),
            {"ch": "CH8", "mode": "hilo", "label": "spare", "show": false},
        ],
        "overview": {"ch": "CH4", "mode": "bandpower", "label": "antrum slow-wave power"},
    })
}

/// Slow-wave shape over one cycle (phase 0..1): fast upstroke, slow decay; mean ~0.
fn wave(p: f64) -> f64 {
    let p = p.rem_euclid(1.0);
    let v = if p < 0.08 { p / 0.08 } else { (-(p - 0.08) / 0.25).exp() };
    v - 0.284
}

struct Model;

struct Jaw {
    open: f64,      // 0 closed .. 1 fully open
    masseter: f64,  // EMG drive 0..1
    digastric: f64, // EMG drive 0..1
}

impl Model {
    fn new() -> Self {
        Model
    }

    fn jaw(&self, t: f64) -> Jaw {
        for &(s, e, f) in &BOUTS {
            if t > s - 0.5 && t < e + 0.5 {
                let ramp = ((t - s) / 0.3).clamp(0.0, 1.0) * ((e - t) / 0.3).clamp(0.0, 1.0);
                let ph = TAU * f * (t - s);
                return Jaw {
                    open: ramp * (1.0 - ph.cos()) / 2.0,
                    masseter: ramp * (-ph.sin()).max(0.0).powi(2),
                    digastric: ramp * ph.sin().max(0.0).powi(2),
                };
            }
        }
        Jaw { open: 0.0, masseter: 0.0, digastric: 0.0 }
    }

    /// How much has been eaten recently (drives the antral slow-wave amplitude).
    fn fed(&self, t: f64) -> f64 {
        BOUTS
            .iter()
            .filter(|(s, _, _)| t > *s)
            .map(|&(s, e, _)| (e - s) / 20.0 * (1.0 - (-(t - s) / 8.0).exp()) * (-(t - e).max(0.0) / 90.0).exp())
            .sum()
    }

    /// Antral contraction 0..~1 (for the cartoon) and the electrode signal in uV.
    fn antrum(&self, t: f64, lag: f64) -> (f64, f64) {
        let amp = 0.5 + 0.6 * self.fed(t).min(1.5);
        let w = wave((t - lag) / 12.0);
        ((w + 0.284) * amp / 1.3, 400.0 * amp * w)
    }

    fn duodenum(&self, t: f64, lag: f64) -> (f64, f64) {
        let amp = 0.5 + 0.5 * (TAU * t / 40.0).sin().powi(2);
        let w = wave((t - lag) / 2.2);
        ((w + 0.284) * amp, 120.0 * amp * w)
    }
}

/// xorshift64* with Box-Muller normals: deterministic, so every demo folder is identical.
struct Rng(u64, Option<f64>);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        if let Some(v) = self.1.take() {
            return v;
        }
        let (r, a) = ((-2.0 * self.uniform().ln()).sqrt(), TAU * self.uniform());
        self.1 = Some(r * a.sin());
        r * a.cos()
    }
}

/// Rows (from the recording's start) of the camera's rising edges: one per frame, plus the one the
/// rig sends when acquisition stops, here 0.5 s after the last frame.
fn trigger_rows() -> Vec<i64> {
    let n_frames = ((CAM_STOP_S - CAM_START_S) * FPS).round() as i64;
    let mut r: Vec<i64> = (0..n_frames).map(|k| (CAM_START_S * FS + k as f64 * FS / FPS).round() as i64).collect();
    r.push(r[r.len() - 1] + (0.5 * FS) as i64);
    r
}

fn write_recording(m: &Model, rec: &Path) -> Result<()> {
    let cont = rec.join("continuous").join(STREAM_FOLDER);
    let ttl = rec.join("events").join(STREAM_FOLDER).join("TTL");
    fs::create_dir_all(&cont)?;
    fs::create_dir_all(&ttl)?;
    let n = (DURATION_S * FS) as usize;
    let mut rng = Rng(0x5eed_cafe_f00d_1234, None);
    let mut out = BufWriter::with_capacity(1 << 20, File::create(cont.join("continuous.dat"))?);
    let to_i16 = |v: f64, bv: f64| (v / bv).round().clamp(-32768.0, 32767.0) as i16;
    for i in 0..n {
        let t = i as f64 / FS;
        let jaw = m.jaw(t);
        let hum = 25.0 * (TAU * 60.0 * t).sin() + 8.0 * (TAU * 180.0 * t + 0.7).sin();
        let artefact = 60.0 * jaw.open;
        let breath = 25.0 * (TAU * 1.4 * t).sin();
        let mut noise = || 4.0 * rng.normal();
        let mut uv = [0.0; 8];
        uv[1] = hum + artefact + noise();
        uv[0] = hum + artefact + noise();
        uv[2] = hum + noise();
        for (k, lag) in [(3, 0.0), (4, 2.0)] {
            uv[k] = m.antrum(t, lag).1 + breath + hum + noise();
        }
        for (k, lag) in [(5, 0.0), (6, 0.4)] {
            uv[k] = m.duodenum(t, lag).1 + breath + hum + noise();
        }
        uv[7] = hum + noise();
        uv[0] += (6.0 + 260.0 * jaw.masseter) * rng.normal();
        uv[2] += (4.0 + 180.0 * jaw.digastric) * rng.normal();
        let mut row = [0i16; 9];
        for k in 0..8 {
            row[k] = to_i16(uv[k], CH_UV);
        }
        row[8] = to_i16(0.2 + 2.5 * jaw.open + 0.004 * rng.normal(), ADC_V);
        out.write_all(bytemuck::cast_slice(&row))?;
    }
    out.flush()?;
    npy::write_i64(&cont.join("sample_numbers.npy"), &(0..n as i64).map(|i| FIRST_SAMPLE + i).collect::<Vec<_>>())?;

    // TTL events: line 1 = camera (3 ms pulses), line 2 = a pulse at each bout start.
    let mut ev: Vec<(i64, i16)> = vec![];
    for r in trigger_rows() {
        ev.extend([(r, 1), (r + 30, -1)]);
    }
    for (s, _, _) in BOUTS {
        let r = (s * FS) as i64;
        ev.extend([(r, 2), (r + 1000, -2)]);
    }
    ev.sort();
    npy::write_i64(&ttl.join("sample_numbers.npy"), &ev.iter().map(|e| FIRST_SAMPLE + e.0).collect::<Vec<_>>())?;
    npy::write_i16(&ttl.join("states.npy"), &ev.iter().map(|e| e.1).collect::<Vec<_>>())?;

    let channel = |name: &str| {
        let adc = name.starts_with("ADC");
        json!({
            "channel_name": name,
            "description": if adc { "ADC input channel (synthetic)" } else { "Headstage channel (synthetic)" },
            "identifier": if adc { "acq-board.rhythm.continuous.adc" } else { "acq-board.rhythm.continuous.ephys" },
            "history": "syncviewr --demo",
            "bit_volts": if adc { ADC_V } else { CH_UV },
            "units": if adc { "V" } else { "uV" },
            "type": if adc { 2 } else { 0 },
        })
    };
    let meta = json!({
        "GUI version": "0.6.7",
        "continuous": [{
            "folder_name": format!("{STREAM_FOLDER}/"),
            "sample_rate": FS,
            "source_processor_name": "Acquisition Board",
            "source_processor_id": 100,
            "stream_name": "acquisition_board",
            "recorded_processor": "Record Node",
            "recorded_processor_id": 101,
            "num_channels": CHANNELS.len(),
            "channels": CHANNELS.iter().map(|c| channel(c)).collect::<Vec<_>>(),
        }],
        "events": [{
            "folder_name": format!("{STREAM_FOLDER}/TTL/"),
            "channel_name": "Acquisition Board TTL Input",
            "description": "Events on digital input lines (synthetic)",
            "identifier": "acq-board.rhythm.events",
            "sample_rate": FS,
            "type": "int16",
            "source_processor": "Acquisition Board",
            "stream_name": "acquisition_board",
            "initial_state": 0,
        }],
        "spikes": [],
    });
    fs::write(rec.join("structure.oebin"), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}

// ---------------------------------------------------------------- video

struct Canvas(Vec<u8>);

impl Canvas {
    fn fill(&mut self, c: [u8; 3]) {
        for px in self.0.chunks_exact_mut(3) {
            px.copy_from_slice(&c);
        }
    }
    fn put(&mut self, x: i64, y: i64, c: [u8; 3]) {
        if (0..W as i64).contains(&x) && (0..H as i64).contains(&y) {
            let i = 3 * (y as usize * W + x as usize);
            self.0[i..i + 3].copy_from_slice(&c);
        }
    }
    /// Fill the pixels within the box for which `inside(x, y)` holds.
    fn shape(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, c: [u8; 3], inside: impl Fn(f64, f64) -> bool) {
        for y in y0.floor() as i64..=y1.ceil() as i64 {
            for x in x0.floor() as i64..=x1.ceil() as i64 {
                if inside(x as f64 + 0.5, y as f64 + 0.5) {
                    self.put(x, y, c);
                }
            }
        }
    }
    fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, c: [u8; 3]) {
        self.shape(x, y, x + w, y + h, c, |px, py| px >= x && px < x + w && py >= y && py < y + h);
    }
    fn ellipse(&mut self, cx: f64, cy: f64, rx: f64, ry: f64, c: [u8; 3]) {
        self.shape(cx - rx, cy - ry, cx + rx, cy + ry, c, |x, y| ((x - cx) / rx).powi(2) + ((y - cy) / ry).powi(2) <= 1.0);
    }
    /// Rectangle [0, len] x [0, thick] in a frame rotated by `angle` (radians, clockwise) about (hx, hy).
    fn rotated_rect(&mut self, hx: f64, hy: f64, angle: f64, u0: f64, len: f64, v0: f64, thick: f64, c: [u8; 3]) {
        let (s, co) = angle.sin_cos();
        let corners = [(u0, v0), (u0 + len, v0), (u0, v0 + thick), (u0 + len, v0 + thick)].map(|(u, v)| (hx + u * co - v * s, hy + u * s + v * co));
        let (xs, ys) = (corners.map(|p| p.0), corners.map(|p| p.1));
        let lo = |a: [f64; 4]| a.into_iter().fold(f64::INFINITY, f64::min);
        let hi = |a: [f64; 4]| a.into_iter().fold(f64::NEG_INFINITY, f64::max);
        self.shape(lo(xs), lo(ys), hi(xs), hi(ys), c, |x, y| {
            let (dx, dy) = (x - hx, y - hy);
            let (u, v) = (dx * co + dy * s, -dx * s + dy * co);
            u >= u0 && u <= u0 + len && v >= v0 && v <= v0 + thick
        });
    }
    fn text(&mut self, x: i64, y: i64, scale: i64, s: &str, c: [u8; 3]) {
        for (k, ch) in s.chars().enumerate() {
            let g = glyph(ch);
            for (row, bits) in g.iter().enumerate() {
                for col in 0..5 {
                    if bits >> (4 - col) & 1 == 1 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                self.put(x + (k as i64 * 6 + col) * scale + dx, y + row as i64 * scale + dy, c);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// 5x7 bitmap glyphs for the few characters the video prints.
fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        _ => [0; 7],
    }
}

fn mix(a: [u8; 3], b: [u8; 3], f: f64) -> [u8; 3] {
    let f = f.clamp(0.0, 1.0);
    [0, 1, 2].map(|i| (a[i] as f64 + (b[i] as f64 - a[i] as f64) * f).round() as u8)
}

/// Frame `k`, showing the model at `t` (its trigger's time, seconds since the recording started).
fn draw(m: &Model, c: &mut Canvas, k: usize, t: f64) {
    const BG: [u8; 3] = [28, 30, 34];
    const INK: [u8; 3] = [225, 225, 220];
    const DIM: [u8; 3] = [130, 132, 138];
    const BONE: [u8; 3] = [214, 190, 160];
    const TOOTH: [u8; 3] = [250, 250, 245];
    c.fill(BG);
    c.text(20, 18, 4, &format!("FRAME {k:05}"), INK);
    let time = format!("T {t:.3} S");
    c.text(W as i64 - 20 - 24 * time.len() as i64 + 4, 18, 4, &time, INK);
    c.rect(0.0, 62.0, W as f64, 2.0, [60, 62, 68]);

    // Jaw, side view: the lower jaw rotates about the hinge; muscles light up with their EMG.
    let jaw = m.jaw(t);
    let (hx, hy) = (60.0, 230.0);
    let angle = jaw.open * 0.45;
    let red = mix([70, 40, 40], [255, 70, 60], jaw.masseter * 1.4);
    let blue = mix([40, 50, 75], [80, 150, 255], jaw.digastric * 1.4);
    c.rotated_rect(hx, hy, angle, 70.0, 120.0, 34.0, 14.0, blue); // digastric, under the jaw
    c.rect(hx - 10.0, hy - 48.0, 230.0, 38.0, BONE); // upper jaw
    for i in 0..7 {
        c.rect(hx + 90.0 + i as f64 * 18.0, hy - 10.0, 12.0, 9.0, TOOTH);
    }
    c.rotated_rect(hx, hy, angle, -10.0, 230.0, 6.0, 26.0, BONE); // lower jaw
    for i in 0..7 {
        c.rotated_rect(hx, hy, angle, 90.0 + i as f64 * 18.0, 12.0, -3.0, 9.0, TOOTH);
    }
    // masseter: from the upper jaw to the lower jaw, just in front of the hinge
    let (sa, ca) = angle.sin_cos();
    let (lx, ly) = (hx + 55.0 * ca - 20.0 * sa, hy + 55.0 * sa + 20.0 * ca);
    c.ellipse((hx + 50.0 + lx) / 2.0, (hy - 30.0 + ly) / 2.0, 24.0, 18.0 + (ly - hy + 30.0) / 2.0, red);
    c.text(20, 330, 3, "JAW", DIM);
    c.text(20, 360, 2, "MASSETER", red);
    c.text(20, 384, 2, "DIGASTRIC", blue);

    // Stomach squeezes with the antral slow wave; the duodenum is a chain of rings with a lag.
    let (sq, _) = m.antrum(t, 0.0);
    let pink = [210, 120, 140];
    c.ellipse(470.0, 200.0, 100.0 * (1.0 - 0.28 * sq), 70.0 * (1.0 - 0.22 * sq), mix(pink, [255, 170, 190], sq));
    c.ellipse(470.0, 200.0, 40.0 * (1.0 - 0.28 * sq), 22.0 * (1.0 - 0.22 * sq), BG);
    c.text(400, 290, 3, "STOMACH", DIM);
    for i in 0..9 {
        let (d, _) = m.duodenum(t, i as f64 * 0.05);
        let r = 14.0 * (1.0 - 0.45 * d);
        c.ellipse(350.0 + i as f64 * 30.0, 380.0, r, r, mix([150, 110, 80], [240, 190, 120], d));
    }
    c.text(400, 420, 3, "DUODENUM", DIM);
    c.text(W as i64 - 70, H as i64 - 22, 2, "DEMO", DIM);
}

/// Packed RGB24 -> planar YUV 4:2:0, Y then U then V (BT.601 limited range, which FFmpeg assumes
/// when decoding it).
fn to_yuv420(rgb: &[u8]) -> Vec<u8> {
    let px = |x: usize, y: usize| {
        let i = 3 * (y * W + x);
        [rgb[i] as i32, rgb[i + 1] as i32, rgb[i + 2] as i32]
    };
    let mut out = Vec::with_capacity(W * H * 3 / 2);
    out.extend(rgb.chunks_exact(3).map(|p| ((66 * p[0] as i32 + 129 * p[1] as i32 + 25 * p[2] as i32 + 128 >> 8) + 16) as u8));
    for k in [[-38, -74, 112], [112, -94, -18]] {
        for y in 0..H / 2 {
            for x in 0..W / 2 {
                let mut acc = 0;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let [r, g, b] = px(2 * x + dx, 2 * y + dy);
                    acc += k[0] * r + k[1] * g + k[2] * b;
                }
                out.push(((acc + 512 >> 10) + 128).clamp(0, 255) as u8);
            }
        }
    }
    out
}

fn write_video(m: &Model, path: &Path) -> Result<()> {
    let mut octx = ff::format::output(&path)?;
    let codec = ff::encoder::find(ff::codec::Id::MPEG4).context("this FFmpeg has no MPEG-4 encoder")?;
    let global = octx.format().flags().contains(ff::format::Flags::GLOBAL_HEADER);
    let mut ost = octx.add_stream(codec)?;
    let mut enc = ff::codec::context::Context::new_with_codec(codec).encoder().video()?;
    let tb = ff::Rational::new(1, FPS as i32);
    enc.set_width(W as u32);
    enc.set_height(H as u32);
    enc.set_format(ff::format::Pixel::YUV420P);
    enc.set_time_base(tb);
    enc.set_frame_rate(Some(ff::Rational::new(FPS as i32, 1)));
    enc.set_gop(30);
    enc.set_bit_rate(700_000);
    let mut threads = ff::threading::Config::count(std::thread::available_parallelism().map_or(4, |n| n.get().min(8)));
    threads.kind = ff::threading::Type::Slice;
    enc.set_threading(threads);
    if global {
        enc.set_flags(ff::codec::Flags::GLOBAL_HEADER);
    }
    let mut enc = enc.open_as(codec)?;
    ost.set_parameters(&enc);
    ost.set_time_base(tb);
    octx.write_header()?;
    let ost_tb = octx.stream(0).context("no output stream")?.time_base();

    let mut yuv = ff::frame::Video::new(ff::format::Pixel::YUV420P, W as u32, H as u32);
    let mut packet = ff::Packet::empty();
    let mut drain = |enc: &mut ff::encoder::Video, octx: &mut ff::format::context::Output| -> Result<()> {
        while enc.receive_packet(&mut packet).is_ok() {
            packet.set_stream(0);
            packet.rescale_ts(tb, ost_tb);
            packet.write_interleaved(octx)?;
        }
        Ok(())
    };
    let rows = trigger_rows();
    let n_frames = rows.len() - 1; // the last trigger has no frame
    for batch in (0..n_frames).collect::<Vec<_>>().chunks(120) {
        let planes: Vec<Vec<u8>> = batch
            .par_iter()
            .map(|&k| {
                let mut canvas = Canvas(vec![0; W * H * 3]);
                draw(m, &mut canvas, k, rows[k] as f64 / FS);
                to_yuv420(&canvas.0)
            })
            .collect();
        for (&k, p) in batch.iter().zip(&planes) {
            let (y, uv) = p.split_at(W * H);
            let (u, v) = uv.split_at(W * H / 4);
            for (plane, src, w) in [(0, y, W), (1, u, W / 2), (2, v, W / 2)] {
                let stride = yuv.stride(plane);
                let dst = yuv.data_mut(plane);
                for (row, line) in src.chunks_exact(w).enumerate() {
                    dst[row * stride..row * stride + w].copy_from_slice(line);
                }
            }
            yuv.set_pts(Some(k as i64));
            enc.send_frame(&yuv)?;
            drain(&mut enc, &mut octx)?;
        }
        eprint!("\rsyncviewr: demo video {:.0}%", 100.0 * batch[0] as f64 / n_frames as f64);
    }
    enc.send_eof()?;
    drain(&mut enc, &mut octx)?;
    octx.write_trailer()?;
    eprintln!("\rsyncviewr: demo video 100%");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triggers_match_frames() {
        let r = trigger_rows();
        assert_eq!(r.len(), 8881); // 296 s at 30 frames/s, plus the end trigger
        assert_eq!(r[0], 20_000);
        assert!(r[..8880].windows(2).all(|w| (333..=334).contains(&(w[1] - w[0]))));
        assert_eq!(r[8880] - r[8879], 5000);
    }

    #[test]
    fn preset_parses() {
        let p: crate::preset::Preset = serde_json::from_value(preset()).unwrap();
        assert_eq!(p.channels.len(), 9);
        assert!(p.channels.iter().all(|s| CHANNELS.contains(&s.ch.as_str())));
    }
}
