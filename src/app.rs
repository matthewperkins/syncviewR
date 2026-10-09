//! The viewer window (port of syncview/app/main.py): toolbar, video, trace rows, overview strip,
//! channel table. Traces are drawn by the GPU line renderer in `gpu.rs`; everything else by egui.

use crate::cache::{CacheBuilder, Trace, TraceCache, WindowWorker};
use crate::filters::MODES;
use crate::oe::{check_sync, Recording};
use crate::preset::{self, Preset, Spec};
use crate::video::{video_duration, VideoDecoder};
use crate::gpu::{offsets_for, LineStrip};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use eframe::egui_wgpu;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const RATES: [f64; 10] = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0];
const TIME_BASES: [f64; 16] = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0, 7200.0];
const WINDOW_MAX_S: f64 = 180.0; // on-demand (uncached) processing only for views up to this span
const PALETTE: [&str; 6] = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"];
pub const BG: Color32 = Color32::from_rgb(0x11, 0x11, 0x11);
pub const FG: Color32 = Color32::from_rgb(0xc3, 0xc2, 0xb7);
pub const LABEL: Color32 = Color32::from_rgb(0xe8, 0xe8, 0xe4);
pub const MUTED: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);
const OVERVIEW_SLOT: usize = 10_000;

// layout of the trace area (points), as in the Python viewer
const MARGIN_X: f32 = 9.0;
const LABEL_W: f32 = 130.0;
const YAXIS_W: f32 = 60.0;
const ROW_GAP: f32 = 2.0;
const TAXIS_H: f32 = 20.0;

pub struct Options {
    pub preset: Preset,
    pub video: Option<PathBuf>,
    pub trigger_line: i64,
    pub start_time: Option<f64>,
    pub time_base: Option<f64>,
    pub play: bool,
}

const EXTRA_KEYS: [&str; 5] = ["order", "smooth_ms", "env_lp", "plot_fs", "win_s"];

/// Python's "{:g}": up to 6 significant digits, trailing zeros removed.
fn g6(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let e = x.abs().log10().floor() as i32;
    if !(-5..6).contains(&e) {
        return crate::filters::py_float(x);
    }
    let d = (5 - e).max(0) as usize;
    let s = format!("{x:.d$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

fn fmt_ylim(y: Option<[f64; 2]>) -> String {
    match y {
        None => "auto".into(),
        Some([a, b]) => format!("{}, {}", g6(a), g6(b)),
    }
}

/// Text-field contents of one channel-table row; parsed back into the spec on Enter / focus loss.
struct RowEdit {
    lo: String,
    hi: String,
    notch: String,
    extra: String,
    y: String,
}

impl RowEdit {
    fn from_spec(s: &Spec) -> Self {
        let p = crate::filters::params(s).unwrap_or_default();
        let band = p.get("band").and_then(|b| b.as_array()).cloned().unwrap_or_default();
        let edge = |i: usize| band.get(i).and_then(|v| v.as_f64()).map(g6).unwrap_or_default();
        let notch = match p.get("notch") {
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|v| v.as_f64()).map(g6).collect::<Vec<_>>().join(","),
            Some(v) => v.as_f64().map(g6).unwrap_or_default(),
            None => String::new(),
        };
        let extra = EXTRA_KEYS
            .iter()
            .filter_map(|k| s.extra.get(*k).and_then(|v| v.as_f64()).map(|v| format!("{k}={}", g6(v))))
            .collect::<Vec<_>>()
            .join(" ");
        RowEdit { lo: edge(0), hi: edge(1), notch, extra, y: fmt_ylim(s.ylim) }
    }

    /// Parse the fields into `s` (fields that don't parse are left as they were).
    fn apply(&self, s: &mut Spec) {
        use serde_json::{json, Value};
        let num = |t: &str| -> Result<Option<f64>, ()> {
            let t = t.trim();
            if t.is_empty() || ["none", "-", "–"].contains(&t.to_lowercase().as_str()) {
                Ok(None)
            } else {
                t.parse::<f64>().map(Some).map_err(|_| ())
            }
        };
        if let (Ok(lo), Ok(hi)) = (num(&self.lo), num(&self.hi)) {
            let current = RowEdit::from_spec(s);
            if self.lo.trim() != current.lo || self.hi.trim() != current.hi {
                s.extra.insert("band".into(), json!([lo, hi]));
            }
        }
        let notch: Result<Vec<f64>, _> =
            self.notch.replace(';', ",").split(',').map(str::trim).filter(|v| !v.is_empty()).map(str::parse::<f64>).collect();
        if let Ok(n) = notch {
            if n.is_empty() {
                s.extra.remove("notch");
            } else {
                s.extra.insert("notch".into(), json!(n));
            }
        }
        let mut seen: Vec<&str> = vec![];
        let text = self.extra.replace(',', " ");
        for kv in text.split_whitespace() {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            if let (Some(k), Ok(v)) = (EXTRA_KEYS.iter().find(|x| **x == k), v.parse::<f64>()) {
                let val = if *k == "order" { json!(v.round() as i64) } else { Value::from(v) };
                s.extra.insert(k.to_string(), val);
                seen.push(k);
            }
        }
        for k in EXTRA_KEYS {
            if !seen.contains(&k) {
                s.extra.remove(k);
            }
        }
        let y: Vec<&str> = self.y.split([',', ';']).map(str::trim).collect();
        if self.y.trim().eq_ignore_ascii_case("auto") || self.y.trim().is_empty() {
            s.ylim = None;
        } else if let [a, b] = y[..] {
            if let (Ok(a), Ok(b)) = (a.parse::<f64>(), b.parse::<f64>()) {
                if a < b {
                    s.ylim = Some([a, b]);
                }
            }
        }
    }
}

struct Row {
    spec: Spec,
    key: String,
    color: Color32,
    auto: Option<(f64, f64)>,
    table_index: usize,
}

pub struct App {
    rec: Arc<Recording>,
    cache: Arc<TraceCache>,
    builder: CacheBuilder,
    window_worker: WindowWorker,
    specs: Vec<Spec>,
    overview: Option<Spec>,
    overview_key: Option<String>,
    rows: Vec<Row>,
    t: f64,
    span: f64,
    rate_i: usize,
    playing: bool,
    last_tick: Option<Instant>,
    trigger_line: i64,
    frames: Vec<f64>,
    frame_period: f64,
    video: Option<VideoDecoder>,
    video_ready: bool,
    video_tex: Option<egui::TextureHandle>,
    video_size: [usize; 2],
    video_caption: String,
    video_msg: String,
    want_frame: Option<usize>,
    edits: Vec<RowEdit>,
    selected: Option<usize>,
    last_dir: Option<PathBuf>,
    span_text: String,
    goto_text: String,
    status: String,
    dialogs: Vec<(String, String)>,
    was_building: bool,
    fps: Option<(Instant, Vec<f64>)>, // SYNCVIEWR_FPS=1: per-second frame-time report
}

fn hex(s: &str) -> Color32 {
    Color32::from_hex(s).unwrap_or(FG)
}

pub fn fmt_time(t: f64, span: Option<f64>) -> String {
    let sign = if t < 0.0 { "-" } else { "" };
    let t = t.abs();
    let h = (t / 3600.0).floor();
    let rem = t - h * 3600.0;
    let m = (rem / 60.0).floor();
    let s = rem - m * 60.0;
    let dec = match span {
        None => 3,
        Some(sp) if sp < 10.0 => 3,
        Some(sp) if sp < 300.0 => 1,
        _ => 0,
    };
    let width = if dec > 0 { 3 + dec } else { 2 };
    let sec = format!("{s:0width$.dec$}");
    if h > 0.0 { format!("{sign}{}:{:02}:{sec}", h as i64, m as i64) } else { format!("{sign}{}:{sec}", m as i64) }
}

/// Python's "%.3g" for the magnitudes used here.
fn g3(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let d = (2 - x.abs().log10().floor() as i32).max(0) as usize;
    let s = format!("{x:.d$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

fn fmt_span(s: f64) -> String {
    if s < 60.0 {
        format!("{} s", g3(s))
    } else if s < 3600.0 {
        format!("{} min", g3(s / 60.0))
    } else {
        format!("{} h", g3(s / 3600.0))
    }
}

fn parse_span(text: &str) -> Option<f64> {
    let t: String = text.trim().to_lowercase().chars().filter(|c| !c.is_whitespace()).collect();
    for (suf, mul) in [("min", 60.0), ("h", 3600.0), ("ms", 1e-3), ("s", 1.0), ("m", 60.0)] {
        if let Some(v) = t.strip_suffix(suf) {
            return v.parse::<f64>().ok().map(|v| v * mul);
        }
    }
    t.parse().ok()
}

const NICE_STEPS: [f64; 23] = [1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0,
    300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0];

fn nice_spacing(span: f64, n_major: f64) -> (f64, f64) {
    let major = NICE_STEPS.iter().copied().find(|s| span / s <= n_major).unwrap_or(7200.0);
    let minor = NICE_STEPS
        .iter()
        .rev()
        .copied()
        .find(|s| *s < major && [2.0, 2.5, 3.0, 4.0, 5.0, 6.0].iter().any(|r| ((major / s) - r).abs() < 1e-9))
        .unwrap_or(major / 5.0);
    (major, minor)
}

fn fmt_tick(v: f64, step: f64) -> String {
    if v.abs() < step * 1e-6 {
        return "0".into();
    }
    let mut d = 0;
    while d < 6 && ((step * 10f64.powi(d)) - (step * 10f64.powi(d)).round()).abs() > 1e-6 {
        d += 1;
    }
    format!("{v:.prec$}", prec = d as usize)
}

impl App {
    /// The viewer for `rec`. The GPU line renderer must already be set up (`gpu::init`).
    pub fn new(ctx: &egui::Context, rec: Arc<Recording>, cache: Arc<TraceCache>, opts: Options) -> Self {
        let ctx = ctx.clone();
        let c1 = ctx.clone();
        let builder = CacheBuilder::new(cache.clone(), move || c1.request_repaint());
        let c2 = ctx.clone();
        let window_worker = WindowWorker::new(cache.clone(), move || c2.request_repaint());
        let frames: Vec<f64> = rec.rising_edges(opts.trigger_line).iter().map(|i| *i as f64 / rec.fs).collect();
        let t = opts.start_time.unwrap_or(frames.first().copied().unwrap_or(0.0)).clamp(0.0, rec.duration());
        let mut app = App {
            builder,
            window_worker,
            specs: vec![],
            overview: None,
            overview_key: None,
            rows: vec![],
            t,
            span: 10.0,
            rate_i: 3,
            playing: false,
            last_tick: None,
            trigger_line: opts.trigger_line,
            frames,
            frame_period: 0.02,
            video: None,
            video_ready: false,
            video_tex: None,
            video_size: [0, 0],
            video_caption: String::new(),
            video_msg: String::new(),
            want_frame: None,
            edits: vec![],
            selected: None,
            last_dir: None,
            span_text: String::new(),
            goto_text: String::new(),
            status: String::new(),
            dialogs: vec![],
            was_building: false,
            fps: std::env::var_os("SYNCVIEWR_FPS").map(|_| (Instant::now(), vec![])),
            rec,
            cache,
        };
        app.apply_preset(opts.preset);
        if let Some(tb) = opts.time_base {
            app.set_span(tb);
        }
        if let Some(v) = opts.video {
            app.attach_video(v, &ctx);
        }
        if opts.play {
            app.toggle_play();
        }
        app
    }

    fn apply_preset(&mut self, p: Preset) {
        let (mut p, mut problems) = preset::check(p, &self.rec);
        if !problems.is_empty() {
            for m in &problems {
                eprintln!("syncviewr: preset: {m}");
            }
            if p.channels.is_empty() {
                problems.push("No usable rows left - showing the recording's own channels instead.".into());
                p = preset::from_recording(&self.rec);
            }
            let body = problems.iter().map(|m| format!("• {m}")).collect::<Vec<_>>().join("\n");
            self.dialogs.push(("Channel preset doesn't match this recording".into(), body));
        }
        self.set_span(p.time_base.unwrap_or(self.span));
        self.overview = p.overview;
        self.specs = p.channels;
        self.selected = None;
        self.sync_edits();
        self.set_specs();
    }

    fn set_specs(&mut self) {
        if self.overview.is_none() {
            // default overview: slow-wave power of the first slow row
            if let Some(s) = self.specs.iter().find(|s| s.mode == "slow") {
                let mut o = Spec::new(&s.ch, "bandpower");
                o.reference = s.reference.clone();
                o.label = Some(format!("{} slow-wave power", s.title()));
                self.overview = Some(o);
            }
        }
        self.overview_key = self.overview.as_ref().map(|o| self.cache.key(o));
        let old: std::collections::HashMap<String, Option<(f64, f64)>> = self.rows.iter().map(|r| (r.key.clone(), r.auto)).collect();
        self.rows = self
            .specs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.show)
            .enumerate()
            .map(|(i, (ti, s))| {
                let key = self.cache.key(s);
                Row {
                    color: s.color.as_deref().map(hex).unwrap_or_else(|| hex(PALETTE[i % PALETTE.len()])),
                    auto: old.get(&key).copied().flatten(),
                    key,
                    spec: s.clone(),
                    table_index: ti,
                }
            })
            .collect();
        // only build what is needed; the overview trace goes first
        let mut want: Vec<Spec> = self.overview.iter().cloned().collect();
        want.extend(self.rows.iter().map(|r| r.spec.clone()));
        self.cache.retain(&want);
        self.builder.request(want, &self.cache);
        let keys: std::collections::HashSet<&String> = self.rows.iter().map(|r| &r.key).collect();
        self.window_worker.results.lock().unwrap().retain(|k, _| keys.contains(k));
    }

    // ------------------------------------------------------------------ clock
    fn set_time(&mut self, t: f64) {
        self.t = t.clamp(0.0, self.rec.duration());
    }

    fn set_span(&mut self, s: f64) {
        self.span = s.clamp(0.05, self.rec.duration().max(0.05));
        self.span_text = fmt_span((self.span * 1000.0).round() / 1000.0);
    }

    fn step_frames(&mut self, n: i64) {
        if self.frames.is_empty() {
            self.set_time(self.t + n as f64 * 0.02);
            return;
        }
        let k = self.frames.partition_point(|x| *x < self.t + 1e-6) as i64 - 1;
        let k = (k + n).clamp(0, self.frames.len() as i64 - 1) as usize;
        self.set_time(self.frames[k]);
    }

    /// Last frame triggered at or before the cursor (None outside the video).
    fn current_frame(&self) -> Option<usize> {
        let last = *self.frames.last()?;
        if self.t > last + self.frame_period {
            return None;
        }
        let k = self.frames.partition_point(|x| *x < self.t + 1e-6) as i64 - 1;
        (k >= 0).then_some(k as usize)
    }

    fn toggle_play(&mut self) {
        self.playing = !self.playing;
        self.last_tick = self.playing.then(Instant::now);
    }

    fn tick(&mut self) {
        if !self.playing {
            return;
        }
        let now = Instant::now();
        let dt = self.last_tick.map_or(0.0, |l| (now - l).as_secs_f64());
        self.last_tick = Some(now);
        if self.t >= self.rec.duration() {
            self.toggle_play();
            return;
        }
        self.set_time(self.t + dt * RATES[self.rate_i]);
    }

    // ------------------------------------------------------------------ video
    pub fn attach_video(&mut self, path: PathBuf, ctx: &egui::Context) {
        let c = ctx.clone();
        self.video_msg = format!("opening {} …", path.file_name().unwrap_or_default().to_string_lossy());
        self.video_ready = false;
        self.video = Some(VideoDecoder::new(path, self.cache.root.join("video_index"), move || c.request_repaint()));
    }

    fn poll_video(&mut self, ctx: &egui::Context) {
        let Some(v) = &self.video else { return };
        let name = v.path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        match v.take_opened() {
            Some(Ok(info)) => {
                match check_sync(&self.rec, info.n_frames, self.trigger_line, video_duration(&v.path)) {
                    Err(e) => {
                        eprintln!("syncviewr: sync error ({name}): {e}");
                        self.video_msg = format!("{name}: {e}");
                        self.dialogs.push(("Video sync failed".into(), format!("{name} on TTL line {}:\n\n• {e}", self.trigger_line)));
                        self.video = None;
                        return;
                    }
                    Ok(si) => {
                        for m in &si.issues {
                            eprintln!("syncviewr: sync warning ({name}): {m}");
                        }
                        if !si.issues.is_empty() {
                            let body = si.issues.iter().map(|m| format!("• {m}")).collect::<Vec<_>>().join("\n\n");
                            self.dialogs.push(("Video sync warning".into(), format!("{name} on TTL line {}:\n\n{body}", self.trigger_line)));
                        }
                        self.frames = si.frames.iter().map(|i| *i as f64 / self.rec.fs).collect();
                        self.frame_period = si.period;
                        self.status = format!(
                            "{name}: {} frames {}×{}, {} extra trigger(s) dropped; video spans {} – {}; decoding on CPU (FFmpeg)",
                            info.n_frames, info.width, info.height, si.extra,
                            fmt_time(self.frames[0], None), fmt_time(*self.frames.last().unwrap(), None)
                        );
                        if self.t < self.frames[0] || self.t > *self.frames.last().unwrap() {
                            self.t = self.frames[0];
                        }
                        self.video_ready = true;
                        self.want_frame = None;
                    }
                }
            }
            Some(Err(msg)) => {
                eprintln!("syncviewr: {msg}");
                self.video_msg = msg;
            }
            None => {}
        }
        if let Some(f) = self.video.as_ref().and_then(|v| v.take_frame()) {
            let img = egui::ColorImage::from_rgb([f.width, f.height], &f.rgb);
            match &mut self.video_tex {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => self.video_tex = Some(ctx.load_texture("video", img, egui::TextureOptions::LINEAR)),
            }
            self.video_size = [f.width, f.height];
            let trig = self.frames.get(f.index).map(|t| fmt_time(*t, None)).unwrap_or_default();
            self.video_caption = format!("frame {}   trigger {}", f.index, trig);
        }
        if self.video_ready {
            match self.current_frame() {
                None => {
                    self.want_frame = None;
                    self.video_msg = "no video at this time".into();
                }
                Some(k) => {
                    self.video_msg.clear();
                    if Some(k) != self.want_frame {
                        self.want_frame = Some(k);
                        self.video.as_ref().unwrap().request(k);
                    }
                }
            }
        }
    }

    fn video_ui(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        ui.allocate_rect(rect, Sense::hover());
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 0.0, Color32::BLACK);
        if !self.video_msg.is_empty() || self.video_tex.is_none() {
            let msg = if self.video_msg.is_empty() { "…" } else { &self.video_msg };
            p.text(rect.center(), Align2::CENTER_CENTER, msg, FontId::proportional(13.0), MUTED);
            return;
        }
        let tex = self.video_tex.as_ref().unwrap();
        let [w, h] = self.video_size;
        let scale = (rect.width() / w as f32).min(rect.height() / h as f32);
        let size = Vec2::new(w as f32 * scale, h as f32 * scale);
        let target = Rect::from_center_size(rect.center(), size);
        p.image(tex.id(), target, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        p.text(target.left_bottom() + Vec2::new(8.0, -6.0), Align2::LEFT_BOTTOM, &self.video_caption, FontId::monospace(11.0), LABEL);
    }

    // ------------------------------------------------------------------ data
    fn trace_for(&self, row: &Row) -> Option<Arc<Trace>> {
        if let Some(t) = self.cache.get(&row.spec) {
            return Some(t);
        }
        let res = self.window_worker.results.lock().unwrap();
        let (lo, hi, tr) = res.get(&row.key)?;
        (*lo <= self.t - self.span / 2.0 && self.t + self.span / 2.0 <= *hi).then(|| tr.clone())
    }

    fn auto_ylim(mode: &str, tr: &Trace) -> (f64, f64) {
        if tr.stats.is_empty() {
            return (-1.0, 1.0);
        }
        let s = |k: &str| tr.stat(k).unwrap_or(1.0);
        match mode {
            "hilo" => {
                let m = s("absmax99.9") * 2.0;
                (-m, m)
            }
            "envelope" | "bandpower" => (0.0, s("99.9") * 1.2),
            _ => {
                let (lo, hi) = (s("0.5"), s("99.5"));
                let r = hi - lo;
                (lo - 0.1 * r, hi + 0.1 * r)
            }
        }
    }

    fn row_ylim(&mut self, i: usize, tr: Option<&Trace>) -> (f64, f64) {
        let row = &mut self.rows[i];
        if let Some([lo, hi]) = row.spec.ylim {
            return (lo, hi);
        }
        if row.auto.is_none() {
            if let Some(tr) = tr {
                row.auto = Some(Self::auto_ylim(&row.spec.mode, tr));
            }
        }
        row.auto.unwrap_or((-1.0, 1.0))
    }

    fn scale_row_y(&mut self, i: usize, f: f64) {
        let tr = self.trace_for(&self.rows[i]);
        let (lo, hi) = self.row_ylim(i, tr.as_deref());
        let row = &mut self.rows[i];
        let new = if matches!(row.spec.mode.as_str(), "envelope" | "bandpower") {
            [lo, lo + (hi - lo) * f]
        } else {
            let c = (lo + hi) / 2.0;
            [c - (hi - lo) / 2.0 * f, c + (hi - lo) / 2.0 * f]
        };
        row.spec.ylim = Some(new);
        self.specs[row.table_index].ylim = Some(new);
        self.edits[row.table_index].y = fmt_ylim(Some(new));
    }

    fn reset_row_y(&mut self, i: usize) {
        let row = &mut self.rows[i];
        row.spec.ylim = None;
        row.auto = None;
        self.specs[row.table_index].ylim = None;
        self.edits[row.table_index].y = fmt_ylim(None);
    }

    // ------------------------------------------------------------------ input
    fn keys(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        use egui::Key;
        let (pressed, shift) = ctx.input(|i| {
            let keys = [Key::ArrowLeft, Key::ArrowRight, Key::PageUp, Key::PageDown, Key::Home, Key::End, Key::Space,
                        Key::OpenBracket, Key::CloseBracket, Key::Plus, Key::Equals, Key::Minus];
            (keys.into_iter().filter(|k| i.key_pressed(*k)).collect::<Vec<_>>(), i.modifiers.shift)
        });
        for k in pressed {
            match k {
                Key::ArrowLeft | Key::ArrowRight => {
                    let d = if k == Key::ArrowLeft { -1 } else { 1 };
                    if shift { self.set_time(self.t + d as f64 * 0.1 * self.span) } else { self.step_frames(d) }
                }
                Key::PageUp => self.set_time(self.t - self.span),
                Key::PageDown => self.set_time(self.t + self.span),
                Key::Home => self.set_time(0.0),
                Key::End => self.set_time(self.rec.duration()),
                Key::Space => self.toggle_play(),
                Key::OpenBracket => self.rate_i = self.rate_i.saturating_sub(1),
                Key::CloseBracket => self.rate_i = (self.rate_i + 1).min(RATES.len() - 1),
                Key::Plus | Key::Equals => self.set_span(self.span * 0.8),
                Key::Minus => self.set_span(self.span / 0.8),
                _ => {}
            }
        }
    }

    /// Mouse wheel / trackpad over the trace rows: zoom time, pan, or (⌘/Ctrl) scale one row's Y.
    fn wheel(&mut self, ctx: &egui::Context, row: Option<usize>, plot_w: f32) {
        let events = ctx.input(|i| i.events.clone());
        for e in events {
            let egui::Event::MouseWheel { unit, delta, modifiers, .. } = e else { continue };
            let per_step = match unit {
                egui::MouseWheelUnit::Line => 1.0,
                egui::MouseWheelUnit::Point => 1.0 / 50.0,
                egui::MouseWheelUnit::Page => 10.0,
            };
            let (sx, sy) = (delta.x as f64 * per_step, delta.y as f64 * per_step);
            if modifiers.command {
                if let Some(r) = row {
                    self.scale_row_y(r, 0.8f64.powf(sy));
                }
            } else if sx.abs() > sy.abs() {
                // sideways swipe: move the data with the fingers
                let dt = if unit == egui::MouseWheelUnit::Point { delta.x as f64 * self.span / plot_w as f64 } else { sx * 0.1 * self.span };
                self.set_time(self.t - dt);
            } else if modifiers.shift {
                self.set_time(self.t - sy * 0.1 * self.span);
            } else if sy != 0.0 {
                self.set_span(self.span * 0.8f64.powf(sy));
            }
        }
    }

    // ------------------------------------------------------------------ panels
    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.add_sized([36.0, 22.0], egui::Button::new(if self.playing { "⏸" } else { "▶" })).clicked() {
                self.toggle_play();
            }
            egui::ComboBox::from_id_salt("rate")
                .width(60.0)
                .selected_text(format!("{}×", RATES[self.rate_i]))
                .show_ui(ui, |ui| {
                    for (i, r) in RATES.iter().enumerate() {
                        ui.selectable_value(&mut self.rate_i, i, format!("{r}×"));
                    }
                });
            ui.separator();
            ui.label("time base");
            let resp = ui.add(egui::TextEdit::singleline(&mut self.span_text).desired_width(70.0));
            if resp.lost_focus() {
                if let Some(s) = parse_span(&self.span_text) {
                    self.set_span(s);
                } else {
                    self.set_span(self.span);
                }
            }
            egui::ComboBox::from_id_salt("span").width(20.0).selected_text("").show_ui(ui, |ui| {
                for s in TIME_BASES {
                    if ui.selectable_label(false, fmt_span(s)).clicked() {
                        self.set_span(s);
                    }
                }
            });
            ui.separator();
            ui.label("go to");
            let resp = ui.add(egui::TextEdit::singleline(&mut self.goto_text).hint_text("h:mm:ss or seconds").desired_width(130.0));
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                let parts: Option<Vec<f64>> = self.goto_text.trim().split(':').map(|p| p.parse().ok()).collect();
                if let Some(parts) = parts {
                    let t = parts.iter().rev().enumerate().map(|(i, v)| v * 60f64.powi(i as i32)).sum();
                    self.set_time(t);
                }
            }
            ui.separator();
            if ui.button("Video…").on_hover_text("attach the video recorded during this session").clicked() {
                let ctx = ui.ctx().clone();
                self.choose_video_dialog(&ctx);
            }
            ui.separator();
            let fr = self.current_frame().map(|k| format!("   frame {k}")).unwrap_or_default();
            ui.label(egui::RichText::new(format!("t = {}  ({:.3} s){fr}", fmt_time(self.t, None), self.t)).monospace().size(13.0));
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let building = *self.builder.progress.lock().unwrap();
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(f) = building {
                    ui.add(egui::ProgressBar::new(f as f32).desired_width(220.0).text(format!("filtering session… {:.0}%", f * 100.0)));
                }
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(&self.status).size(12.0));
                });
            });
        });
    }

    // ------------------------------------------------------------------ channel table

    /// Rebuild the table's text fields from the specs (after loading, reordering, ⌘+scroll, …).
    fn sync_edits(&mut self) {
        self.edits = self.specs.iter().map(RowEdit::from_spec).collect();
    }

    fn preset(&self) -> Preset {
        Preset { time_base: Some(self.span), channels: self.specs.clone(), overview: self.overview.clone() }
    }

    fn load_preset_dialog(&mut self) {
        let mut d = rfd::FileDialog::new().set_title("Load channel preset").add_filter("JSON", &["json"]);
        if let Some(dir) = &self.last_dir {
            d = d.set_directory(dir);
        }
        let Some(path) = d.pick_file() else { return };
        self.last_dir = path.parent().map(PathBuf::from);
        let parsed = std::fs::read_to_string(&path)
            .map_err(anyhow::Error::from)
            .and_then(|t| serde_json::from_str::<Preset>(&t).map_err(anyhow::Error::from));
        match parsed {
            Ok(p) => {
                self.overview = None;
                self.apply_preset(p);
                self.status = format!("loaded preset {}", path.display());
            }
            Err(e) => self.dialogs.push(("Could not load preset".into(), format!("{}:\n\n{e:#}", path.display()))),
        }
    }

    fn save_preset_dialog(&mut self) {
        let mut d = rfd::FileDialog::new().set_title("Save channel preset").add_filter("JSON", &["json"]).set_file_name("preset.json");
        if let Some(dir) = &self.last_dir {
            d = d.set_directory(dir);
        }
        let Some(mut path) = d.save_file() else { return };
        if path.extension().is_none_or(|e| e != "json") {
            path.set_extension("json");
        }
        self.last_dir = path.parent().map(PathBuf::from);
        let res = serde_json::to_string_pretty(&self.preset()).map_err(anyhow::Error::from)
            .and_then(|t| std::fs::write(&path, t + "\n").map_err(anyhow::Error::from));
        match res {
            Ok(()) => self.status = format!("saved preset {}", path.display()),
            Err(e) => self.dialogs.push(("Could not save preset".into(), format!("{}:\n\n{e:#}", path.display()))),
        }
    }

    fn choose_video_dialog(&mut self, ctx: &egui::Context) {
        let mut d = rfd::FileDialog::new().set_title("Video recorded during this session").add_filter("Video", &["mp4", "mkv", "mov", "avi"]);
        if let Some(dir) = self.video.as_ref().and_then(|v| v.path.parent().map(PathBuf::from)).or(self.last_dir.clone()) {
            d = d.set_directory(dir);
        }
        if let Some(path) = d.pick_file() {
            self.video_tex = None;
            self.attach_video(path, ctx);
        }
    }

    fn channels_ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.label("Channels").on_hover_text(
                "Low/High Hz: filter band (blank = none). Notch: e.g. 60 or 60,180.\n\
                 DSP wzrd: key=value for order, smooth_ms, env_lp, plot_fs, win_s.\n\
                 Y range: 'auto' or 'lo, hi'. Text fields apply on Enter or when you click away.\n\
                 ⌘/Ctrl+scroll over a trace scales its Y range; double-click a trace resets it to auto.\n\
                 Click a row's number to select it for Up / Down / Remove, and as the template for Add.",
            )
        });
        ui.separator();
        let mut changed = false;
        let mut action: Option<&str> = None;
        egui::Panel::bottom("channel_buttons").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                for (text, tip) in [("Add", "add a row (a copy of the selected one)"), ("Remove", "remove the selected row"),
                                    ("Up", "move the selected row up"), ("Down", "move the selected row down"),
                                    ("Load…", "load a channel preset (.json)"), ("Save…", "save rows, time base and overview as a preset")] {
                    if ui.button(text).on_hover_text(tip).clicked() {
                        action = Some(text);
                    }
                }
            });
        });
        let names = self.rec.ch_names.clone();
        let mut refs = vec!["—".to_string()];
        refs.extend(names.iter().cloned());
        egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
            egui::Grid::new("channels").striped(true).spacing([4.0, 5.0]).show(ui, |ui| {
                for h in ["", "Show", "Label", "Ch", "Ref", "Mode", "Low Hz", "High Hz", "Notch", "DSP wzrd", "Y range"] {
                    let l = ui.label(egui::RichText::new(h).small());
                    if h == "DSP wzrd" {
                        l.on_hover_text(
                            "key=value, space separated; blank = the mode's defaults.\n\
                             order      Butterworth order of the Low/High Hz band (all modes; 4 hilo/envelope, 2 slow/bandpower)\n\
                             smooth_ms  envelope: moving-average window after rectifying (20)\n\
                             env_lp     envelope: final low-pass, Hz (40)\n\
                             plot_fs    slow/envelope/bandpower: stored & drawn sample rate, Hz (1000/1000/10)\n\
                             win_s      bandpower: RMS window, s (30)",
                        );
                    }
                }
                ui.end_row();
                for i in 0..self.specs.len() {
                    if ui.selectable_label(self.selected == Some(i), format!("{:>2}", i + 1)).clicked() {
                        self.selected = if self.selected == Some(i) { None } else { Some(i) };
                    }
                    let s = &mut self.specs[i];
                    let e = &mut self.edits[i];
                    changed |= ui.checkbox(&mut s.show, "").changed();
                    let mut label = s.label.clone().unwrap_or_default();
                    if ui.add_sized([92.0, 20.0], egui::TextEdit::singleline(&mut label)).changed() {
                        s.label = Some(label);
                        changed = true;
                    }
                    egui::ComboBox::from_id_salt(("ch", i)).width(54.0).selected_text(&s.ch).show_ui(ui, |ui| {
                        for n in &names {
                            changed |= ui.selectable_value(&mut s.ch, n.clone(), n).changed();
                        }
                    });
                    let mut r = s.reference.clone().unwrap_or_else(|| "—".into());
                    egui::ComboBox::from_id_salt(("ref", i)).width(54.0).selected_text(&r).show_ui(ui, |ui| {
                        for n in &refs {
                            if ui.selectable_value(&mut r, n.clone(), n).changed() {
                                changed = true;
                            }
                        }
                    });
                    s.reference = (r != "—").then_some(r);
                    let before = s.mode.clone();
                    egui::ComboBox::from_id_salt(("mode", i)).width(78.0).selected_text(&s.mode).show_ui(ui, |ui| {
                        for m in MODES {
                            ui.selectable_value(&mut s.mode, m.to_string(), m);
                        }
                    });
                    if s.mode != before {
                        // band, order etc. belong to the mode: start the new mode from its defaults
                        s.extra.remove("band");
                        for k in EXTRA_KEYS {
                            s.extra.remove(k);
                        }
                        s.ylim = None;
                        *e = RowEdit::from_spec(s);
                        changed = true;
                    }
                    let mut commit = false;
                    for (buf, w) in [(&mut e.lo, 42.0), (&mut e.hi, 42.0), (&mut e.notch, 50.0), (&mut e.extra, 92.0), (&mut e.y, 76.0)] {
                        let resp = ui.add_sized([w, 20.0], egui::TextEdit::singleline(buf));
                        commit |= resp.lost_focus();
                    }
                    if commit {
                        let before = s.clone();
                        e.apply(s);
                        *e = RowEdit::from_spec(s); // show what was understood (invalid input reverts)
                        changed |= *s != before;
                    }
                    ui.end_row();
                }
            });
        });
        match action {
            Some("Add") => {
                let mut s = self.selected.and_then(|i| self.specs.get(i).cloned()).unwrap_or_else(|| Spec::new(&names[0], "slow"));
                s.ylim = None;
                s.show = true;
                self.specs.push(s);
                self.selected = Some(self.specs.len() - 1);
                changed = true;
            }
            Some("Remove") => {
                if let Some(i) = self.selected.filter(|i| *i < self.specs.len()) {
                    self.specs.remove(i);
                    self.selected = None;
                    changed = true;
                }
            }
            Some(dir @ ("Up" | "Down")) => {
                if let Some(i) = self.selected {
                    let j = if dir == "Up" { i.checked_sub(1) } else { Some(i + 1).filter(|j| *j < self.specs.len()) };
                    if let Some(j) = j {
                        self.specs.swap(i, j);
                        self.selected = Some(j);
                        changed = true;
                    }
                }
            }
            Some("Load…") => self.load_preset_dialog(),
            Some("Save…") => self.save_preset_dialog(),
            _ => {}
        }
        if changed {
            self.sync_edits();
            self.set_specs();
        }
    }

    fn traces_ui(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, BG);
        let ppp = ui.ctx().pixels_per_point();
        let x0 = rect.left() + MARGIN_X + LABEL_W + YAXIS_W;
        let x1 = rect.right() - MARGIN_X;
        let top = rect.top() + 4.0;
        let bottom = rect.bottom() - 2.0 - TAXIS_H;
        let n = self.rows.len();
        if n == 0 || x1 <= x0 + 10.0 || bottom <= top + 10.0 {
            painter.text(rect.center(), Align2::CENTER_CENTER, "no channels shown (tick some in the Channels panel)", FontId::proportional(13.0), MUTED);
            return;
        }
        let row_h = ((bottom - top) - ROW_GAP * (n as f32 - 1.0)) / n as f32;
        let plot_w = x1 - x0;
        let (t, span) = (self.t, self.span);
        let (t_lo, t_hi) = (t - span / 2.0, t + span / 2.0);
        let mut need_window = vec![];
        let font = FontId::proportional(12.0);
        let small = FontId::proportional(11.0);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..n {
            let r = Rect::from_min_max(Pos2::new(x0, top + i as f32 * (row_h + ROW_GAP)), Pos2::new(x1, top + i as f32 * (row_h + ROW_GAP) + row_h));
            painter.text(Pos2::new(x0 - YAXIS_W - 4.0, r.center().y), Align2::RIGHT_CENTER, self.rows[i].spec.title(), font.clone(), LABEL);
            let tr = self.trace_for(&self.rows[i]);
            let (lo, hi) = self.row_ylim(i, tr.as_deref());
            // y axis: few, round ticks so stacked rows' labels don't collide
            let ax = x0 - 2.0;
            painter.line_segment([Pos2::new(ax, r.top()), Pos2::new(ax, r.bottom())], Stroke::new(1.0, FG));
            let ymap = |v: f64| r.bottom() - ((v - lo) / (hi - lo)) as f32 * r.height();
            let ticks: Vec<(f64, f64)> = if r.height() < 50.0 {
                if lo <= 0.0 && 0.0 <= hi { vec![(0.0, 1.0)] } else { vec![] }
            } else {
                let target = 0.95 * (hi - lo) / 2.0;
                let e = 10f64.powf(target.log10().floor());
                let step = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0].iter().map(|m| m * e).filter(|s| *s <= target).fold(e, f64::max);
                let k0 = (lo / step).ceil() as i64;
                let k1 = (hi / step).floor() as i64;
                (k0..=k1).map(|k| (k as f64 * step, step)).collect()
            };
            for (v, step) in ticks {
                let y = ymap(v);
                painter.line_segment([Pos2::new(ax - 4.0, y), Pos2::new(ax, y)], Stroke::new(1.0, FG));
                painter.text(Pos2::new(ax - 6.0, y), Align2::RIGHT_CENTER, fmt_tick(v, step), small.clone(), FG);
            }
            match tr {
                None => {
                    let msg = if span <= WINDOW_MAX_S { "filtering…" } else { "filtering session… (zoom in to preview)" };
                    painter.text(r.center(), Align2::CENTER_CENTER, msg, small.clone(), MUTED);
                    if span <= WINDOW_MAX_S {
                        need_window.push(self.rows[i].spec.clone());
                    }
                }
                Some(tr) => {
                    xs.clear();
                    ys.clear();
                    tr.view(t_lo - 0.01 * span, t_hi + 0.01 * span, (plot_w * ppp) as f64 * 1.02, &mut xs, &mut ys);
                    let points: Vec<[f32; 2]> = xs
                        .iter()
                        .zip(&ys)
                        .map(|(x, y)| [((x - t) / (span / 2.0)) as f32, (((*y as f64 - lo) / (hi - lo)) * 2.0 - 1.0) as f32])
                        .collect();
                    let cb = LineStrip { slot: i, points, color: self.rows[i].color, offsets: offsets_for(r, ppp) };
                    ui.painter().with_clip_rect(r.intersect(rect)).add(egui_wgpu::Callback::new_paint_callback(r, cb));
                }
            }
        }
        // cursor line and time axis (time relative to the cursor; signed tick labels)
        let cx = x0 + plot_w / 2.0;
        let fg_painter = ui.painter_at(rect);
        fg_painter.line_segment([Pos2::new(cx, top), Pos2::new(cx, bottom)], Stroke::new(1.0, Color32::WHITE));
        fg_painter.line_segment([Pos2::new(x0, bottom + 1.0), Pos2::new(x1, bottom + 1.0)], Stroke::new(1.0, FG));
        let (major, minor) = nice_spacing(span, (plot_w as f64 / 110.0).floor().max(3.0));
        let tx = |v: f64| x0 + ((v + span / 2.0) / span) as f32 * plot_w;
        for (step, len, labelled) in [(minor, 3.0, false), (major, 5.0, true)] {
            let mut k = (-span / 2.0 / step).ceil() as i64;
            while (k as f64) * step <= span / 2.0 {
                let v = k as f64 * step;
                fg_painter.line_segment([Pos2::new(tx(v), bottom + 1.0), Pos2::new(tx(v), bottom + 1.0 + len)], Stroke::new(1.0, FG));
                if labelled {
                    let label = if v.abs() < major * 1e-3 {
                        "0".to_string()
                    } else if major >= 60.0 {
                        format!("{}{}", if v > 0.0 { "+" } else { "" }, fmt_time(v, Some(major * 10.0)))
                    } else {
                        let d = (-(major.log10().floor()) as i32).max(0) as usize;
                        format!("{v:+.d$}")
                    };
                    fg_painter.text(Pos2::new(tx(v), bottom + 6.0), Align2::CENTER_TOP, label, small.clone(), FG);
                }
                k += 1;
            }
        }
        // on-demand processing of the visible window for rows that aren't cached yet
        if !need_window.is_empty() {
            self.window_worker.request(need_window, t_lo - span, t_hi + span);
        }
        // mouse
        let hover_row = resp.hover_pos().and_then(|p| {
            let i = ((p.y - top) / (row_h + ROW_GAP)).floor();
            (i >= 0.0 && (i as usize) < n).then_some(i as usize)
        });
        if resp.hovered() {
            self.wheel(ui.ctx(), hover_row, plot_w);
        }
        if resp.dragged_by(egui::PointerButton::Primary) {
            self.set_time(self.t - resp.drag_delta().x as f64 * self.span / plot_w as f64);
        }
        if resp.double_clicked() {
            if let Some(i) = hover_row {
                self.reset_row_y(i);
            }
        }
    }

    fn overview_ui(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 0.0, BG);
        let ppp = ui.ctx().pixels_per_point();
        let x0 = rect.left() + 70.0;
        let x1 = rect.right() - MARGIN_X;
        let top = rect.top() + 6.0;
        let bottom = rect.bottom() - TAXIS_H - 2.0;
        let plot = Rect::from_min_max(Pos2::new(x0, top), Pos2::new(x1, bottom));
        let dur = self.rec.duration();
        let tx = |v: f64| x0 + (v / dur) as f32 * (x1 - x0);
        let small = FontId::proportional(11.0);
        p.line_segment([Pos2::new(x0 - 1.0, top), Pos2::new(x0 - 1.0, bottom)], Stroke::new(1.0, FG));
        p.line_segment([Pos2::new(x0, bottom + 1.0), Pos2::new(x1, bottom + 1.0)], Stroke::new(1.0, FG));
        let title = self.overview.as_ref().map(|o| o.title()).unwrap_or_default();
        let tr = self.overview.as_ref().and_then(|o| self.cache.get(o));
        match tr {
            Some(tr) if plot.width() > 10.0 => {
                let (mut xs, mut ys) = (vec![], vec![]);
                tr.view(0.0, dur, (plot.width() * ppp) as f64, &mut xs, &mut ys);
                let finite = ys.iter().filter(|v| v.is_finite());
                let lo = finite.clone().fold(f32::INFINITY, |a, b| a.min(*b)) as f64;
                let hi = finite.fold(f32::NEG_INFINITY, |a, b| a.max(*b)) as f64;
                if lo.is_finite() && hi > lo {
                    let pad = 0.05 * (hi - lo);
                    let (lo, hi) = (lo - pad, hi + pad);
                    let points = xs
                        .iter()
                        .zip(&ys)
                        .map(|(x, y)| [((x / dur) * 2.0 - 1.0) as f32, (((*y as f64 - lo) / (hi - lo)) * 2.0 - 1.0) as f32])
                        .collect();
                    let cb = LineStrip { slot: OVERVIEW_SLOT, points, color: FG, offsets: offsets_for(plot, ppp) };
                    ui.painter().with_clip_rect(plot).add(egui_wgpu::Callback::new_paint_callback(plot, cb));
                    // a few y labels
                    let target = (hi - lo) / 2.0;
                    let e = 10f64.powf(target.log10().floor());
                    let step = [1.0, 2.0, 2.5, 5.0].iter().map(|m| m * e).filter(|s| *s <= target).fold(e, f64::max);
                    let mut k = (lo / step).ceil() as i64;
                    while k as f64 * step <= hi {
                        let v = k as f64 * step;
                        let y = bottom - ((v - lo) / (hi - lo)) as f32 * plot.height();
                        p.text(Pos2::new(x0 - 5.0, y), Align2::RIGHT_CENTER, fmt_tick(v, step), small.clone(), FG);
                        k += 1;
                    }
                }
            }
            _ => {
                p.text(plot.center(), Align2::CENTER_CENTER, if self.overview.is_some() { "filtering…" } else { "" }, small.clone(), MUTED);
            }
        }
        p.text(Pos2::new(x0 + 4.0, top + 2.0), Align2::LEFT_TOP, title, FontId::proportional(12.0), FG);
        // view region and cursor
        let (a, b) = (tx(self.t - self.span / 2.0).max(x0), tx(self.t + self.span / 2.0).min(x1));
        p.rect_filled(Rect::from_min_max(Pos2::new(a, top), Pos2::new(b.max(a + 1.0), bottom)), 0.0, Color32::from_white_alpha(40));
        p.line_segment([Pos2::new(tx(self.t), top), Pos2::new(tx(self.t), bottom)], Stroke::new(1.0, Color32::WHITE));
        // time axis
        let (major, _) = nice_spacing(dur, ((x1 - x0) as f64 / 110.0).floor().max(3.0));
        let mut k = 1;
        while k as f64 * major < dur {
            let v = k as f64 * major;
            p.line_segment([Pos2::new(tx(v), bottom + 1.0), Pos2::new(tx(v), bottom + 6.0)], Stroke::new(1.0, FG));
            p.text(Pos2::new(tx(v), bottom + 6.0), Align2::CENTER_TOP, fmt_time(v, Some(major * 10.0)), small.clone(), FG);
            k += 1;
        }
        if resp.clicked() || resp.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = resp.interact_pointer_pos() {
                self.set_time(((pos.x - x0) / (x1 - x0)) as f64 * dur);
            }
        }
    }

    fn dialogs_ui(&mut self, ctx: &egui::Context) {
        let mut close = None;
        for (i, (title, body)) in self.dialogs.iter().enumerate() {
            egui::Window::new(title.as_str()).id(egui::Id::new(("dialog", i))).collapsible(false).resizable(false)
                .default_width(460.0).anchor(Align2::CENTER_CENTER, Vec2::new(0.0, i as f32 * 24.0))
                .show(ctx, |ui| {
                    ui.label(body.as_str());
                    if ui.button("OK").clicked() {
                        close = Some(i);
                    }
                });
        }
        if let Some(i) = close {
            self.dialogs.remove(i);
        }
    }
}

impl App {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        if let Some((last, times)) = &mut self.fps {
            times.push(ctx.input(|i| i.unstable_dt) as f64 * 1000.0);
            if last.elapsed().as_secs_f64() >= 1.0 {
                times.sort_by(|a, b| a.total_cmp(b));
                let n = times.len();
                eprintln!("fps {n:4}  frame ms median {:.1}  95% {:.1}  max {:.1}", times[n / 2], times[n * 95 / 100], times[n - 1]);
                *last = Instant::now();
                times.clear();
            }
        }
        self.tick();
        self.poll_video(&ctx);
        self.keys(&ctx);
        let building = self.builder.progress.lock().unwrap().is_some();
        if self.was_building && !building {
            for r in &mut self.rows {
                r.auto = None; // switch from window-based to session-based auto ranges
            }
        }
        self.was_building = building;

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(3.0);
            self.toolbar(ui);
            ui.add_space(3.0);
        });
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
        egui::Panel::bottom("overview").exact_size(110.0).frame(egui::Frame::NONE.fill(BG).inner_margin(4.0)).show(ui, |ui| self.overview_ui(ui));
        let max_w = (ui.available_width() * 0.45).max(120.0);
        egui::Panel::right("channels").resizable(true).default_size(700.0f32.min(max_w)).max_size(max_w).show(ui, |ui| self.channels_ui(ui));
        egui::CentralPanel::no_frame().frame(egui::Frame::NONE.fill(BG).inner_margin(4.0)).show(ui, |ui| {
            if self.video.is_some() || self.video_tex.is_some() {
                let h = ui.available_height();
                egui::Panel::top("video")
                    .resizable(true)
                    .default_size(h * 0.45)
                    .size_range(60.0..=h - 80.0)
                    .frame(egui::Frame::NONE.fill(Color32::BLACK))
                    .show(ui, |ui| self.video_ui(ui));
            }
            self.traces_ui(ui);
        });
        self.dialogs_ui(&ctx);
        if self.playing {
            ctx.request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec() -> Spec {
        serde_json::from_value(json!({"ch": "CH5", "ref": "CH6", "mode": "hilo", "label": "x",
            "band": [100.0, 3000.0], "notch": [60.0, 120.0], "order": 3, "ylim": [-50.0, 50.0]})).unwrap()
    }

    #[test]
    fn table_fields_round_trip() {
        let s = spec();
        let e = RowEdit::from_spec(&s);
        assert_eq!((e.lo.as_str(), e.hi.as_str(), e.notch.as_str(), e.extra.as_str(), e.y.as_str()),
                   ("100", "3000", "60,120", "order=3", "-50, 50"));
        let mut t = s.clone();
        e.apply(&mut t);
        assert_eq!(t, s, "unchanged fields must not change the spec (or its cache key)");
    }

    #[test]
    fn table_edits() {
        let mut s = spec();
        let mut e = RowEdit::from_spec(&s);
        e.lo = "200".into();
        e.hi = "".into(); // blank = no low-pass
        e.notch = "50".into();
        e.extra = "order=2 smooth_ms=5 bogus=1".into();
        e.y = "auto".into();
        e.apply(&mut s);
        assert_eq!(s.extra["band"], json!([200.0, null]));
        assert_eq!(s.extra["notch"], json!([50.0]));
        assert_eq!((s.extra["order"].clone(), s.extra["smooth_ms"].clone()), (json!(2), json!(5.0)));
        assert!(s.extra.get("bogus").is_none() && s.ylim.is_none());
        let mut e = RowEdit::from_spec(&s);
        e.lo = "abc".into(); // invalid: band left as it was
        e.notch = "".into(); // cleared
        e.extra = "".into();
        e.y = "10, 1".into(); // lo >= hi: ignored
        e.apply(&mut s);
        assert_eq!(s.extra["band"], json!([200.0, null]));
        assert!(s.extra.get("notch").is_none() && s.extra.get("order").is_none() && s.ylim.is_none());
    }

    #[test]
    fn preset_json_shape() {
        let p = Preset { time_base: Some(30.0), channels: vec![spec(), Spec::new("CH1", "slow")], overview: None };
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["channels"][0]["ref"], "CH6");
        assert_eq!(v["channels"][0]["band"], json!([100.0, 3000.0]));
        assert!(v["channels"][1].get("ref").is_none());
        let back: Preset = serde_json::from_value(v).unwrap();
        assert_eq!(back.channels, p.channels);
    }
}
