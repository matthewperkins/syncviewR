//! Exporting a clip: the behaviour video on top with a time/frame clock, and the chosen traces
//! scrolling underneath with a fixed cursor at each frame's trigger, encoded to MP4 (the layout of
//! Python syncview's clips).

use crate::cache::{Trace, TraceCache};
use crate::filters::{pad_samples, process};
use crate::oe::Recording;
use crate::preset::Spec;
use crate::video::FrameReader;
use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use anyhow::{bail, Context, Result};
use ffmpeg_next as ff;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tiny_skia::{Paint, PathBuilder, Pixmap, Stroke, Transform};

pub const PALETTE: [&str; 6] = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"];
const BG: [u8; 3] = [0x11, 0x11, 0x11];
const FG: [u8; 3] = [0xe8, 0xe8, 0xe4];
const GRID: [u8; 3] = [0x3a, 0x3a, 0x38];
const MUTED: [u8; 3] = [0x99, 0x99, 0x96];

/// The muscle or organ a row records: its label without a trailing number ("masseter3" ->
/// "masseter", "antrum 2" -> "antrum"). Unlabelled rows are each their own group.
fn group(spec: &Spec) -> String {
    let Some(label) = spec.label.as_deref().map(str::to_lowercase) else { return format!("\0{}", spec.ch) };
    let stem = label.trim_end_matches(|c: char| c.is_ascii_digit()).trim_end_matches([' ', '_', '-', '.']);
    if stem.is_empty() { label } else { stem.to_string() }
}

/// A colour per row: the spec's own colour if it has one, otherwise one palette colour per group,
/// so all of a muscle's channels (masseter1..4) share a colour.
pub fn row_colors(specs: &[&Spec]) -> Vec<String> {
    let mut groups: Vec<String> = vec![];
    specs
        .iter()
        .map(|s| {
            if let Some(c) = &s.color {
                return c.clone();
            }
            let g = group(s);
            let k = groups.iter().position(|x| *x == g).unwrap_or_else(|| {
                groups.push(g);
                groups.len() - 1
            });
            PALETTE[k % PALETTE.len()].to_string()
        })
        .collect()
}

pub fn parse_hex(s: &str) -> [u8; 3] {
    let h = s.trim_start_matches('#');
    let v = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).unwrap_or(0xcc);
    [v(0), v(2), v(4)]
}

/// One trace in the clip.
#[derive(Clone)]
pub struct ClipRow {
    pub spec: Spec,
    pub label: String,
    pub unit: String,
    pub color: [u8; 3],
    /// Y range; None: from the clip's data.
    pub ylim: Option<(f64, f64)>,
}

#[derive(Clone)]
pub struct ClipJob {
    pub out: PathBuf,
    pub video: PathBuf,
    /// Seconds since the recording started.
    pub from: f64,
    pub to: f64,
    /// Seconds of data across the chart (half past, half future).
    pub time_base: f64,
    /// Output width in pixels (default: the video's).
    pub width: Option<u32>,
    /// Playback speed: 1 real time, 0.25 four times slower.
    pub speed: f64,
    pub rows: Vec<ClipRow>,
}

pub struct ClipDone {
    pub frames: usize,
    pub size: (u32, u32),
    pub fps: f64,
    pub codec: String,
}

// ------------------------------------------------------------------ text

struct Text {
    fonts: Vec<FontRef<'static>>,
}

#[derive(Clone, Copy, PartialEq)]
enum Align {
    Left,
    Center,
    Right,
}

impl Text {
    fn new() -> Self {
        let fonts = [epaint_default_fonts::UBUNTU_LIGHT, epaint_default_fonts::HACK_REGULAR]
            .iter()
            .filter_map(|b| FontRef::try_from_slice(b).ok())
            .collect();
        Self { fonts }
    }

    /// The first font that has `c` (so arrows and µ fall back to Hack), or None.
    fn font_for(&self, c: char) -> Option<&FontRef<'static>> {
        self.fonts.iter().find(|f| f.glyph_id(c).0 != 0)
    }

    fn width(&self, text: &str, px: f32) -> f32 {
        text.chars().filter_map(|c| self.font_for(c).map(|f| f.as_scaled(PxScale::from(px)).h_advance(f.glyph_id(c)))).sum()
    }

    /// Draw `text` with its baseline at y, blending `color` by glyph coverage.
    #[allow(clippy::too_many_arguments)]
    fn draw(&self, pm: &mut Pixmap, text: &str, x: f32, y: f32, px: f32, color: [u8; 3], align: Align) {
        let w = self.width(text, px);
        let mut pen = match align {
            Align::Left => x,
            Align::Center => x - w / 2.0,
            Align::Right => x - w,
        };
        let (pw, ph) = (pm.width() as i32, pm.height() as i32);
        for c in text.chars() {
            let Some(f) = self.font_for(c) else { continue };
            let id = f.glyph_id(c);
            let scaled = f.as_scaled(PxScale::from(px));
            let g = id.with_scale_and_position(px, ab_glyph::point(pen, y));
            if let Some(og) = f.outline_glyph(g) {
                let b = og.px_bounds();
                let data = pm.data_mut();
                og.draw(|gx, gy, cov| {
                    let (xx, yy) = (b.min.x as i32 + gx as i32, b.min.y as i32 + gy as i32);
                    if xx < 0 || yy < 0 || xx >= pw || yy >= ph {
                        return;
                    }
                    let i = 4 * (yy as usize * pw as usize + xx as usize);
                    let a = cov.clamp(0.0, 1.0);
                    for k in 0..3 {
                        data[i + k] = (data[i + k] as f32 * (1.0 - a) + color[k] as f32 * a).round() as u8;
                    }
                });
            }
            pen += scaled.h_advance(id);
        }
    }

    /// White text with a dark outline (readable over video).
    fn draw_outlined(&self, pm: &mut Pixmap, text: &str, x: f32, y: f32, px: f32) {
        let o = (px / 12.0).max(1.0);
        for (dx, dy) in [(-o, 0.0), (o, 0.0), (0.0, -o), (0.0, o), (-o, -o), (o, o), (-o, o), (o, -o)] {
            self.draw(pm, text, x + dx, y + dy, px, [0, 0, 0], Align::Left);
        }
        self.draw(pm, text, x, y, px, [255, 255, 255], Align::Left);
    }
}

// ------------------------------------------------------------------ layout and drawing

fn paint(c: [u8; 3], alpha: u8) -> Paint<'static> {
    let mut p = Paint::default();
    p.set_color_rgba8(c[0], c[1], c[2], alpha);
    p.anti_alias = true;
    p
}

fn line(pm: &mut Pixmap, pts: &[(f32, f32)], c: [u8; 3], alpha: u8, width: f32) {
    let mut pb = PathBuilder::new();
    let mut it = pts.iter();
    let Some(&(x, y)) = it.next() else { return };
    pb.move_to(x, y);
    for &(x, y) in it {
        pb.line_to(x, y);
    }
    if let Some(path) = pb.finish() {
        let stroke = Stroke { width, ..Default::default() };
        pm.stroke_path(&path, &paint(c, alpha), &stroke, Transform::identity(), None);
    }
}

const NICE: [f64; 21] =
    [1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0];

fn fmt_num(v: f64, step: f64) -> String {
    let d = (-step.log10().floor()).max(0.0) as usize;
    let s = format!("{v:.d$}");
    if s.trim_start_matches('-').chars().all(|c| c == '0' || c == '.') { "0".into() } else { s }
}

/// Two to four "nice" ticks inside [lo, hi].
fn y_ticks(lo: f64, hi: f64) -> (Vec<f64>, f64) {
    let r = hi - lo;
    if r.is_nan() || r <= 0.0 {
        return (vec![], 1.0);
    }
    let raw = r / 3.0;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0].iter().map(|m| m * mag).find(|s| *s >= raw * 0.999).unwrap_or(10.0 * mag);
    let first = (lo / step).ceil() as i64;
    let last = (hi / step).floor() as i64;
    ((first..=last).map(|k| k as f64 * step).collect(), step)
}

/// The Python renderer's automatic Y range for a mode, from the clip's samples.
fn auto_ylim(mode: &str, y: &[f32]) -> (f64, f64) {
    let mut v: Vec<f64> = y.iter().map(|x| *x as f64).filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return (-1.0, 1.0);
    }
    let pct = |v: &mut Vec<f64>, p: f64| {
        let k = ((p / 100.0) * (v.len() - 1) as f64).round() as usize;
        *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
    };
    let (lo, hi) = match mode {
        "hilo" => {
            let mut a: Vec<f64> = v.iter().map(|x| x.abs()).collect();
            let m = pct(&mut a, 99.95) * 1.1;
            (-m, m)
        }
        "envelope" | "bandpower" => (0.0, pct(&mut v, 99.9) * 1.15),
        _ => {
            let (lo, hi) = (pct(&mut v, 0.2), pct(&mut v, 99.8));
            let r = hi - lo;
            (lo - 0.1 * r, hi + 0.1 * r)
        }
    };
    if hi > lo { (lo, hi) } else { (lo - 1.0, lo + 1.0) }
}

/// A row's data for [t_lo, t_hi]: the whole-session cache if built, else processed now.
fn row_trace(rec: &Recording, cache: &TraceCache, spec: &Spec, t_lo: f64, t_hi: f64) -> Result<Arc<Trace>> {
    if let Some(t) = cache.get(spec) {
        return Ok(t);
    }
    let fs = rec.fs;
    let i0 = (t_lo.max(0.0) * fs) as i64;
    let i1 = (t_hi.min(rec.duration()) * fs) as i64;
    let pad = pad_samples(spec, fs) as i64;
    let (first, len) = rec.clip(i0 - pad, i1 + pad);
    let ch = rec.ch(&spec.ch).with_context(|| format!("no channel {}", spec.ch))?;
    let r = spec.reference.as_deref().filter(|r| !r.is_empty()).and_then(|r| rec.ch(r));
    let (y, a, step) = process(rec.trace(ch, r, first, len), first as i64, fs, spec)?;
    let ys: Vec<f32> = y.iter().map(|v| *v as f32).collect();
    Ok(Arc::new(Trace::from_samples(a as f64 / fs, step as f64 / fs, ys)))
}

struct Layout {
    w: u32,
    h: u32,
    vid_w: u32,
    vid_h: u32,
    vid_x: u32,
    plot_x0: f32,
    plot_x1: f32,
    rows: Vec<(f32, f32)>, // (top, bottom) of each row
    font: f32,
}

fn layout(native: (u32, u32), width: Option<u32>, n_rows: usize) -> Layout {
    let even = |v: f64| ((v / 2.0).round() as u32 * 2).max(2);
    let w = even(width.map_or(native.0 as f64, |w| w as f64));
    let vid_h = even(native.1 as f64 * w as f64 / native.0 as f64);
    let s = (w as f32 / 1280.0).clamp(0.6, 3.0); // sizes are for a 1280-pixel-wide clip
    let font = 15.0 * s;
    // data panel: 40% of the height (Python's video_frac 0.6), at least 34 px per row
    let data_h = ((vid_h as f32 / 0.6 - vid_h as f32).max(n_rows as f32 * 34.0 * s + 60.0 * s)) as u32;
    let h = even((vid_h + data_h) as f64);
    let (top, bottom, gap) = (vid_h as f32 + 10.0 * s, h as f32 - 46.0 * s, 12.0 * s);
    let n = n_rows.max(1) as f32;
    let rh = ((bottom - top - gap * (n - 1.0)) / n).max(4.0);
    let rows = (0..n_rows).map(|i| (top + i as f32 * (rh + gap), top + i as f32 * (rh + gap) + rh)).collect();
    Layout { w, h, vid_w: w, vid_h, vid_x: 0, plot_x0: 175.0 * s, plot_x1: w as f32 - 16.0 * s, rows, font }
}

/// What doesn't change from frame to frame: labels, axes, grid.
fn background(l: &Layout, job: &ClipJob, ylims: &[(f64, f64)], text: &Text) -> Pixmap {
    let mut pm = Pixmap::new(l.w, l.h).expect("clip size");
    pm.fill(tiny_skia::Color::from_rgba8(BG[0], BG[1], BG[2], 255));
    let (x0, x1) = (l.plot_x0, l.plot_x1);
    let half = job.time_base / 2.0;
    let px_per_s = (x1 - x0) as f64 / job.time_base;
    let step = NICE.iter().copied().find(|s| job.time_base / s <= 8.0).unwrap_or(1800.0);
    let ticks: Vec<f64> = ((-half / step).ceil() as i64..=(half / step).floor() as i64).map(|k| k as f64 * step).collect();
    let xof = |t: f64| x0 + ((t + half) * px_per_s) as f32;
    let small = l.font * 0.8;
    for (i, ((top, bot), row)) in l.rows.iter().zip(&job.rows).enumerate() {
        for &t in &ticks {
            line(&mut pm, &[(xof(t), *top), (xof(t), *bot)], GRID, 255, 0.6 * l.font / 15.0);
        }
        // label and unit, right-aligned left of the plot
        let mid = (top + bot) / 2.0;
        text.draw(&mut pm, &row.label, x0 - 52.0 * l.font / 15.0, mid - 1.0, l.font, FG, Align::Right);
        text.draw(&mut pm, &format!("({})", row.unit), x0 - 52.0 * l.font / 15.0, mid + l.font, small, MUTED, Align::Right);
        // y ticks
        let (lo, hi) = ylims[i];
        let (yt, ystep) = y_ticks(lo, hi);
        for v in yt {
            let y = bot - ((v - lo) / (hi - lo)) as f32 * (bot - top);
            line(&mut pm, &[(x0 - 4.0, y), (x0, y)], FG, 200, 1.0);
            text.draw(&mut pm, &fmt_num(v, ystep), x0 - 7.0, y + small * 0.35, small, MUTED, Align::Right);
        }
    }
    // time axis under the last row
    if let Some(&(_, bot)) = l.rows.last() {
        for &t in &ticks {
            let lab = if t > 0.0 { format!("+{}", fmt_num(t, step)) } else { fmt_num(t, step) };
            text.draw(&mut pm, &lab, xof(t), bot + small * 1.4, small, FG, Align::Center);
        }
        let cap = "time relative to the video frame (s)   ← past | future →";
        text.draw(&mut pm, cap, (x0 + x1) / 2.0, bot + small * 1.4 + l.font * 1.25, small, MUTED, Align::Center);
    }
    pm
}

// ------------------------------------------------------------------ encoding

struct Encoder {
    octx: ff::format::context::Output,
    enc: ff::encoder::Video,
    scaler: ff::software::scaling::Context,
    stream: usize,
    tb: ff::Rational,
    n: i64,
    codec: String,
}

impl Encoder {
    /// H.264 where this FFmpeg has an encoder for it that opens (x264 in system builds, Apple's
    /// encoder on macOS), otherwise FFmpeg's own MPEG-4 encoder at high quality.
    fn open(path: &Path, w: u32, h: u32, fps: f64) -> Result<Self> {
        // (testing: SYNCVIEWR_CLIP_CODEC=mpeg4 forces an encoder)
        let forced = std::env::var("SYNCVIEWR_CLIP_CODEC").ok();
        let order: Vec<&str> = match &forced {
            Some(c) => vec![c.as_str()],
            None => vec!["libx264", "h264_videotoolbox", "mpeg4"],
        };
        let mut errors = vec![];
        for name in order {
            let Some(codec) = ff::encoder::find_by_name(name) else { continue };
            match Self::open_with(path, w, h, fps, codec, name) {
                Ok(e) => return Ok(e),
                Err(e) => errors.push(format!("{name}: {e:#}")),
            }
        }
        bail!("no video encoder could be opened{}", errors.iter().map(|e| format!("\n  {e}")).collect::<String>())
    }

    fn open_with(path: &Path, w: u32, h: u32, fps: f64, codec: ff::Codec, name: &str) -> Result<Self> {
        let mut octx = ff::format::output(&path).with_context(|| format!("creating {}", path.display()))?;
        let name = name.to_string();
        let global = octx.format().flags().contains(ff::format::Flags::GLOBAL_HEADER);
        let mut ctx = ff::codec::context::Context::new_with_codec(codec).encoder().video()?;
        let fmts: Vec<ff::format::Pixel> = codec.video().ok().and_then(|v| v.formats().map(|f| f.collect())).unwrap_or_default();
        let pix = [ff::format::Pixel::YUV420P, ff::format::Pixel::NV12]
            .into_iter()
            .find(|p| fmts.is_empty() || fmts.contains(p))
            .unwrap_or(ff::format::Pixel::YUV420P);
        let tb = ff::Rational::new(1000, (fps * 1000.0).round().max(1.0) as i32);
        ctx.set_width(w);
        ctx.set_height(h);
        ctx.set_format(pix);
        ctx.set_time_base(tb);
        ctx.set_frame_rate(Some(ff::Rational::new((fps * 1000.0).round() as i32, 1000)));
        ctx.set_gop((fps * 2.0).round().max(1.0) as u32);
        ctx.set_bit_rate((w as f64 * h as f64 * fps * 0.25).max(2e6) as usize);
        let mut flags = if global { ff::codec::Flags::GLOBAL_HEADER } else { ff::codec::Flags::empty() };
        let mut opts = ff::Dictionary::new();
        match name.as_str() {
            "libx264" => {
                opts.set("crf", "18");
                opts.set("preset", "medium");
            }
            "h264_videotoolbox" => {
                // let VideoToolbox fall back to Apple's software encoder (virtual machines)
                opts.set("allow_sw", "1");
            }
            "mpeg4" => {
                // constant quality 2 (of 1..31): close to transparent
                ctx.set_global_quality(2 * ff::ffi::FF_QP2LAMBDA);
                flags |= ff::codec::Flags::QSCALE;
            }
            _ => {}
        }
        ctx.set_flags(flags);
        let enc = ctx.open_with(opts).with_context(|| format!("opening the {name} encoder"))?;
        let mut st = octx.add_stream(codec)?;
        st.set_parameters(&enc);
        st.set_time_base(tb);
        let stream = st.index();
        let mut mux = ff::Dictionary::new();
        mux.set("movflags", "+faststart");
        octx.write_header_with(mux)?;
        let scaler = ff::software::scaling::Context::get(ff::format::Pixel::RGBA, w, h, pix, w, h, ff::software::scaling::Flags::BICUBIC)?;
        Ok(Self { octx, enc, scaler, stream, tb, n: 0, codec: name })
    }

    fn drain(&mut self) -> Result<()> {
        let mut pkt = ff::Packet::empty();
        let st_tb = self.octx.stream(self.stream).unwrap().time_base();
        while self.enc.receive_packet(&mut pkt).is_ok() {
            pkt.set_stream(self.stream);
            pkt.rescale_ts(self.tb, st_tb);
            pkt.write_interleaved(&mut self.octx)?;
        }
        Ok(())
    }

    fn push(&mut self, pm: &Pixmap) -> Result<()> {
        let (w, h) = (pm.width(), pm.height());
        let mut rgba = ff::frame::Video::new(ff::format::Pixel::RGBA, w, h);
        let stride = rgba.stride(0);
        let src = pm.data();
        let dst = rgba.data_mut(0);
        for row in 0..h as usize {
            dst[row * stride..row * stride + 4 * w as usize].copy_from_slice(&src[row * 4 * w as usize..(row + 1) * 4 * w as usize]);
        }
        let mut out = ff::frame::Video::empty();
        self.scaler.run(&rgba, &mut out)?;
        out.set_pts(Some(self.n));
        self.n += 1;
        self.enc.send_frame(&out)?;
        self.drain()
    }

    fn finish(mut self) -> Result<String> {
        self.enc.send_eof()?;
        self.drain()?;
        self.octx.write_trailer()?;
        Ok(self.codec)
    }
}

// ------------------------------------------------------------------ rendering

/// Render `job`. `frames` is the recording row of each video frame's trigger; `period` the camera
/// frame period (s). progress(fraction); cancel() true stops (the partial file is removed).
#[allow(clippy::too_many_arguments)]
pub fn render(
    job: &ClipJob,
    rec: &Recording,
    cache: &TraceCache,
    frames: &[i64],
    period: f64,
    index_dir: &Path,
    progress: &dyn Fn(f64),
    cancel: &dyn Fn() -> bool,
) -> Result<ClipDone> {
    let fs = rec.fs;
    if job.to.is_nan() || job.to <= job.from {
        bail!("the clip must end after it starts");
    }
    if job.speed.is_nan() || job.speed <= 0.0 || job.time_base.is_nan() || job.time_base <= 0.0 {
        bail!("speed and time base must be positive");
    }
    let k0 = frames.partition_point(|r| (*r as f64 / fs) < job.from);
    let k1 = frames.partition_point(|r| (*r as f64 / fs) <= job.to);
    if k1 <= k0 {
        bail!("no video frames between {:.3} s and {:.3} s", job.from, job.to);
    }
    if job.rows.is_empty() {
        bail!("no traces to show");
    }
    let half = job.time_base / 2.0;
    let (t_lo, t_hi) = (job.from - half - 1.0, job.to + half + 1.0);
    let traces: Vec<Arc<Trace>> = job.rows.iter().map(|r| row_trace(rec, cache, &r.spec, t_lo, t_hi)).collect::<Result<_>>()?;
    let ylims: Vec<(f64, f64)> = job
        .rows
        .iter()
        .zip(&traces)
        .map(|(r, tr)| {
            r.ylim.unwrap_or_else(|| {
                let (mut xs, mut ys) = (vec![], vec![]);
                tr.view(t_lo, t_hi, 20_000.0, &mut xs, &mut ys);
                auto_ylim(&r.spec.mode, &ys)
            })
        })
        .collect();
    let native = FrameReader::native_size(&job.video)?;
    let l = layout(native, job.width, job.rows.len());
    let text = Text::new();
    let bg = background(&l, job, &ylims, &text);
    let mut reader = FrameReader::open(&job.video, index_dir, k0, l.vid_w, l.vid_h)?;
    let fps = job.speed / period;
    let tmp = job.out.with_extension(format!("partial.{}", job.out.extension().and_then(|e| e.to_str()).unwrap_or("mp4")));
    let mut enc = Encoder::open(&tmp, l.w, l.h, fps)?;
    let result = (|| -> Result<()> {
        use rayon::prelude::*;
        let (x0, x1) = (l.plot_x0, l.plot_x1);
        let n_px = (x1 - x0) as f64;
        let lw = (1.1 * l.font / 15.0).max(1.0);
        let draw = |k: usize, f: &crate::video::Frame| -> Pixmap {
            let mut pm = bg.clone();
            // video: copy row by row into the RGBA pixmap
            let data = pm.data_mut();
            let (fw, fh) = (f.width.min(l.vid_w as usize), f.height.min(l.vid_h as usize));
            for row in 0..fh {
                let src = &f.rgb[3 * row * f.width..3 * (row * f.width + fw)];
                let dst = &mut data[4 * (row * l.w as usize + l.vid_x as usize)..4 * (row * l.w as usize + l.vid_x as usize + fw)];
                for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(3)) {
                    d[..3].copy_from_slice(s);
                    d[3] = 255;
                }
            }
            let tc = frames[k] as f64 / fs;
            let clock = format!("t = {tc:.3} s   frame {k}");
            text.draw_outlined(&mut pm, &clock, 12.0 * l.font / 15.0, l.vid_h as f32 - 10.0 * l.font / 15.0, (l.vid_h as f32 / 30.0).max(13.0));
            let (mut xs, mut ys) = (vec![], vec![]);
            for ((tr, &(top, bot)), (row, &(lo, hi))) in traces.iter().zip(&l.rows).zip(job.rows.iter().zip(&ylims)) {
                xs.clear();
                ys.clear();
                tr.view(tc - half, tc + half, n_px, &mut xs, &mut ys);
                let pts: Vec<(f32, f32)> = xs
                    .iter()
                    .zip(&ys)
                    .filter(|(t, _)| (**t - tc).abs() <= half + 1e-9)
                    .map(|(t, v)| {
                        let x = x0 + ((t - tc + half) / job.time_base * n_px) as f32;
                        let y = bot - (((*v as f64 - lo) / (hi - lo)) as f32).clamp(-0.02, 1.02) * (bot - top);
                        (x, y)
                    })
                    .collect();
                line(&mut pm, &pts, row.color, 255, lw);
                let xc = (x0 + x1) / 2.0;
                line(&mut pm, &[(xc, top), (xc, bot)], FG, 205, 1.0 * l.font / 15.0);
            }
            pm
        };
        // decode a batch in order, draw it in parallel, encode in order
        let batch = rayon::current_num_threads().max(1) * 2;
        let mut k = k0;
        while k < k1 {
            if cancel() {
                bail!("cancelled");
            }
            let n = batch.min(k1 - k);
            let vids: Vec<crate::video::Frame> = (0..n).map(|_| reader.next()).collect::<Result<_>>()?;
            let pms: Vec<Pixmap> = vids.par_iter().enumerate().map(|(i, f)| draw(k + i, f)).collect();
            for pm in &pms {
                enc.push(pm)?;
            }
            k += n;
            progress((k - k0) as f64 / (k1 - k0) as f64);
        }
        Ok(())
    })();
    match result {
        Ok(()) => {
            let codec = enc.finish()?;
            std::fs::rename(&tmp, &job.out).with_context(|| format!("writing {}", job.out.display()))?;
            Ok(ClipDone { frames: k1 - k0, size: (l.w, l.h), fps, codec })
        }
        Err(e) => {
            drop(enc);
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(ch: &str, label: Option<&str>, color: Option<&str>) -> Spec {
        let mut s = Spec::new(ch, "hilo");
        s.label = label.map(str::to_string);
        s.color = color.map(str::to_string);
        s
    }

    #[test]
    fn muscles_share_colours() {
        let specs = [
            spec("CH13", Some("masseter1"), None),
            spec("CH14", Some("masseter2"), None),
            spec("CH1", Some("digastric1"), None),
            spec("CH2", Some("digastric 2"), None),
            spec("CH5", Some("antrum1"), Some("#ffffff")),
            spec("CH6", Some("antrum2"), None),
            spec("CH9", None, None),
            spec("CH10", None, None),
        ];
        let refs: Vec<&Spec> = specs.iter().collect();
        let c = row_colors(&refs);
        assert_eq!(c[0], c[1]);
        assert_eq!(c[2], c[3]);
        assert_ne!(c[0], c[2]);
        assert_eq!(c[4], "#ffffff"); // a preset's own colour wins
        assert_ne!(c[5], c[0]);
        assert_ne!(c[6], c[7]); // unlabelled rows keep a colour each
    }

    #[test]
    fn ticks_and_labels() {
        assert_eq!(y_ticks(-120.0, 130.0).0, vec![-100.0, 0.0, 100.0]);
        assert_eq!(fmt_num(0.5, 0.5), "0.5");
        assert_eq!(fmt_num(-0.0, 1.0), "0");
        let t = Text::new();
        assert!(t.width("masseter1 (µV) ← →", 15.0) > 50.0);
    }
}
