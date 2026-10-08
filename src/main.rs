//! syncviewR: an Open Ephys + behaviour-video viewer in Rust (egui + wgpu), modelled on the Python
//! syncview. Video frame k is locked to the k-th rising edge of the camera trigger line.

mod app;
mod cache;
mod filters;
mod gpu;
mod npy;
mod oe;
mod preset;
mod video;

use anyhow::{Context, Result};
use clap::Parser;
use eframe::egui;
use std::path::PathBuf;
use std::sync::Arc;

/// View Open Ephys recordings side by side with a behaviour video, frame-locked to the camera trigger.
///
/// Mouse: scroll = zoom time, sideways swipe / Shift+scroll / drag = pan, ⌘/Ctrl+scroll = scale a
/// row's Y, double-click = reset it; click or drag the overview strip to jump.
/// Keys: ←/→ one video frame (Shift: 10 % of the view), PgUp/PgDn one view, Home/End, Space play,
/// [ / ] playback speed, +/- zoom.
#[derive(Parser)]
#[command(version, verbatim_doc_comment)]
struct Cli {
    /// Open Ephys recording folder (…/experimentN/recordingM, containing structure.oebin)
    #[arg(long)]
    rec: PathBuf,
    /// video recorded during this recording (one camera)
    #[arg(long)]
    video: Option<PathBuf>,
    /// channel preset JSON (same format as Python syncview); default: all electrode channels
    #[arg(long)]
    preset: Option<PathBuf>,
    /// cache folder for filtered traces and video indexes (default: $SYNCVIEWR_CACHE, else the
    /// platform's user cache folder + /syncviewr). Pointing it at a Python syncview cache reuses it.
    #[arg(long)]
    cache: Option<PathBuf>,
    /// Open Ephys continuous stream holding the data and the camera TTL
    #[arg(long, default_value = "acquisition_board")]
    stream: String,
    /// TTL line carrying one pulse per video frame
    #[arg(long, default_value_t = 1)]
    trigger_line: i64,
    /// start at this time (seconds since the recording started)
    #[arg(long)]
    time: Option<f64>,
    /// initial time base (seconds shown across the window)
    #[arg(long)]
    time_base: Option<f64>,
    /// start playing (playback speed with [ and ])
    #[arg(long)]
    play: bool,
    /// (testing) save a screenshot (PPM) after SECONDS and quit: --screenshot out.ppm --screenshot-after 5
    #[arg(long, hide = true)]
    screenshot: Option<PathBuf>,
    #[arg(long, hide = true, default_value_t = 5.0)]
    screenshot_after: f64,
    /// (testing) build the cache for the preset's rows and overview, then exit
    #[arg(long, hide = true)]
    build_only: bool,
    /// (testing) decode these comma-separated frame indexes of --video into raw RGB files in DIR
    #[arg(long, hide = true, value_names = ["INDEXES", "DIR"], num_args = 2)]
    dump_frames: Option<Vec<String>>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ffmpeg_next::init().context("initialising FFmpeg")?;
    ffmpeg_next::util::log::set_level(ffmpeg_next::util::log::Level::Error);
    if let (Some(d), Some(v)) = (&cli.dump_frames, &cli.video) {
        let idx: Vec<usize> = d[0].split(',').map(|s| s.parse()).collect::<Result<_, _>>()?;
        return video::dump_frames(v, &idx, std::path::Path::new(&d[1]));
    }
    let rec = match oe::Recording::open(&cli.rec, &cli.stream) {
        Ok(r) => Arc::new(r),
        Err(e) => {
            eprintln!("syncviewr: cannot open recording: {e:#}");
            std::process::exit(2);
        }
    };
    let root = cli.cache.clone().unwrap_or_else(cache::default_root);
    eprintln!("syncviewr: cache folder {}", root.display());
    let cache = Arc::new(cache::TraceCache::new(rec.clone(), root)?);
    let preset = match &cli.preset {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?)
            .with_context(|| format!("parsing {}", p.display()))?,
        None => preset::from_recording(&rec),
    };
    if cli.build_only {
        let p: preset::Preset = preset;
        let (p, _) = preset::check(p, &rec);
        let mut specs: Vec<preset::Spec> = p.channels.into_iter().filter(|s| s.show).collect();
        specs.extend(p.overview);
        let t0 = std::time::Instant::now();
        cache.build(&specs, &|f| eprint!("\rbuilding {:.0}%", f * 100.0), &|| false)?;
        eprintln!("\nsyncviewr: built {} traces in {:.1} s", specs.len(), t0.elapsed().as_secs_f64());
        return Ok(());
    }
    let title = format!(
        "syncviewR — {}/{}  ({:.2} h, {} Hz)",
        rec.rec_dir.parent().and_then(|p| p.file_name()).unwrap_or_default().to_string_lossy(),
        rec.rec_dir.file_name().unwrap_or_default().to_string_lossy(),
        rec.duration() / 3600.0,
        rec.fs
    );
    let opts = app::Options {
        preset,
        video: cli.video,
        trigger_line: cli.trigger_line,
        start_time: cli.time,
        time_base: cli.time_base,
        screenshot: cli.screenshot.map(|p| (p, cli.screenshot_after)),
        play: cli.play,
    };
    let native = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default().with_inner_size([1700.0, 1000.0]).with_title(&title),
        ..Default::default()
    };
    eframe::run_native("syncviewr", native, Box::new(move |cc| Ok(Box::new(app::App::new(cc, rec, cache, opts)))))
        .map_err(|e| anyhow::anyhow!("{e}"))
}
