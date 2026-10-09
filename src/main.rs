//! syncviewR: an Open Ephys + behaviour-video viewer in Rust (egui + wgpu), modelled on the Python
//! syncview. Video frame k is locked to the k-th rising edge of the camera trigger line.

// Windows: no console window when started by double-click; see attach_console.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod app;
mod cache;
mod demo;
mod filters;
mod folder;
mod gpu;
mod npy;
mod oe;
mod preset;
mod shell;
mod splash;
mod video;

use anyhow::{Context, Result};
use clap::Parser;
use eframe::egui;
use std::path::PathBuf;
use std::sync::Arc;

/// View Open Ephys recordings side by side with a behaviour video, frame-locked to the camera trigger.
///
/// With no arguments it opens a start page: drop an Open Ephys folder and/or a video on it, or
/// try the demo.
///
/// Mouse: scroll = zoom time, sideways swipe / Shift+scroll / drag = pan, ⌘/Ctrl+scroll = scale a
/// row's Y, double-click = reset it; click or drag the overview strip to jump.
/// Keys: ←/→ one video frame (Shift: 10 % of the view), PgUp/PgDn one view, Home/End, Space play,
/// [ / ] playback speed, +/- zoom.
#[derive(Parser)]
#[command(version, verbatim_doc_comment)]
struct Cli {
    /// a folder holding an Open Ephys recording, and optionally one video and one preset JSON (e.g.
    /// a shared sample); they are found and opened together. --video / --preset override. If it
    /// holds several recordings or videos, the start page lists them to pick from.
    #[arg(value_name = "FOLDER", conflicts_with_all = ["rec", "demo"])]
    folder: Option<PathBuf>,
    /// Open Ephys recording folder (…/experimentN/recordingM, containing structure.oebin)
    #[arg(long)]
    rec: Option<PathBuf>,
    /// try syncviewR on synthetic data: writes a 5-minute recording, a matching video and a preset
    /// (~100 MB) into DIR (default: the cache folder) on first use, then opens them
    #[arg(long, value_name = "DIR", num_args = 0..=1, conflicts_with = "rec")]
    demo: Option<Option<PathBuf>>,
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
    /// Open Ephys continuous stream holding the data and the camera TTL (default:
    /// acquisition_board, else the recording's first stream)
    #[arg(long)]
    stream: Option<String>,
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
    /// (testing) start on the start page as if these files and folders had been dropped on it
    #[arg(long, hide = true, num_args = 1..)]
    drop: Vec<PathBuf>,
    /// (testing) build the cache for the preset's rows and overview, then exit
    #[arg(long, hide = true)]
    build_only: bool,
    /// (testing) decode these comma-separated frame indexes of --video into raw RGB files in DIR
    #[arg(long, hide = true, value_names = ["INDEXES", "DIR"], num_args = 2)]
    dump_frames: Option<Vec<String>>,
}

/// Window, Dock and ⌘-Tab icon (assets/icon.svg, made by assets/make_icon.py; rendered with
/// `rsvg-convert -w 512 -h 512`).
pub(crate) const ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");

/// Windows: a GUI-subsystem program has no console, so when started from a terminal, write
/// messages (and --help) to the terminal it was started from.
#[cfg(windows)]
fn attach_console() {
    use windows_sys::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
    // SAFETY: plain Win32 call; failure (no parent console) is harmless.
    unsafe {
        AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

fn main() -> Result<()> {
    #[cfg(windows)]
    attach_console();
    let cli = Cli::parse();
    ffmpeg_next::init().context("initialising FFmpeg")?;
    ffmpeg_next::util::log::set_level(ffmpeg_next::util::log::Level::Error);
    if let (Some(d), Some(v)) = (&cli.dump_frames, &cli.video) {
        let idx: Vec<usize> = d[0].split(',').map(|s| s.parse()).collect::<Result<_, _>>()?;
        return video::dump_frames(v, &idx, std::path::Path::new(&d[1]));
    }
    let root = cli.cache.clone().unwrap_or_else(cache::default_root);
    eprintln!("syncviewr: cache folder {}", root.display());
    let mut start = shell::Start::Splash(vec![]);
    if let Some(dir) = &cli.demo {
        let d = demo::ensure(dir.as_ref().unwrap_or(&root))?;
        let time = cli.time.or(Some(17.0)); // just before the first chewing bout
        start = shell::Start::Viewer(splash::Launch { rec: d.rec, video: cli.video.clone().or(Some(d.video)), preset: cli.preset.clone().or(Some(d.preset)), time });
    } else if let Some(dir) = &cli.folder {
        start = match folder::resolve(dir, cli.video.is_none(), cli.preset.is_none()) {
            Ok(f) => {
                let show = |p: &Option<PathBuf>| p.as_ref().map_or("none".into(), |p| p.display().to_string());
                eprintln!("syncviewr: recording {}", f.rec.display());
                eprintln!("syncviewr: video {}", show(&f.video));
                eprintln!("syncviewr: preset {}", show(&f.preset));
                shell::Start::Viewer(splash::Launch {
                    rec: f.rec,
                    video: cli.video.clone().or(f.video),
                    preset: cli.preset.clone().or(f.preset),
                    time: cli.time,
                })
            }
            // several recordings or videos: let the start page ask which
            Err(_) => shell::Start::Splash(vec![dir.clone()]),
        };
    } else if let Some(rec) = &cli.rec {
        start = shell::Start::Viewer(splash::Launch { rec: rec.clone(), video: cli.video.clone(), preset: cli.preset.clone(), time: cli.time });
    }
    if !cli.drop.is_empty() {
        start = shell::Start::Splash(cli.drop.clone());
    }
    if cli.build_only {
        let shell::Start::Viewer(l) = &start else { anyhow::bail!("--build-only needs one recording") };
        let rec = Arc::new(oe::Recording::open(&l.rec, cli.stream.as_deref())?);
        let cache = Arc::new(cache::TraceCache::new(rec.clone(), root)?);
        let (p, _) = preset::check(shell::load_preset(&l.preset, &rec)?, &rec);
        let mut specs: Vec<preset::Spec> = p.channels.into_iter().filter(|s| s.show).collect();
        specs.extend(p.overview);
        let t0 = std::time::Instant::now();
        cache.build(&specs, &|f| eprint!("\rbuilding {:.0}%", f * 100.0), &|| false)?;
        eprintln!("\nsyncviewr: built {} traces in {:.1} s", specs.len(), t0.elapsed().as_secs_f64());
        return Ok(());
    }
    let settings = shell::Settings { cache_root: root, stream: cli.stream, trigger_line: cli.trigger_line, time_base: cli.time_base, play: cli.play };
    let screenshot = cli.screenshot.map(|p| (p, cli.screenshot_after));
    let native = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1700.0, 1000.0])
            .with_title("syncviewR")
            .with_app_id("syncviewr")
            .with_drag_and_drop(true)
            .with_icon(Arc::new(eframe::icon_data::from_png_bytes(ICON_PNG).context("decoding the app icon")?)),
        ..Default::default()
    };
    eframe::run_native("syncviewr", native, Box::new(move |cc| Ok(Box::new(shell::Shell::new(cc, settings, start, screenshot)))))
        .map_err(|e| anyhow::anyhow!("{e}"))
}
