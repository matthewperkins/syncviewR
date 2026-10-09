//! The window: the splash page until a recording is chosen, then the viewer. Dropping a folder on
//! the viewer goes back to the splash page with it; dropping one video attaches it.

use crate::app::{self, App};
use crate::splash::{Launch, Splash};
use crate::{cache, oe, preset};
use anyhow::{Context, Result};
use eframe::egui;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Command-line settings that apply to whatever is opened.
pub struct Settings {
    pub cache_root: PathBuf,
    pub stream: Option<String>,
    pub trigger_line: i64,
    pub time_base: Option<f64>,
    pub play: bool,
}

pub enum Start {
    /// Open this straight away (falling back to the splash page if it can't be opened).
    Viewer(Launch),
    /// The splash page, as if these paths had been dropped on it.
    Splash(Vec<PathBuf>),
}

enum State {
    Splash(Box<Splash>),
    Viewer(Box<App>),
}

pub struct Shell {
    settings: Settings,
    state: State,
    screenshot: Option<(PathBuf, f64)>,
    started: Instant,
    shot_requested: bool,
}

pub fn title(rec: &oe::Recording) -> String {
    format!(
        "syncviewR — {}/{}  ({:.2} h, {} Hz)",
        rec.rec_dir.parent().and_then(|p| p.file_name()).unwrap_or_default().to_string_lossy(),
        rec.rec_dir.file_name().unwrap_or_default().to_string_lossy(),
        rec.duration() / 3600.0,
        rec.fs
    )
}

pub fn load_preset(path: &Option<PathBuf>, rec: &oe::Recording) -> Result<preset::Preset> {
    Ok(match path {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?)
            .with_context(|| format!("parsing {}", p.display()))?,
        None => preset::from_recording(rec),
    })
}

impl Shell {
    pub fn new(cc: &eframe::CreationContext<'_>, settings: Settings, start: Start, screenshot: Option<(PathBuf, f64)>) -> Self {
        crate::gpu::init(cc.wgpu_render_state.as_ref().expect("syncviewr needs the wgpu renderer"));
        let ctx = &cc.egui_ctx;
        ctx.set_visuals(egui::Visuals::dark());
        let splash = Splash::new(ctx, settings.cache_root.clone(), settings.trigger_line, settings.stream.clone());
        let mut shell = Self { settings, state: State::Splash(Box::new(splash)), screenshot, started: Instant::now(), shot_requested: false };
        match start {
            Start::Viewer(l) => shell.open(ctx, l),
            Start::Splash(paths) => {
                if let State::Splash(s) = &mut shell.state {
                    s.take_paths(paths);
                }
            }
        }
        shell
    }

    /// Open the viewer, or stay on (or return to) the splash page with the error.
    fn open(&mut self, ctx: &egui::Context, l: Launch) {
        match self.viewer(ctx, &l) {
            Ok(app) => {
                self.state = State::Viewer(Box::new(app));
            }
            Err(e) => {
                if !matches!(self.state, State::Splash(_)) {
                    self.state = State::Splash(Box::new(Splash::new(ctx, self.settings.cache_root.clone(), self.settings.trigger_line, self.settings.stream.clone())));
                }
                if let State::Splash(s) = &mut self.state {
                    s.error(format!("Couldn't open {}: {e:#}", l.rec.display()));
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Title("syncviewR".into()));
            }
        }
    }

    fn viewer(&self, ctx: &egui::Context, l: &Launch) -> Result<App> {
        let rec = Arc::new(oe::Recording::open(&l.rec, self.settings.stream.as_deref())?);
        let cache = Arc::new(cache::TraceCache::new(rec.clone(), self.settings.cache_root.clone())?);
        let preset = load_preset(&l.preset, &rec)?;
        eprintln!("syncviewr: recording {}", rec.rec_dir.display());
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title(&rec)));
        let opts = app::Options {
            preset,
            video: l.video.clone(),
            trigger_line: self.settings.trigger_line,
            start_time: l.time,
            time_base: self.settings.time_base,
            play: self.settings.play,
        };
        Ok(App::new(ctx, rec, cache, opts))
    }

    fn screenshot(&mut self, ctx: &egui::Context) {
        let Some((path, after)) = self.screenshot.clone() else { return };
        if !self.shot_requested && self.started.elapsed().as_secs_f64() >= after {
            self.shot_requested = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        let shot = ctx.input(|i| {
            i.events.iter().find_map(|e| if let egui::Event::Screenshot { image, .. } = e { Some(image.clone()) } else { None })
        });
        if let Some(img) = shot {
            let [w, h] = img.size;
            let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
            for px in &img.pixels {
                out.extend_from_slice(&[px.r(), px.g(), px.b()]);
            }
            match std::fs::write(&path, out) {
                Ok(()) => eprintln!("syncviewr: screenshot saved to {}", path.display()),
                Err(e) => eprintln!("syncviewr: screenshot failed: {e}"),
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}

impl eframe::App for Shell {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty()).collect());
        let launch = match &mut self.state {
            State::Splash(s) => {
                if !dropped.is_empty() {
                    s.take_paths(dropped);
                }
                s.ui(ui)
            }
            State::Viewer(app) => {
                if let [one] = dropped.as_slice()
                    && crate::folder::is_video(one)
                {
                    app.attach_video(one.clone(), &ctx);
                }
                let to_splash = dropped.len() > 1 || dropped.iter().any(|p| !crate::folder::is_video(p));
                if to_splash {
                    let mut s = Splash::new(&ctx, self.settings.cache_root.clone(), self.settings.trigger_line, self.settings.stream.clone());
                    s.take_paths(dropped);
                    ctx.send_viewport_cmd(egui::ViewportCommand::Title("syncviewR".into()));
                    self.state = State::Splash(Box::new(s));
                } else {
                    app.ui(ui);
                }
                None
            }
        };
        if let Some(l) = launch {
            self.open(&ctx, l);
        }
        self.screenshot(&ctx);
    }
}
