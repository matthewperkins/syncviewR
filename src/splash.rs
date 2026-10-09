//! The splash page: try the demo, or drop (or choose) an Open Ephys folder and/or one video. A
//! folder holding several recordings (a day's session) or several videos opens a list to pick
//! from with ↑/↓ and Enter.

use crate::app::{fmt_time, BG, FG, LABEL, MUTED};
use crate::{demo, folder, video};
use eframe::egui::{self, Color32, Key, RichText};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::Instant;

/// What to open in the viewer.
pub struct Launch {
    pub rec: PathBuf,
    pub video: Option<PathBuf>,
    pub preset: Option<PathBuf>,
    pub time: Option<f64>,
}

const ACCENT: Color32 = Color32::from_rgb(0x9a, 0x8c, 0xe0); // the icon's violet rim light

#[derive(PartialEq)]
enum Pick {
    Recording,
    Video,
}

struct Item {
    /// None: "no video".
    path: Option<PathBuf>,
    label: String,
    detail: String,
}

struct Picker {
    what: Pick,
    items: Vec<Item>,
    sel: usize,
    moved: bool,
}

pub struct Splash {
    cache_root: PathBuf,
    trigger_line: i64,
    rec: Option<PathBuf>,
    video: Option<PathBuf>,
    preset: Option<PathBuf>,
    picker: Option<Picker>,
    /// Videos found with several recordings: offered once a recording is picked.
    later_videos: Option<(PathBuf, Vec<PathBuf>)>,
    msg: Option<(String, bool)>,
    demo: Option<(Receiver<Result<demo::Demo, String>>, Instant)>,
    icon: egui::TextureHandle,
    launch: Option<Launch>,
}

/// Milliseconds since 1970 as a local date and time (to the minute), like the session folder
/// names the GUI makes. UTC where the local time zone isn't available.
fn local_time(ms: i64) -> String {
    #[cfg(unix)]
    {
        let t = ms.div_euclid(1000) as libc::time_t;
        // SAFETY: localtime_r only writes the struct it is given.
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        if !unsafe { libc::localtime_r(&t, &mut tm) }.is_null() {
            return format!("{}-{:02}-{:02} {:02}:{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);
        }
    }
    let secs = ms.div_euclid(1000);
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Howard Hinnant's civil_from_days
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y}-{m:02}-{d:02} {:02}:{:02} UTC", rem / 3600, rem % 3600 / 60)
}

fn rel(p: &Path, base: &Path) -> String {
    match p.strip_prefix(base) {
        Ok(r) if !r.as_os_str().is_empty() => r.display().to_string(),
        _ => p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned()),
    }
}

/// One line about a recording: length, channels, start, camera triggers.
fn describe_recording(path: &Path, line: i64) -> String {
    let rec = match openephys::Recording::open(path) {
        Ok(r) => r,
        Err(e) => return format!("can't open: {e}"),
    };
    let Ok(c) = rec.stream("acquisition_board").or_else(|_| rec.main_stream()) else { return "no continuous data".into() };
    let mut parts = vec![
        fmt_time(c.duration(), Some(3600.0)),
        format!("{} ch @ {} kHz", c.n_channels(), c.sample_rate() / 1000.0),
    ];
    if let Some(ms) = rec.sync_messages().ok().flatten().and_then(|m| m.software_time_ms) {
        parts.push(format!("started {}", local_time(ms)));
    }
    if let Ok(l) = u16::try_from(line) {
        let n: usize = rec.ttl_for(c).map(|t| t.lines().get(&l).map_or(0, |e| e.rising)).sum();
        parts.push(if n > 0 { format!("{n} triggers on line {l}") } else { format!("no triggers on line {l}") });
    }
    parts.join("  ·  ")
}

fn describe_video(path: &Path) -> String {
    let size = std::fs::metadata(path).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);
    match video::video_duration(path) {
        Some(d) => format!("{}  ·  {size:.0} MB", fmt_time(d, Some(3600.0))),
        None => format!("{size:.0} MB"),
    }
}

impl Splash {
    pub fn new(ctx: &egui::Context, cache_root: PathBuf, trigger_line: i64) -> Self {
        let icon = eframe::icon_data::from_png_bytes(crate::ICON_PNG).expect("built-in icon");
        let img = egui::ColorImage::from_rgba_unmultiplied([icon.width as usize, icon.height as usize], &icon.rgba);
        let icon = ctx.load_texture("syncviewr-icon", img, egui::TextureOptions::LINEAR);
        Self {
            cache_root,
            trigger_line,
            rec: None,
            video: None,
            preset: None,
            picker: None,
            later_videos: None,
            msg: None,
            demo: None,
            icon,
            launch: None,
        }
    }

    pub fn error(&mut self, e: String) {
        self.msg = Some((e, true));
    }

    fn note(&mut self, m: String) {
        self.msg = Some((m, false));
    }

    /// Files and folders dropped (or chosen, or given on the command line).
    pub fn take_paths(&mut self, paths: Vec<PathBuf>) {
        self.msg = None;
        let videos: Vec<&PathBuf> = paths.iter().filter(|p| folder::is_video(p)).collect();
        match videos.as_slice() {
            [] => {}
            [v] => self.video = Some((*v).clone()),
            many => {
                self.error(format!("Drop one video at a time ({} were dropped).", many.len()));
                return;
            }
        }
        let video_dropped = videos.len() == 1;
        for p in paths.iter().filter(|p| p.extension().is_some_and(|e| e == "json")) {
            self.preset = Some(p.clone());
        }
        let recs: Vec<&PathBuf> = paths
            .iter()
            .filter(|p| p.is_dir() || p.extension().is_some_and(|e| e.eq_ignore_ascii_case("nwb")) || p.ends_with("structure.oebin"))
            .collect();
        let Some(first) = recs.first() else {
            if video_dropped && self.rec.is_none() {
                self.note("Video added. Now drop the Open Ephys folder it goes with.".into());
            }
            if self.rec.is_some() {
                self.go();
            }
            return;
        };
        if recs.len() > 1 {
            self.note(format!("Several folders were dropped; using {}.", first.display()));
        }
        let dir = if first.ends_with("structure.oebin") { first.parent().unwrap_or(first).to_path_buf() } else { (*first).clone() };
        let found = openephys::find_recordings(&dir, 6);
        if dir.is_dir()
            && let Some(p) = folder::pick_preset(&folder::presets(&dir))
        {
            self.preset = Some(p);
        }
        let vids = if video_dropped || !dir.is_dir() { vec![] } else { folder::videos(&dir, &found) };
        match found.as_slice() {
            [] => self.error(format!("No Open Ephys recording in {}: looked for folders with structure.oebin, and .nwb files.", dir.display())),
            [one] => {
                self.rec = Some(one.clone());
                self.offer_videos(&dir, vids);
            }
            _ => {
                let items = found.iter().map(|r| Item { path: Some(r.clone()), label: rel(r, &dir), detail: describe_recording(r, self.trigger_line) }).collect();
                self.picker = Some(Picker { what: Pick::Recording, items, sel: 0, moved: true });
                self.later_videos = Some((dir.clone(), vids));
            }
        }
    }

    /// After a recording is known: use the folder's one video, or ask which, then open.
    fn offer_videos(&mut self, dir: &Path, vids: Vec<PathBuf>) {
        match vids.as_slice() {
            [] => self.go(),
            [v] => {
                self.video = Some(v.clone());
                self.go();
            }
            _ => {
                let mut items = vec![Item { path: None, label: "No video".into(), detail: String::new() }];
                items.extend(vids.iter().map(|v| Item { path: Some(v.clone()), label: rel(v, dir), detail: describe_video(v) }));
                // pre-select the dropped video if it's one of them, else the first video
                let sel = self.video.as_ref().and_then(|v| vids.iter().position(|x| x == v)).map_or(1, |i| i + 1);
                self.picker = Some(Picker { what: Pick::Video, items, sel, moved: true });
            }
        }
    }

    fn go(&mut self) {
        if let Some(rec) = self.rec.clone() {
            self.launch = Some(Launch { rec, video: self.video.clone(), preset: self.preset.clone(), time: None });
        }
    }

    fn chose(&mut self, path: Option<PathBuf>) {
        let Some(p) = self.picker.take() else { return };
        match p.what {
            Pick::Recording => {
                self.rec = path;
                let (dir, vids) = self.later_videos.take().unwrap_or_default();
                if self.video.is_some() { self.go() } else { self.offer_videos(&dir, vids) }
            }
            Pick::Video => {
                self.video = path;
                self.go();
            }
        }
    }

    fn start_demo(&mut self) {
        let (tx, rx) = channel();
        let root = self.cache_root.clone();
        std::thread::spawn(move || {
            let _ = tx.send(demo::ensure(&root).map_err(|e| format!("{e:#}")));
        });
        self.demo = Some((rx, Instant::now()));
        self.msg = None;
    }

    fn poll_demo(&mut self, ctx: &egui::Context) {
        let Some((rx, _)) = &self.demo else { return };
        match rx.try_recv() {
            Ok(Ok(d)) => {
                self.demo = None;
                self.launch = Some(Launch { rec: d.rec, video: Some(d.video), preset: Some(d.preset), time: Some(17.0) });
            }
            Ok(Err(e)) => {
                self.demo = None;
                self.error(format!("Couldn't write the demo: {e}"));
            }
            Err(_) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
        }
    }

    fn choose_folder(&mut self) {
        if let Some(d) = rfd::FileDialog::new().set_title("Open Ephys folder: a session, Record Node or recording").pick_folder() {
            self.take_paths(vec![d]);
        }
    }

    fn choose_video(&mut self) {
        if let Some(v) = rfd::FileDialog::new().set_title("Video recorded during the session").add_filter("Video", &["mp4", "mkv", "mov", "avi"]).pick_file() {
            self.take_paths(vec![v]);
        }
    }

    /// Draw the page; returns what to open once the user has chosen.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<Launch> {
        let ctx = ui.ctx().clone();
        self.poll_demo(&ctx);
        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
        egui::CentralPanel::no_frame().frame(egui::Frame::NONE.fill(BG)).show(ui, |ui| {
            let w = 620f32.min(ui.available_width() - 32.0);
            ui.vertical_centered(|ui| {
                ui.add_space(((ui.available_height() - 600.0) / 2.0).max(16.0));
                ui.add(egui::Image::new(&self.icon).fit_to_exact_size(egui::vec2(112.0, 112.0)));
                ui.add_space(6.0);
                ui.label(RichText::new("syncviewR").size(30.0).color(LABEL));
                ui.label(RichText::new("Open Ephys recordings, frame-locked to behaviour video").color(MUTED));
                ui.add_space(22.0);

                // drop zone
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 170.0), egui::Sense::click());
                let stroke_col = if hovering || resp.hovered() { ACCENT } else { Color32::from_gray(70) };
                let fill = if hovering { ACCENT.gamma_multiply(0.08) } else { Color32::from_gray(22) };
                ui.painter().rect(rect, 10.0, fill, egui::Stroke::new(if hovering { 2.0 } else { 1.0 }, stroke_col), egui::StrokeKind::Inside);
                let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink(16.0)).layout(egui::Layout::top_down(egui::Align::Center)));
                child.add_space(22.0);
                child.label(RichText::new(if hovering { "Drop to open" } else { "Drop an Open Ephys folder here" }).size(19.0).color(LABEL));
                child.add_space(6.0);
                child.label(RichText::new("a day's session, a Record Node, or one recording; and/or one video").color(MUTED));
                child.add_space(14.0);
                child.horizontal(|ui| {
                    let bw = 150.0;
                    ui.add_space((ui.available_width() - 2.0 * bw - 8.0) / 2.0);
                    if ui.add_sized([bw, 26.0], egui::Button::new("Choose folder…")).clicked() {
                        self.choose_folder();
                    }
                    if ui.add_sized([bw, 26.0], egui::Button::new("Choose video…")).clicked() {
                        self.choose_video();
                    }
                });
                if resp.clicked() {
                    self.choose_folder();
                }

                // what's waiting
                if let Some(v) = self.video.clone() {
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        ui.add_space((ui.available_width() - w) / 2.0);
                        ui.label(RichText::new("Video:").color(MUTED));
                        ui.label(RichText::new(v.file_name().unwrap_or_default().to_string_lossy()).color(FG));
                        if ui.small_button("✕").on_hover_text("Don't use this video").clicked() {
                            self.video = None;
                        }
                    });
                }
                if let Some((m, err)) = &self.msg {
                    ui.add_space(10.0);
                    ui.allocate_ui(egui::vec2(w, 0.0), |ui| {
                        ui.label(RichText::new(m).color(if *err { Color32::from_rgb(0xe0, 0x70, 0x60) } else { FG }));
                    });
                }

                ui.add_space(26.0);
                ui.label(RichText::new("or").color(MUTED));
                ui.add_space(10.0);
                match &self.demo {
                    Some((_, t0)) => {
                        ui.horizontal(|ui| {
                            ui.add_space((ui.available_width() - 330.0) / 2.0);
                            ui.spinner();
                            ui.label(RichText::new(format!("Writing the demo data (first time only)… {:.0} s", t0.elapsed().as_secs_f64())).color(FG));
                        });
                    }
                    None => {
                        if ui.add_sized([220.0, 34.0], egui::Button::new(RichText::new("Try the demo").size(16.0))).clicked() {
                            self.start_demo();
                        }
                        ui.label(RichText::new("a synthetic 5-minute recording with a matching video").color(MUTED).size(12.5));
                    }
                }
            });
        });
        self.picker_ui(&ctx);
        self.launch.take()
    }

    fn picker_ui(&mut self, ctx: &egui::Context) {
        let Some(p) = &mut self.picker else { return };
        let n = p.items.len();
        let (mut chosen, mut cancel) = (false, false);
        ctx.input_mut(|i| {
            // in the order pressed, several per frame when keys repeat fast
            for e in std::mem::take(&mut i.events) {
                let step: i64 = match &e {
                    egui::Event::Key { key: Key::ArrowDown, pressed: true, modifiers, .. } if modifiers.is_none() => 1,
                    egui::Event::Key { key: Key::ArrowUp, pressed: true, modifiers, .. } if modifiers.is_none() => -1,
                    egui::Event::Key { key: Key::PageDown, pressed: true, modifiers, .. } if modifiers.is_none() => 8,
                    egui::Event::Key { key: Key::PageUp, pressed: true, modifiers, .. } if modifiers.is_none() => -8,
                    _ => {
                        i.events.push(e);
                        continue;
                    }
                };
                p.sel = (p.sel as i64 + step).clamp(0, n as i64 - 1) as usize;
                p.moved = true;
            }
            if i.consume_key(egui::Modifiers::NONE, Key::End) {
                p.sel = n - 1;
                p.moved = true;
            }
            if i.consume_key(egui::Modifiers::NONE, Key::Home) {
                p.sel = 0;
                p.moved = true;
            }
            chosen = i.consume_key(egui::Modifiers::NONE, Key::Enter);
            cancel = i.consume_key(egui::Modifiers::NONE, Key::Escape);
        });
        let title = match p.what {
            Pick::Recording => format!("{n} recordings: which one?"),
            Pick::Video => format!("{} videos: which one goes with this recording?", n - 1),
        };
        let modal = egui::Modal::new(egui::Id::new("syncviewr-picker")).show(ctx, |ui| {
            ui.set_width(760f32.min(ctx.content_rect().width() - 80.0));
            ui.label(RichText::new(title).size(17.0).color(LABEL));
            ui.label(RichText::new("Up / Down arrows to move, Enter to open, Esc to cancel; or double-click").color(MUTED));
            ui.add_space(8.0);
            egui::ScrollArea::vertical().max_height(ctx.content_rect().height() * 0.6).show(ui, |ui| {
                for (k, it) in p.items.iter().enumerate() {
                    let selected = k == p.sel;
                    let frame = egui::Frame::new()
                        .fill(if selected { ACCENT.gamma_multiply(0.22) } else { Color32::TRANSPARENT })
                        .corner_radius(6.0)
                        .inner_margin(egui::Margin::symmetric(10, 6));
                    let r = frame
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(RichText::new(&it.label).color(if selected { LABEL } else { FG }).monospace());
                            if !it.detail.is_empty() {
                                ui.label(RichText::new(&it.detail).color(MUTED).size(12.5));
                            }
                        })
                        .response
                        .interact(egui::Sense::click());
                    if r.clicked() {
                        p.sel = k;
                    }
                    if r.double_clicked() {
                        p.sel = k;
                        chosen = true;
                    }
                    if selected && p.moved {
                        r.scroll_to_me(None);
                    }
                }
            });
            p.moved = false;
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Open").clicked() {
                    chosen = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
            });
        });
        if modal.should_close() {
            cancel = true;
        }
        if chosen {
            let path = p.items[p.sel].path.clone();
            self.chose(path);
        } else if cancel {
            self.picker = None;
            self.later_videos = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openephys::oebin::{ChannelInfo, ContinuousInfo};

    /// A session folder with two recordings and two videos.
    fn session(dir: &Path) {
        for exp in ["experiment1", "experiment2"] {
            let mut w = openephys::RecordingWriter::create(dir.join("Record Node 101").join(exp).join("recording1"), "1.0.2").unwrap();
            let info = ContinuousInfo {
                folder_name: "Acquisition_Board-100.acquisition_board".into(),
                sample_rate: 1000.0,
                stream_name: Some("acquisition_board".into()),
                num_channels: 1,
                channels: vec![ChannelInfo { channel_name: "CH1".into(), bit_volts: 0.195, ..Default::default() }],
                ..Default::default()
            };
            w.add_continuous(info, &[0; 100], &(0..100).collect::<Vec<_>>(), None).unwrap();
            w.finish().unwrap();
        }
        std::fs::create_dir_all(dir.join("Movies")).unwrap();
        for v in ["a.mp4", "b.mp4"] {
            std::fs::write(dir.join("Movies").join(v), b"").unwrap();
        }
    }

    /// One frame with these keys pressed.
    fn frame(ctx: &egui::Context, s: &mut Splash, keys: &[Key]) -> Option<Launch> {
        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 800.0))),
            ..Default::default()
        };
        for &key in keys {
            input.events.push(egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: Default::default() });
        }
        let mut out = None;
        let _ = ctx.run_ui(input, |ui| out = out.take().or(s.ui(ui)));
        out
    }

    #[test]
    fn pick_recording_then_video_with_keys() {
        let dir = std::env::temp_dir().join(format!("syncviewr_splash_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        session(&dir);
        let ctx = egui::Context::default();
        let mut s = Splash::new(&ctx, dir.clone(), 1);

        s.take_paths(vec![dir.clone()]);
        assert!(frame(&ctx, &mut s, &[]).is_none());
        let p = s.picker.as_ref().unwrap();
        assert!(p.what == Pick::Recording && p.items.len() == 2);
        assert_eq!(p.items[1].label, "Record Node 101/experiment2/recording1");
        assert!(p.items[0].detail.contains("1 ch @ 1 kHz"), "{}", p.items[0].detail);

        // Down, Down (stays on the last), Up, Down → experiment2; Enter → the video list
        assert!(frame(&ctx, &mut s, &[Key::ArrowDown, Key::ArrowDown, Key::ArrowUp, Key::ArrowDown]).is_none());
        assert_eq!(s.picker.as_ref().unwrap().sel, 1);
        assert!(frame(&ctx, &mut s, &[Key::Enter]).is_none());
        let p = s.picker.as_ref().unwrap();
        assert!(p.what == Pick::Video);
        assert_eq!(p.items.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(), ["No video", "Movies/a.mp4", "Movies/b.mp4"]);
        assert_eq!(p.sel, 1);

        // Down, Enter → b.mp4, and the viewer opens
        let l = frame(&ctx, &mut s, &[Key::ArrowDown]).or_else(|| frame(&ctx, &mut s, &[Key::Enter])).unwrap();
        assert!(l.rec.ends_with("experiment2/recording1"));
        assert!(l.video.unwrap().ends_with("Movies/b.mp4"));

        // Esc cancels; "No video" opens without one
        s.take_paths(vec![dir.clone()]);
        frame(&ctx, &mut s, &[Key::Escape]);
        assert!(s.picker.is_none());
        s.video = None;
        s.take_paths(vec![dir.clone()]);
        frame(&ctx, &mut s, &[Key::Enter]);
        let l = frame(&ctx, &mut s, &[Key::Home]).or_else(|| frame(&ctx, &mut s, &[Key::Enter])).unwrap();
        assert!(l.rec.ends_with("experiment1/recording1") && l.video.is_none());

        // two videos dropped at once are refused; one recording folder opens straight away
        s.take_paths(vec![dir.join("Movies/a.mp4"), dir.join("Movies/b.mp4")]);
        assert!(s.msg.as_ref().unwrap().1 && s.video.is_none());
        s.take_paths(vec![dir.join("Record Node 101/experiment1")]);
        let l = frame(&ctx, &mut s, &[]).unwrap();
        assert!(l.rec.ends_with("experiment1/recording1") && l.video.is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
