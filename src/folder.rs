//! `syncviewr FOLDER`: find the recording, video and preset inside a folder, so a shared sample
//! (or a session folder) opens without typing `--rec "Record Node 101/…"` and `--video …`.

use crate::preset::Preset;
use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

const VIDEO_EXT: [&str; 4] = ["mp4", "mov", "avi", "mkv"];

#[derive(Debug, Default)]
pub struct Found {
    pub rec: PathBuf,
    pub video: Option<PathBuf>,
    pub preset: Option<PathBuf>,
}

/// Folders and files under `dir`, `depth` levels deep at most, hidden entries skipped, sorted.
fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')) {
            continue;
        }
        if p.is_dir() {
            out.push(p.clone());
            if depth > 0 {
                walk(&p, depth - 1, out);
            }
        } else {
            out.push(p);
        }
    }
}

fn list(paths: &[PathBuf], root: &Path) -> String {
    paths.iter().map(|p| format!("\n  {}", p.strip_prefix(root).unwrap_or(p).display())).collect()
}

/// The one Open Ephys recording (a folder with structure.oebin) under `dir`, the one video, and the
/// one preset JSON at the top of `dir`. Several candidates are an error naming them, except
/// presets, where `preset.json` wins. `want_video` / `want_preset` false skip those searches.
pub fn resolve(dir: &Path, want_video: bool, want_preset: bool) -> Result<Found> {
    if !dir.is_dir() {
        bail!("{} is not a folder", dir.display());
    }
    let mut all = vec![dir.to_path_buf()];
    walk(dir, 6, &mut all);
    let recs: Vec<PathBuf> = all.iter().filter(|p| p.join("structure.oebin").is_file()).cloned().collect();
    let rec = match recs.as_slice() {
        [] => bail!("no Open Ephys recording (a folder containing structure.oebin) in {}", dir.display()),
        [r] => r.clone(),
        _ => bail!(
            "{} holds several recordings; pick one with --rec:{}",
            dir.display(),
            list(&recs, dir)
        ),
    };
    let mut found = Found { rec, ..Default::default() };
    if want_video {
        let vids: Vec<PathBuf> = all
            .iter()
            .filter(|p| {
                p.is_file()
                    && !p.starts_with(&found.rec)
                    && p.extension().is_some_and(|e| VIDEO_EXT.contains(&e.to_string_lossy().to_lowercase().as_str()))
            })
            .cloned()
            .collect();
        found.video = match vids.as_slice() {
            [] => None,
            [v] => Some(v.clone()),
            _ => bail!("{} holds several videos; pick one with --video:{}", dir.display(), list(&vids, dir)),
        };
    }
    if want_preset {
        let mut presets: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "json") && is_preset(p))
            .collect();
        presets.sort();
        found.preset = match presets.as_slice() {
            [] => None,
            [p] => Some(p.clone()),
            _ => match presets.iter().find(|p| p.file_name().is_some_and(|n| n == "preset.json")) {
                Some(p) => Some(p.clone()),
                None => bail!(
                    "{} holds several presets; pick one with --preset (or name it preset.json):{}",
                    dir.display(),
                    list(&presets, dir)
                ),
            },
        };
    }
    Ok(found)
}

fn is_preset(p: &Path) -> bool {
    std::fs::read_to_string(p)
        .ok()
        .and_then(|s| serde_json::from_str::<Preset>(&s).ok())
        .is_some_and(|p| !p.channels.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(p: &Path, text: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }

    #[test]
    fn finds_recording_video_and_preset() {
        let d = std::env::temp_dir().join(format!("syncviewr_folder_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let rec = d.join("Record Node 101/experiment3/recording1");
        touch(&rec.join("structure.oebin"), "{}");
        touch(&d.join("BASLER_CAM.mp4"), "");
        touch(&d.join("perkins.json"), r#"{"channels": [{"ch": "CH1", "mode": "hilo"}]}"#);
        touch(&d.join("notes.json"), r#"{"something": 1}"#);
        let f = resolve(&d, true, true).unwrap();
        assert_eq!(f.rec, rec);
        assert_eq!(f.video, Some(d.join("BASLER_CAM.mp4")));
        assert_eq!(f.preset, Some(d.join("perkins.json")));
        // the recording folder itself works too, without a video or preset
        let f = resolve(&rec, true, true).unwrap();
        assert_eq!((f.rec, f.video, f.preset), (rec.clone(), None, None));
        // two videos: ask
        touch(&d.join("other.MOV"), "");
        let e = resolve(&d, true, true).unwrap_err().to_string();
        assert!(e.contains("several videos") && e.contains("other.MOV"), "{e}");
        assert!(resolve(&d, false, true).is_ok());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
