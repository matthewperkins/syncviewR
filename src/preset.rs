//! Channel specs and presets (same JSON as the Python syncview).

use crate::filters::MODES;
use crate::oe::Recording;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const MAX_SHOWN: usize = 16; // rows shown by the recording-derived preset; the rest start hidden

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Spec {
    pub ch: String,
    #[serde(rename = "ref", default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default = "yes")]
    pub show: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ylim: Option<[f64; 2]>,
    /// Filter overrides (band, order, notch, plot_fs, …) and anything else in the JSON.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

fn yes() -> bool {
    true
}

impl Spec {
    pub fn new(ch: &str, mode: &str) -> Self {
        Spec { ch: ch.into(), reference: None, mode: mode.into(), label: None, show: true, color: None, ylim: None, extra: Map::new() }
    }

    pub fn title(&self) -> String {
        self.label.clone().filter(|l| !l.is_empty()).unwrap_or_else(|| self.ch.clone())
    }

    /// The spec without its display-only fields (what the processed trace depends on).
    pub fn processing(&self) -> Spec {
        Spec { label: None, show: true, color: None, ylim: None, ..self.clone() }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Preset {
    #[serde(default)]
    pub time_base: Option<f64>,
    #[serde(default)]
    pub channels: Vec<Spec>,
    #[serde(default)]
    pub overview: Option<Spec>,
}

/// Default preset: every electrode channel (Open Ephys type 0 / uV; ADC and AUX skipped), labelled by
/// channel name, drawn as EMG (hilo); the first MAX_SHOWN rows visible.
pub fn from_recording(rec: &Recording) -> Preset {
    let mut chans: Vec<&String> = rec.ch_names.iter().zip(&rec.ch_types).filter(|(_, t)| **t == Some(0)).map(|(n, _)| n).collect();
    if chans.is_empty() {
        chans = rec.ch_names.iter().zip(&rec.units).filter(|(_, u)| *u == "uV").map(|(n, _)| n).collect();
    }
    if chans.is_empty() {
        chans = rec.ch_names.iter().collect();
    }
    let channels = chans
        .iter()
        .enumerate()
        .map(|(i, n)| Spec { label: Some((*n).clone()), show: i < MAX_SHOWN, ..Spec::new(n, "hilo") })
        .collect();
    Preset { time_base: Some(10.0), channels, overview: None }
}

/// Drop rows (and the overview) that refer to channels or modes this recording doesn't have.
pub fn check(mut p: Preset, rec: &Recording) -> (Preset, Vec<String>) {
    let bad = |s: &Spec| -> Option<String> {
        let missing: Vec<&str> = [Some(&s.ch), s.reference.as_ref()]
            .into_iter()
            .flatten()
            .filter(|c| !c.is_empty() && rec.ch(c).is_none())
            .map(|c| c.as_str())
            .collect();
        if !missing.is_empty() {
            return Some(format!("channel {} not in this recording", missing.join(", ")));
        }
        if !MODES.contains(&s.mode.as_str()) {
            return Some(format!("unknown mode {:?}", s.mode));
        }
        None
    };
    let mut problems = vec![];
    p.channels.retain(|s| match bad(s) {
        Some(why) => {
            problems.push(format!("row {:?} skipped: {why}", s.title()));
            false
        }
        None => true,
    });
    if let Some(o) = &p.overview {
        if let Some(why) = bad(o) {
            problems.push(format!("overview {:?} skipped: {why}", o.title()));
            p.overview = None;
        }
    }
    (p, problems)
}
