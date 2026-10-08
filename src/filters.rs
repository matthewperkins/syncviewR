//! Channel processing modes (port of syncview/core/filters.py).
//!
//! All filters are zero-phase (forward-backward IIR / centred moving average). Every mode maps a raw
//! trace to an output on a *global* grid: absolute row indices that are multiples of the mode's
//! decimation step, so results from different windows/chunks line up exactly.
//!
//! The filter design and `sosfiltfilt` reproduce scipy.signal (butter, iirnotch, sosfiltfilt with
//! odd padding and steady-state initial conditions); tests/fixtures compare against scipy's output.

use crate::preset::Spec;
use anyhow::{bail, Result};
use num_complex::Complex64 as C;
use serde_json::{json, Map, Value};
use std::f64::consts::PI;

pub const MODES: [&str; 4] = ["slow", "hilo", "envelope", "bandpower"];

pub fn mode_defaults(mode: &str) -> Option<Map<String, Value>> {
    let v = match mode {
        // slow GI signals: band-pass, plotted at plot_fs
        "slow" => json!({"band": [0.03, 100.0], "order": 2, "notch": null, "plot_fs": 1000.0, "pad_s": 120.0}),
        // EMG, high-passed, drawn as per-pixel-column min/max
        "hilo" => json!({"band": [150.0, 4000.0], "order": 4, "notch": null, "pad_s": 1.0}),
        // EMG envelope: band-pass -> rectify -> centred boxcar -> low-pass
        "envelope" => json!({"band": [150.0, 4000.0], "order": 4, "notch": null, "smooth_ms": 20.0, "env_lp": 40.0,
                             "plot_fs": 1000.0, "pad_s": 1.0}),
        // slow-wave band power (RMS in a centred moving window)
        "bandpower" => json!({"band": [0.03, 0.25], "order": 2, "notch": null, "win_s": 30.0, "plot_fs": 10.0, "pad_s": 240.0}),
        _ => return None,
    };
    match v {
        Value::Object(m) => Some(m),
        _ => None,
    }
}

/// Mode defaults updated with any (non-null) overrides present in the spec.
pub fn params(spec: &Spec) -> Result<Map<String, Value>> {
    let Some(mut p) = mode_defaults(&spec.mode) else {
        bail!("unknown mode {:?} (choose from {})", spec.mode, MODES.join(", "))
    };
    for (k, v) in p.iter_mut() {
        if let Some(o) = spec.extra.get(k) {
            if !o.is_null() {
                *v = o.clone();
            }
        }
    }
    Ok(p)
}

struct P(Map<String, Value>);

impl P {
    fn f(&self, k: &str) -> f64 {
        self.0.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN)
    }
    fn has(&self, k: &str) -> bool {
        self.0.contains_key(k)
    }
    fn band(&self) -> (Option<f64>, Option<f64>) {
        let b = self.0.get("band").and_then(Value::as_array);
        let get = |i: usize| b.and_then(|b| b.get(i)).and_then(Value::as_f64);
        (get(0), get(1))
    }
    fn order(&self) -> usize {
        self.0.get("order").and_then(Value::as_f64).unwrap_or(2.0) as usize
    }
    fn notch(&self) -> Vec<f64> {
        match self.0.get("notch") {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_f64).collect(),
            Some(v) => v.as_f64().into_iter().filter(|f| *f != 0.0).collect(),
            None => vec![],
        }
    }
}

/// Stable hash of everything that affects the processed trace (not label/colour/ylim). Identical to
/// Python's `spec_key`, so the two programs can share one cache folder.
pub fn spec_key(spec: &Spec, rec_dir: &str) -> String {
    let mut d = params(spec).unwrap_or_default();
    d.remove("pad_s");
    d.insert("rec".into(), Value::String(rec_dir.into()));
    d.insert("ch".into(), Value::String(spec.ch.clone()));
    d.insert("ref".into(), spec.reference.clone().filter(|r| !r.is_empty()).map_or(Value::Null, Value::String));
    d.insert("mode".into(), Value::String(spec.mode.clone()));
    let text = py_json(&Value::Object(d));
    sha1_smol::Sha1::from(text.as_bytes()).digest().to_string()[..16].to_string()
}

/// `json.dumps(v, sort_keys=True)` as Python writes it (separators ", " / ": ", ensure_ascii, float repr).
pub fn py_json(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => if *b { "true" } else { "false" }.into(),
        Value::Number(n) => {
            if n.is_f64() {
                py_float(n.as_f64().unwrap())
            } else {
                n.to_string()
            }
        }
        Value::String(s) => py_str(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let items: Vec<String> = keys.iter().map(|k| format!("{}: {}", py_str(k), py_json(&m[*k]))).collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

fn py_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Python's repr(float): shortest round-trip digits, fixed notation for 1e-4 <= |x| < 1e16.
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let sci = format!("{:e}", x); // shortest round-trip digits, e.g. "-1.5e-5"
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        let s = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        };
        format!("{sign}{s}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{sign}{m}e{}{:02}", if exp < 0 { "-" } else { "+" }, exp.abs())
    }
}

fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Decimation step (in raw samples) of the mode's output.
pub fn out_step(spec: &Spec, fs: f64) -> usize {
    let p = P(params(spec).unwrap_or_default());
    if p.has("plot_fs") {
        py_round(fs / p.f("plot_fs")).max(1) as usize
    } else {
        1
    }
}

pub fn pad_samples(spec: &Spec, fs: f64) -> usize {
    let p = P(params(spec).unwrap_or_default());
    (p.f("pad_s") * fs) as usize
}

// ------------------------------------------------------------------ filter design

pub type Sos = Vec<[f64; 6]>;

/// scipy.signal.butter(order, band, btype, fs=fs, output="sos") for the band given as (lo, hi);
/// None when neither edge applies (no filtering).
pub fn butter_sos(order: usize, lo: Option<f64>, hi: Option<f64>, fs: f64) -> Option<Sos> {
    let lo = lo.filter(|v| *v != 0.0 && !v.is_nan());
    let hi = hi.filter(|v| *v != 0.0 && !v.is_nan() && *v < 0.49 * fs);
    let warp = |f: f64| 4.0 * (PI * (2.0 * f / fs) / 2.0).tan(); // scipy pre-warps at internal fs = 2
    let n = order as i64;
    let mut p: Vec<C> = (0..order)
        .map(|i| -(C::i() * PI * (-n + 1 + 2 * i as i64) as f64 / (2 * n) as f64).exp())
        .collect();
    let mut z: Vec<C> = vec![];
    let mut k = 1.0;
    match (lo, hi) {
        (Some(lo), Some(hi)) => {
            let (w0, w1) = (warp(lo), warp(hi));
            let (bw, wo) = (w1 - w0, (w0 * w1).sqrt());
            // lp2bp_zpk
            let deg = p.len() - z.len();
            let zl: Vec<C> = z.iter().map(|v| v * bw / 2.0).collect();
            let pl: Vec<C> = p.iter().map(|v| v * bw / 2.0).collect();
            let split = |v: &[C]| -> Vec<C> {
                let mut out: Vec<C> = v.iter().map(|x| x + (x * x - wo * wo).sqrt()).collect();
                out.extend(v.iter().map(|x| x - (x * x - wo * wo).sqrt()));
                out
            };
            z = split(&zl);
            z.extend(std::iter::repeat_n(C::new(0.0, 0.0), deg));
            p = split(&pl);
            k *= bw.powi(deg as i32);
        }
        (Some(lo), None) => {
            // lp2hp_zpk
            let wo = warp(lo);
            let deg = p.len() - z.len();
            let kz: C = z.iter().map(|v| -v).product();
            let kp: C = p.iter().map(|v| -v).product();
            k *= (kz / kp).re;
            z = z.iter().map(|v| wo / v).collect();
            z.extend(std::iter::repeat_n(C::new(0.0, 0.0), deg));
            p = p.iter().map(|v| wo / v).collect();
        }
        (None, Some(hi)) => {
            // lp2lp_zpk
            let wo = warp(hi);
            let deg = p.len() - z.len();
            z = z.iter().map(|v| v * wo).collect();
            p = p.iter().map(|v| v * wo).collect();
            k *= wo.powi(deg as i32);
        }
        (None, None) => return None,
    }
    // bilinear_zpk at fs = 2
    let fs2 = 4.0;
    let deg = p.len() - z.len();
    let kz: C = z.iter().map(|v| fs2 - v).product();
    let kp: C = p.iter().map(|v| fs2 - v).product();
    k *= (kz / kp).re;
    let mut zd: Vec<C> = z.iter().map(|v| (fs2 + v) / (fs2 - v)).collect();
    zd.extend(std::iter::repeat_n(C::new(-1.0, 0.0), deg));
    let pd: Vec<C> = p.iter().map(|v| (fs2 + v) / (fs2 - v)).collect();
    Some(zpk2sos(&zd, &pd, k))
}

/// Group zeros and poles into second-order sections (conjugate pairs together); gain in the first section.
fn zpk2sos(z: &[C], p: &[C], k: f64) -> Sos {
    let pairs = |v: &[C]| -> Vec<[f64; 3]> {
        let tol = 1e-9;
        let mut cplx: Vec<C> = v.iter().filter(|x| x.im > tol).copied().collect();
        cplx.sort_by(|a, b| a.norm().total_cmp(&b.norm()));
        let mut real: Vec<f64> = v.iter().filter(|x| x.im.abs() <= tol).map(|x| x.re).collect();
        real.sort_by(|a, b| a.total_cmp(b));
        let mut out: Vec<[f64; 3]> = cplx.iter().map(|c| [1.0, -2.0 * c.re, c.norm_sqr()]).collect();
        // pair the smallest real root with the largest (keeps +1/-1 zeros of a band-pass together)
        while real.len() >= 2 {
            let a = real.remove(0);
            let b = real.pop().unwrap();
            out.push([1.0, -(a + b), a * b]);
        }
        if let Some(a) = real.pop() {
            out.push([1.0, -a, 0.0]);
        }
        out
    };
    let mut a = pairs(p);
    // poles closest to the unit circle last (as scipy does)
    a.sort_by(|x, y| x[2].abs().total_cmp(&y[2].abs()));
    let b = pairs(z);
    let n = a.len().max(b.len());
    let mut sos: Sos = (0..n)
        .map(|i| {
            let bi = b.get(i).copied().unwrap_or([1.0, 0.0, 0.0]);
            let ai = a.get(i).copied().unwrap_or([1.0, 0.0, 0.0]);
            [bi[0], bi[1], bi[2], ai[0], ai[1], ai[2]]
        })
        .collect();
    for v in &mut sos[0][..3] {
        *v *= k;
    }
    sos
}

/// scipy.signal.iirnotch(f0, q, fs) as one section.
pub fn iirnotch(f0: f64, q: f64, fs: f64) -> [f64; 6] {
    let w0 = 2.0 * f0 / fs;
    let bw = w0 / q * PI;
    let w0 = w0 * PI;
    let beta = (bw / 2.0).tan(); // gb = 1/sqrt(2)
    let gain = 1.0 / (1.0 + beta);
    [gain, -2.0 * gain * w0.cos(), gain, 1.0, -2.0 * gain * w0.cos(), 2.0 * gain - 1.0]
}

fn sosfilt_zi(sos: &Sos) -> Vec<[f64; 2]> {
    let mut scale = 1.0;
    sos.iter()
        .map(|s| {
            let (b0, b1, b2, a1, a2) = (s[0], s[1], s[2], s[4], s[5]);
            let (bb0, bb1) = (b1 - a1 * b0, b2 - a2 * b0);
            let z0 = (bb0 + bb1) / (1.0 + a1 + a2);
            let z1 = bb1 - a2 * z0;
            let out = [scale * z0, scale * z1];
            scale *= (b0 + b1 + b2) / (1.0 + a1 + a2);
            out
        })
        .collect()
}

fn sosfilt_inplace(sos: &Sos, x: &mut [f64], zi: &[[f64; 2]], x0: f64) {
    for (s, z) in sos.iter().zip(zi) {
        let (b0, b1, b2, a1, a2) = (s[0], s[1], s[2], s[4], s[5]);
        let (mut z0, mut z1) = (z[0] * x0, z[1] * x0);
        for v in x.iter_mut() {
            let xi = *v;
            let y = b0 * xi + z0;
            z0 = b1 * xi - a1 * y + z1;
            z1 = b2 * xi - a2 * y;
            *v = y;
        }
    }
}

/// scipy.signal.sosfiltfilt(sos, x) (padtype "odd", default padlen). Returns x unchanged if it is
/// too short for the padding (scipy raises).
pub fn sosfiltfilt(sos: &Sos, x: &[f64]) -> Vec<f64> {
    let nzb = sos.iter().filter(|s| s[2] == 0.0).count();
    let nza = sos.iter().filter(|s| s[5] == 0.0).count();
    let ntaps = 2 * sos.len() + 1 - nzb.min(nza);
    let edge = 3 * ntaps;
    let n = x.len();
    if n <= edge {
        return x.to_vec();
    }
    // odd extension
    let mut ext = Vec::with_capacity(n + 2 * edge);
    ext.extend((1..=edge).rev().map(|i| 2.0 * x[0] - x[i]));
    ext.extend_from_slice(x);
    ext.extend((0..edge).map(|i| 2.0 * x[n - 1] - x[n - 2 - i]));
    let zi = sosfilt_zi(sos);
    let x0 = ext[0];
    sosfilt_inplace(sos, &mut ext, &zi, x0);
    ext.reverse();
    let y0 = ext[0];
    sosfilt_inplace(sos, &mut ext, &zi, y0);
    ext.reverse();
    ext[edge..edge + n].to_vec()
}

fn filt(x: Vec<f64>, fs: f64, lo: Option<f64>, hi: Option<f64>, order: usize) -> Vec<f64> {
    match butter_sos(order, lo, hi, fs) {
        Some(sos) => sosfiltfilt(&sos, &x),
        None => x,
    }
}

/// scipy.ndimage.uniform_filter1d(x, size) with mode "reflect" (centred window, odd size).
pub fn uniform_filter1d(x: &[f64], size: usize) -> Vec<f64> {
    let n = x.len() as i64;
    if n == 0 {
        return vec![];
    }
    let h = (size / 2) as i64;
    let left = h; // window [i - h, i - h + size)
    let refl = |mut i: i64| -> f64 {
        let p = 2 * n;
        i = i.rem_euclid(p);
        x[(if i < n { i } else { p - 1 - i }) as usize]
    };
    let mut cum = Vec::with_capacity((n + size as i64) as usize + 1);
    cum.push(0.0);
    let mut s = 0.0;
    for j in -left..n - left + size as i64 {
        s += refl(j);
        cum.push(s);
    }
    (0..n as usize).map(|i| (cum[i + size] - cum[i]) / size as f64).collect()
}

/// y sampled at absolute indices a + k*step: keep only samples on the global grid of step*factor.
fn decimate(y: Vec<f64>, a: i64, step: i64, factor: i64) -> Result<(Vec<f64>, i64, i64)> {
    let new = step * factor;
    let r = (-a).rem_euclid(new);
    if r % step != 0 {
        bail!("misaligned decimation");
    }
    let k0 = (r / step) as usize;
    Ok((y.into_iter().skip(k0).step_by(factor as usize).collect(), a + k0 as i64 * step, new))
}

/// Apply the spec's mode to raw trace x (physical units) whose first sample is absolute row a.
/// Returns (y, a_out, step): y[k] belongs to absolute row a_out + k*step.
pub fn process(x: Vec<f64>, a: i64, fs: f64, spec: &Spec) -> Result<(Vec<f64>, i64, i64)> {
    let p = P(params(spec)?);
    let sub: Vec<f64> = x.iter().step_by((x.len() / 10000).max(1)).copied().collect();
    let med = crate::oe::median(&sub);
    let mut x: Vec<f64> = x.into_iter().map(|v| v - med).collect();
    for f0 in p.notch() {
        x = sosfiltfilt(&vec![iirnotch(f0, 30.0, fs)], &x);
    }
    let (lo, hi) = p.band();
    let order = p.order();
    match spec.mode.as_str() {
        "hilo" => Ok((filt(x, fs, lo, hi, order), a, 1)),
        "envelope" => {
            let y = filt(x, fs, lo, hi, order);
            let n = (py_round(p.f("smooth_ms") * 1e-3 * fs) | 1).max(1) as usize;
            let y: Vec<f64> = y.into_iter().map(f64::abs).collect();
            let y = uniform_filter1d(&y, n);
            let y = filt(y, fs, None, Some(p.f("env_lp")), 4);
            decimate(y, a, 1, out_step(spec, fs) as i64)
        }
        "slow" => {
            let y = filt(x, fs, None, hi, order); // low-pass at full rate
            let (y, a, step) = decimate(y, a, 1, out_step(spec, fs) as i64)?;
            let y = filt(y, fs / step as f64, lo, None, order); // high-pass at the plot rate
            Ok((y, a, step))
        }
        "bandpower" => {
            let total = out_step(spec, fs) as i64;
            let f1 = if total % 10 == 0 && total > 10 { 10 } else { 1 };
            let y = if f1 > 1 { filt(x, fs, None, Some(0.4 * fs / f1 as f64), 4) } else { x };
            let (y, a, step) = decimate(y, a, 1, f1)?;
            let f2 = total / step;
            let y = filt(y, fs / step as f64, None, Some(0.4 * fs / total as f64), 4);
            let (y, a, step) = decimate(y, a, step, f2)?;
            let y = filt(y, fs / step as f64, lo, hi, order);
            let n = (py_round(p.f("win_s") * fs / step as f64) | 1).max(1) as usize;
            let sq: Vec<f64> = y.iter().map(|v| v * v).collect();
            Ok((uniform_filter1d(&sq, n).into_iter().map(f64::sqrt).collect(), a, step))
        }
        m => bail!("unknown mode {m:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> Option<(serde_json::Value, std::path::PathBuf)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let cases = std::fs::read_to_string(dir.join("cases.json")).ok()?;
        Some((serde_json::from_str(&cases).unwrap(), dir))
    }

    /// Every mode against Python syncview's output (tests/make_fixtures.py writes the references).
    #[test]
    fn matches_python_process() {
        let Some((c, dir)) = fixtures() else {
            eprintln!("no fixtures: run tests/make_fixtures.py");
            return;
        };
        let (fs, bv, a) = (c["fs"].as_f64().unwrap(), c["bit_volts"].as_f64().unwrap(), c["a"].as_i64().unwrap());
        let x: Vec<f64> = crate::npy::read_i64(&dir.join("raw.npy")).unwrap().into_iter().map(|v| v as f64 * bv).collect();
        for case in c["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let spec: Spec = serde_json::from_value(case["spec"].clone()).unwrap();
            let want = crate::npy::read_f64(&dir.join(format!("{name}.npy"))).unwrap();
            let (y, a_out, step) = process(x.clone(), a, fs, &spec).unwrap();
            assert_eq!((a_out, step, y.len()), (case["a_out"].as_i64().unwrap(), case["step"].as_i64().unwrap(), want.len()), "{name}");
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let err = y.iter().zip(&want).fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
            eprintln!("{name:14} max |rust - python| = {err:.3e}  (signal max {scale:.3e}, relative {:.1e})", err / scale);
            assert!(err <= 1e-7 * scale, "{name}: relative error {:.2e}", err / scale);
        }
    }

    #[test]
    fn matches_python_spec_key() {
        let Some((c, _)) = fixtures() else { return };
        for k in c["keys"].as_array().unwrap() {
            let spec: Spec = serde_json::from_value(k["spec"].clone()).unwrap();
            assert_eq!(spec_key(&spec, k["rec"].as_str().unwrap()), k["key"].as_str().unwrap(), "{}", k["spec"]);
        }
    }

    #[test]
    fn python_float_repr() {
        for (x, s) in [(150.0, "150.0"), (0.03, "0.03"), (4000.0, "4000.0"), (1e-5, "1e-05"), (0.0001, "0.0001"),
                       (1e16, "1e+16"), (123456789.5, "123456789.5"), (-2.5, "-2.5"), (1.5e-7, "1.5e-07")] {
            assert_eq!(py_float(x), s);
        }
    }
}
