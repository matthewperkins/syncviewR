//! Whole-session processed traces on disk + a min/max pyramid for constant-cost drawing at any zoom
//! (port of syncview/data/cache.py; same folder layout, so Python and Rust can share a cache).
//!
//! Layout:  <root>/<spec_key>/
//!             level0.npy             processed trace, float32, sample k <-> absolute row k*step
//!             min{k}.npy max{k}.npy  min/max over blocks of FACTOR**k level-0 samples (k >= 1)
//!             meta.json              written last; its presence marks a complete trace

use crate::filters::{out_step, pad_samples, process, spec_key};
use crate::npy::{self, F32Array};
use crate::oe::Recording;
use crate::preset::Spec;
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

pub const FACTOR: usize = 8;
const CHUNK_S: f64 = 600.0; // seconds of data processed per chunk (plus filter padding on each side)

pub struct Trace {
    pub y: F32Array,
    pub dt: f64,
    pub t0: f64,
    pub mins: Vec<F32Array>, // level k >= 1 (level 0 is y)
    pub maxs: Vec<F32Array>,
    pub stats: Map<String, Value>,
}

impl Trace {
    /// In-memory trace (e.g. a window processed on demand), with its pyramid.
    pub fn from_samples(t0: f64, dt: f64, y: Vec<f32>) -> Self {
        let (mut mins, mut maxs) = (vec![], vec![]);
        let (mut mn, mut mx) = (y.clone(), y.clone());
        while mn.len() / FACTOR >= 256 {
            (mn, mx) = reduce(&mn, &mx);
            mins.push(F32Array::Owned(mn.clone()));
            maxs.push(F32Array::Owned(mx.clone()));
        }
        let stats = stats(&y);
        Trace { y: F32Array::Owned(y), dt, t0, mins, maxs, stats }
    }

    fn open(folder: &Path) -> Result<Self> {
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(folder.join("meta.json"))?)?;
        let levels = meta["levels"].as_u64().context("meta.json: levels")? as usize;
        let mut mins = vec![];
        let mut maxs = vec![];
        for k in 1..=levels {
            mins.push(npy::load_f32(&folder.join(format!("min{k}.npy")))?);
            maxs.push(npy::load_f32(&folder.join(format!("max{k}.npy")))?);
        }
        Ok(Trace {
            y: npy::load_f32(&folder.join("level0.npy"))?,
            dt: meta["dt"].as_f64().context("meta.json: dt")?,
            t0: 0.0,
            mins,
            maxs,
            stats: meta["stats"].as_object().cloned().unwrap_or_default(),
        })
    }

    pub fn stat(&self, k: &str) -> Option<f64> {
        self.stats.get(k).and_then(Value::as_f64)
    }

    /// Points to draw for [t_lo, t_hi] on n_px pixel columns, appended to (xs, ys).
    ///
    /// Few samples per pixel -> the samples themselves. Otherwise min and max per pixel column
    /// (interleaved lo,hi,lo,hi…) from the coarsest pyramid level that still has >= 1 block per
    /// column. Columns sit on a grid anchored at t=0, so panning doesn't make them flicker.
    pub fn view(&self, t_lo: f64, t_hi: f64, n_px: f64, xs: &mut Vec<f64>, ys: &mut Vec<f32>) {
        let n_px = n_px.max(16.0);
        let off = self.t0;
        let (t_lo, t_hi) = (t_lo - off, t_hi - off);
        let span = t_hi - t_lo;
        let per_px = span / self.dt / n_px;
        let y = self.y.as_slice();
        if per_px < 2.0 {
            let a = ((t_lo / self.dt).floor() as i64 - 1).max(0) as usize;
            let b = (((t_hi / self.dt).ceil() as i64 + 2).max(0) as usize).min(y.len());
            for i in a..b.max(a) {
                xs.push(off + i as f64 * self.dt);
                ys.push(y[i]);
            }
            return;
        }
        let k = ((per_px.ln() / (FACTOR as f64).ln()) as usize).min(self.mins.len());
        let blk = self.dt * (FACTOR as f64).powi(k as i32);
        let (mn, mx) = if k == 0 { (y, y) } else { (self.mins[k - 1].as_slice(), self.maxs[k - 1].as_slice()) };
        let w = span / n_px;
        let m0 = (t_lo / w).floor() as i64 - 1;
        let m1 = (t_hi / w).ceil() as i64 + 1;
        let edge = |m: i64| -> usize { ((m as f64 * w / blk).round_ties_even().max(0.0) as usize).min(mn.len()) };
        let mut e0 = edge(m0);
        for m in m0..m1 {
            let e1 = edge(m + 1);
            if e1 > e0 {
                let lo = mn[e0..e1].iter().copied().fold(f32::INFINITY, f32::min);
                let hi = mx[e0..e1].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let t = off + (m as f64 + 0.5) * w;
                xs.extend([t, t]);
                ys.extend([lo, hi]);
            }
            e0 = e1;
        }
    }
}

fn reduce(mn: &[f32], mx: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let nmn = mn.chunks(FACTOR).map(|c| c.iter().copied().fold(f32::INFINITY, f32::min)).collect();
    let nmx = mx.chunks(FACTOR).map(|c| c.iter().copied().fold(f32::NEG_INFINITY, f32::max)).collect();
    (nmn, nmx)
}

/// Percentiles of (a subsample of) y, as Python's cache writes them.
fn stats(y: &[f32]) -> Map<String, Value> {
    let mut sub: Vec<f64> = y.iter().step_by((y.len() / 2_000_000).max(1)).map(|v| *v as f64).filter(|v| v.is_finite()).collect();
    let mut out = Map::new();
    if sub.is_empty() {
        return out;
    }
    sub.sort_by(|a, b| a.total_cmp(b));
    let pct = |s: &[f64], q: f64| {
        let pos = q / 100.0 * (s.len() - 1) as f64;
        let (i, f) = (pos.floor() as usize, pos - pos.floor());
        if i + 1 < s.len() { s[i] + (s[i + 1] - s[i]) * f } else { s[i] }
    };
    for (q, name) in [(0.1, "0.1"), (0.5, "0.5"), (1.0, "1"), (50.0, "50"), (99.0, "99"), (99.5, "99.5"), (99.9, "99.9")] {
        out.insert(name.into(), json!(pct(&sub, q)));
    }
    let mut abs: Vec<f64> = sub.iter().map(|v| v.abs()).collect();
    abs.sort_by(|a, b| a.total_cmp(b));
    out.insert("absmax99.9".into(), json!(pct(&abs, 99.9)));
    out
}

pub struct TraceCache {
    pub rec: Arc<Recording>,
    pub root: PathBuf,
    rec_dir: String,
    open: Mutex<HashMap<String, Arc<Trace>>>,
}

impl TraceCache {
    pub fn new(rec: Arc<Recording>, root: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&root).with_context(|| format!("creating cache folder {}", root.display()))?;
        let rec_dir = rec.rec_dir.to_string_lossy().into_owned();
        Ok(TraceCache { rec, root, rec_dir, open: Mutex::new(HashMap::new()) })
    }

    pub fn key(&self, spec: &Spec) -> String {
        spec_key(spec, &self.rec_dir)
    }

    /// The finished trace for spec, or None if it hasn't been built.
    pub fn get(&self, spec: &Spec) -> Option<Arc<Trace>> {
        let key = self.key(spec);
        let mut open = self.open.lock().unwrap();
        if let Some(t) = open.get(&key) {
            return Some(t.clone());
        }
        let folder = self.root.join(&key);
        if !folder.join("meta.json").exists() {
            return None;
        }
        match Trace::open(&folder) {
            Ok(t) => {
                let t = Arc::new(t);
                open.insert(key, t.clone());
                Some(t)
            }
            Err(e) => {
                eprintln!("syncviewr: unreadable cache entry {} ({e:#}); rebuilding", folder.display());
                let _ = std::fs::remove_file(folder.join("meta.json"));
                None
            }
        }
    }

    /// Forget open traces except those of specs.
    pub fn retain(&self, specs: &[Spec]) {
        let keep: HashSet<String> = specs.iter().map(|s| self.key(s)).collect();
        self.open.lock().unwrap().retain(|k, _| keep.contains(k));
    }

    pub fn missing(&self, specs: &[Spec]) -> Vec<Spec> {
        let mut seen = HashSet::new();
        specs.iter().filter(|s| self.get(s).is_none() && seen.insert(self.key(s))).cloned().collect()
    }

    /// Process every not-yet-cached spec over the whole recording. progress(fraction) after each
    /// chunk; cancel() returning true aborts (partial results are removed). Returns Ok(false) if cancelled.
    pub fn build(&self, specs: &[Spec], progress: &dyn Fn(f64), cancel: &dyn Fn() -> bool) -> Result<bool> {
        let specs = self.missing(specs);
        if specs.is_empty() {
            return Ok(true);
        }
        let rec = &self.rec;
        let fs = rec.fs;
        let n = rec.n_samples();
        let steps: Vec<usize> = specs.iter().map(|s| out_step(s, fs)).collect();
        let align = steps.iter().fold(1usize, |a, &b| a / gcd(a, b) * b);
        let chunk = align.max((CHUNK_S * fs) as usize / align * align);
        let pad = specs.iter().map(|s| pad_samples(s, fs)).max().unwrap_or(0) as i64;
        let tmp: Vec<PathBuf> = specs.iter().map(|s| self.root.join(format!("{}.building", self.key(s)))).collect();
        let result = (|| -> Result<bool> {
            // outputs go straight into memory-mapped level0.npy files (16 two-hour EMG traces would be ~5 GB)
            let mut maps = vec![];
            for (d, st) in tmp.iter().zip(&steps) {
                let _ = std::fs::remove_dir_all(d);
                std::fs::create_dir_all(d)?;
                maps.push(npy::create_f32_mmap(&d.join("level0.npy"), n.div_ceil(*st))?);
            }
            // the channels the specs need, each read once per chunk
            let mut chans: Vec<usize> = vec![];
            let mut wanted: Vec<(usize, Option<usize>)> = vec![];
            for s in &specs {
                let ch = rec.ch(&s.ch).with_context(|| format!("unknown channel {}", s.ch))?;
                let r = s.reference.as_deref().filter(|r| !r.is_empty()).and_then(|r| rec.ch(r));
                for c in std::iter::once(ch).chain(r) {
                    if !chans.contains(&c) {
                        chans.push(c);
                    }
                }
                let col = |c: usize| chans.iter().position(|x| *x == c).unwrap();
                wanted.push((col(ch), r.map(col)));
            }
            let bv: Vec<f64> = chans.iter().map(|c| rec.bit_volts(*c)).collect();
            let mut c0 = 0;
            while c0 < n {
                if cancel() {
                    return Ok(false);
                }
                let c1 = (c0 + chunk).min(n);
                let (first, len) = rec.clip(c0 as i64 - pad, c1 as i64 + pad);
                let cols = rec.raw_columns(&chans, first, len);
                maps.par_iter_mut().zip(&specs).zip(&steps).zip(&wanted).try_for_each(|(((map, s), &st), &(a, r))| -> Result<()> {
                    let out = map.as_mut_slice();
                    // same arithmetic as Recording::trace, so cached values don't change
                    let x: Vec<f64> = match r {
                        None => cols[a].iter().map(|v| *v as f64 * bv[a]).collect(),
                        Some(b) => cols[a].iter().zip(&cols[b]).map(|(u, w)| *u as f64 * bv[a] - *w as f64 * bv[b]).collect(),
                    };
                    let (y, a, st2) = process(x, first as i64, fs, s)?;
                    debug_assert_eq!(st2 as usize, st);
                    let k0 = ((c0 as i64 - a) / st as i64).max(0) as usize;
                    let k1 = (k0 + (c1 - c0).div_ceil(st)).min(y.len());
                    let o0 = c0 / st;
                    for (j, v) in y[k0..k1].iter().enumerate() {
                        if o0 + j < out.len() {
                            out[o0 + j] = *v as f32;
                        }
                    }
                    Ok(())
                })?;
                drop(cols);
                progress(c1 as f64 / n as f64);
                c0 = c1;
            }
            // finish each trace (zoom pyramid, stats, metadata, move into place) in parallel; the
            // mapped level0 files need no flush, as readers map the same pages
            if cancel() {
                return Ok(false);
            }
            specs.par_iter().zip(&steps).zip(&tmp).zip(&maps).try_for_each(|(((s, st), d), map)| -> Result<()> {
                let y = map.as_slice();
                let levels = build_pyramid(y, d)?;
                let mut spec_json = serde_json::to_value(s.processing())?;
                if let Value::Object(m) = &mut spec_json {
                    m.remove("show");
                }
                let meta = json!({"spec": spec_json, "step": st, "dt": *st as f64 / fs, "n": y.len(), "levels": levels,
                                  "stats": stats(y), "built": format!("syncviewr {}", env!("CARGO_PKG_VERSION"))});
                std::fs::write(d.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
                let fin = self.root.join(self.key(s));
                let _ = std::fs::remove_dir_all(&fin);
                std::fs::rename(d, &fin)?;
                Ok(())
            })?;
            Ok(true)
        })();
        if !matches!(result, Ok(true)) {
            for d in &tmp {
                let _ = std::fs::remove_dir_all(d);
            }
        }
        result
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn build_pyramid(y: &[f32], folder: &Path) -> Result<usize> {
    if y.len() / FACTOR < 256 {
        return Ok(0);
    }
    let (mut mn, mut mx) = reduce(y, y);
    let mut k = 1;
    npy::write_f32(&folder.join("min1.npy"), &mn)?;
    npy::write_f32(&folder.join("max1.npy"), &mx)?;
    while mn.len() / FACTOR >= 256 {
        (mn, mx) = reduce(&mn, &mx);
        k += 1;
        npy::write_f32(&folder.join(format!("min{k}.npy")), &mn)?;
        npy::write_f32(&folder.join(format!("max{k}.npy")), &mx)?;
    }
    Ok(k)
}

// ------------------------------------------------------------------ background workers

/// Builds whole-session caches in a background thread; a newer request supersedes the current one.
pub struct CacheBuilder {
    state: Arc<(Mutex<BuildState>, Condvar)>,
    pub progress: Arc<Mutex<Option<f64>>>, // Some(fraction) while building
    thread: Option<std::thread::JoinHandle<()>>,
}

#[derive(Default)]
struct BuildState {
    want: Option<Vec<Spec>>,
    wanted_keys: HashSet<String>,
    stop: bool,
}

impl CacheBuilder {
    pub fn new(cache: Arc<TraceCache>, on_change: impl Fn() + Send + 'static) -> Self {
        let state: Arc<(Mutex<BuildState>, Condvar)> = Default::default();
        let progress = Arc::new(Mutex::new(None));
        let (st, pr) = (state.clone(), progress.clone());
        let thread = std::thread::Builder::new()
            .name("cache-builder".into())
            .spawn(move || loop {
                let specs = {
                    let (m, cv) = &*st;
                    let mut s = m.lock().unwrap();
                    while s.want.is_none() && !s.stop {
                        s = cv.wait(s).unwrap();
                    }
                    if s.stop {
                        return;
                    }
                    s.want.take().unwrap()
                };
                let todo = cache.missing(&specs);
                if !todo.is_empty() {
                    let keys: HashSet<String> = todo.iter().map(|s| cache.key(s)).collect();
                    let cancel = || {
                        let s = st.0.lock().unwrap();
                        s.stop || !keys.is_subset(&s.wanted_keys)
                    };
                    let report = |f: f64| {
                        *pr.lock().unwrap() = Some(f);
                        on_change();
                    };
                    report(0.0);
                    if let Err(e) = cache.build(&todo, &report, &cancel) {
                        eprintln!("syncviewr: cache build failed: {e:#}");
                    }
                }
                *pr.lock().unwrap() = None;
                on_change();
            })
            .expect("spawn cache builder");
        CacheBuilder { state, progress, thread: Some(thread) }
    }

    pub fn request(&self, specs: Vec<Spec>, cache: &TraceCache) {
        let (m, cv) = &*self.state;
        let mut s = m.lock().unwrap();
        s.wanted_keys = specs.iter().map(|x| cache.key(x)).collect();
        s.want = Some(specs);
        cv.notify_all();
    }
}

impl Drop for CacheBuilder {
    fn drop(&mut self) {
        {
            let (m, cv) = &*self.state;
            m.lock().unwrap().stop = true;
            cv.notify_all();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Processes just a window of data for specs that aren't cached yet (fast feedback while the
/// whole-session build runs). Results land in `results`, keyed by cache key.
pub struct WindowWorker {
    job: Arc<(Mutex<Option<(Vec<Spec>, f64, f64)>>, Condvar)>,
    pub results: Arc<Mutex<HashMap<String, (f64, f64, Arc<Trace>)>>>,
    stop: Arc<AtomicBool>,
}

impl WindowWorker {
    pub fn new(cache: Arc<TraceCache>, on_change: impl Fn() + Send + 'static) -> Self {
        let job: Arc<(Mutex<Option<(Vec<Spec>, f64, f64)>>, Condvar)> = Default::default();
        let results: Arc<Mutex<HashMap<String, (f64, f64, Arc<Trace>)>>> = Default::default();
        let stop = Arc::new(AtomicBool::new(false));
        let (jb, res, stp) = (job.clone(), results.clone(), stop.clone());
        std::thread::Builder::new()
            .name("window-worker".into())
            .spawn(move || loop {
                let (specs, t_lo, t_hi) = {
                    let (m, cv) = &*jb;
                    let mut j = m.lock().unwrap();
                    while j.is_none() && !stp.load(Ordering::Relaxed) {
                        j = cv.wait(j).unwrap();
                    }
                    if stp.load(Ordering::Relaxed) {
                        return;
                    }
                    j.take().unwrap()
                };
                let rec = &cache.rec;
                let fs = rec.fs;
                let i0 = (t_lo.max(0.0) * fs) as i64;
                let i1 = (t_hi.min(rec.duration()) * fs) as i64;
                for s in &specs {
                    if jb.0.lock().unwrap().is_some() {
                        break; // superseded
                    }
                    let pad = pad_samples(s, fs) as i64;
                    let (first, len) = rec.clip(i0 - pad, i1 + pad);
                    let Some(ch) = rec.ch(&s.ch) else { continue };
                    let r = s.reference.as_deref().filter(|r| !r.is_empty()).and_then(|r| rec.ch(r));
                    match process(rec.trace(ch, r, first, len), first as i64, fs, s) {
                        Ok((y, a, step)) => {
                            let k0 = ((i0 - a).max(0) as usize).div_ceil(step as usize);
                            let k1 = (((i1 - a).max(0) as usize).div_ceil(step as usize)).min(y.len());
                            if k1 > k0 + 1 {
                                let t0 = (a + k0 as i64 * step) as f64 / fs;
                                let ys: Vec<f32> = y[k0..k1].iter().map(|v| *v as f32).collect();
                                let tr = Trace::from_samples(t0, step as f64 / fs, ys);
                                res.lock().unwrap().insert(cache.key(s), (t_lo, t_hi, Arc::new(tr)));
                                on_change();
                            }
                        }
                        Err(e) => eprintln!("syncviewr: {e:#}"),
                    }
                }
            })
            .expect("spawn window worker");
        WindowWorker { job, results, stop }
    }

    pub fn request(&self, specs: Vec<Spec>, t_lo: f64, t_hi: f64) {
        let (m, cv) = &*self.job;
        *m.lock().unwrap() = Some((specs, t_lo, t_hi));
        cv.notify_all();
    }
}

impl Drop for WindowWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.job.1.notify_all();
    }
}

/// Cache root: --cache > $SYNCVIEWR_CACHE > the platform's per-user cache folder.
pub fn default_root() -> PathBuf {
    if let Some(p) = std::env::var_os("SYNCVIEWR_CACHE") {
        return PathBuf::from(p);
    }
    platform_root()
}

pub fn platform_root() -> PathBuf {
    let home = || PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(home)
    } else if cfg!(target_os = "macos") {
        home().join("Library/Caches")
    } else {
        std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).unwrap_or_else(|| home().join(".cache"))
    };
    base.join("syncviewr")
}
