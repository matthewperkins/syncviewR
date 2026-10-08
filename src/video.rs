//! Frame-exact video decoding with FFmpeg (port of syncview's CpuDecoder + VideoDecoder thread).
//!
//! Frame index = rank of the packet's presentation timestamp (read from the container by a demux,
//! no decoding), so indexing stays frame-exact even when timestamps jitter. The sorted timestamps
//! are cached as `<cache>/video_index/<stem>.<hash>.npy`, the same file Python syncview writes.

use crate::npy;
use anyhow::{bail, Context, Result};
use ffmpeg_next as ff;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

pub struct Frame {
    pub index: usize,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>, // packed RGB24
}

/// Sorted presentation timestamps of the video stream's packets: entry k is frame k.
pub fn frame_pts(path: &Path, index_dir: &Path) -> Result<Vec<i64>> {
    let path = std::fs::canonicalize(path)?;
    let md = std::fs::metadata(&path)?;
    let mtime_ns = md.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let tag = format!("{}|{}|{}", path.display(), md.len(), mtime_ns);
    let tag = &sha1_smol::Sha1::from(tag.as_bytes()).digest().to_string()[..16];
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let file = index_dir.join(format!("{stem}.{tag}.npy"));
    if let Ok(v) = npy::read_i64(&file) {
        return Ok(v);
    }
    let mut ictx = ff::format::input(&path)?;
    let si = ictx.streams().best(ff::media::Type::Video).context("no video stream")?.index();
    let mut pts: Vec<i64> = ictx
        .packets()
        .filter(|(s, p)| s.index() == si && p.size() > 0)
        .filter_map(|(_, p)| p.pts())
        .collect();
    pts.sort_unstable();
    let save = || -> Result<()> {
        std::fs::create_dir_all(index_dir)?;
        let tmp = file.with_extension(format!("{}.tmp", std::process::id()));
        npy::write_i64(&tmp, &pts)?;
        std::fs::rename(&tmp, &file)?;
        Ok(())
    };
    if let Err(e) = save() {
        eprintln!("syncviewr: could not save the video index to {}: {e:#}", file.display());
    }
    Ok(pts)
}

/// Frames between keyframes if the stream has a fixed GOP, else None (first 2000 packets).
fn keyframe_interval(path: &Path) -> Option<usize> {
    let mut ictx = ff::format::input(&path).ok()?;
    let si = ictx.streams().best(ff::media::Type::Video)?.index();
    let flags: Vec<bool> =
        ictx.packets().filter(|(s, p)| s.index() == si && p.size() > 0).take(2000).map(|(_, p)| p.is_key()).collect();
    let keys: Vec<usize> = flags.iter().enumerate().filter(|(_, k)| **k).map(|(i, _)| i).collect();
    if keys.len() < 3 || keys[0] != 0 {
        return None;
    }
    let d: Vec<usize> = keys.windows(2).map(|w| w[1] - w[0]).collect();
    if d.iter().all(|x| *x == d[0]) { Some(d[0]) } else { None }
}

pub fn video_duration(path: &Path) -> Option<f64> {
    let ictx = ff::format::input(&path).ok()?;
    let d = ictx.duration();
    (d > 0).then(|| d as f64 / ff::ffi::AV_TIME_BASE as f64)
}

struct Decoder {
    ictx: ff::format::context::Input,
    si: usize,
    dec: ff::decoder::Video,
    scaler: Option<ff::software::scaling::Context>,
    pts: Vec<i64>,
    want: usize,
    pending: VecDeque<ff::frame::Video>,
    eof: bool,
}

impl Decoder {
    fn open(path: &Path, pts: Vec<i64>) -> Result<Self> {
        let ictx = ff::format::input(&path)?;
        let stream = ictx.streams().best(ff::media::Type::Video).context("no video stream")?;
        let si = stream.index();
        let mut ctx = ff::codec::context::Context::from_parameters(stream.parameters())?;
        ctx.set_threading(ff::codec::threading::Config { kind: ff::codec::threading::Type::Frame, count: 0, ..Default::default() });
        let dec = ctx.decoder().video()?;
        if pts.is_empty() {
            bail!("no timestamped video packets found");
        }
        let mut d = Decoder { ictx, si, dec, scaler: None, pts, want: 0, pending: VecDeque::new(), eof: false };
        d.seek_to_index(0)?;
        Ok(d)
    }

    /// Lands on the keyframe at/before frame i; next_frames decodes forward and discards up to i.
    fn seek_to_index(&mut self, i: usize) -> Result<()> {
        // av_seek_frame on the video stream, in its time base, backward: what PyAV's container.seek does
        let ret = unsafe {
            ff::ffi::av_seek_frame(self.ictx.as_mut_ptr(), self.si as i32, self.pts[i], ff::ffi::AVSEEK_FLAG_BACKWARD as i32)
        };
        if ret < 0 {
            bail!("seek failed ({})", ff::Error::from(ret));
        }
        self.dec.flush();
        self.pending.clear();
        self.eof = false;
        self.want = i;
        Ok(())
    }

    fn decoded(&mut self) -> Option<ff::frame::Video> {
        loop {
            if let Some(f) = self.pending.pop_front() {
                return Some(f);
            }
            let mut f = ff::frame::Video::empty();
            if self.dec.receive_frame(&mut f).is_ok() {
                return Some(f);
            }
            if self.eof {
                return None;
            }
            // feed the next packet of our stream
            let mut fed = false;
            for (s, p) in self.ictx.packets() {
                if s.index() == self.si {
                    let _ = self.dec.send_packet(&p);
                    fed = true;
                    break;
                }
            }
            if !fed {
                let _ = self.dec.send_eof();
                self.eof = true;
            }
        }
    }

    /// The next k frames (from `want` on), converted to RGB.
    fn next_frames(&mut self, k: usize) -> Result<Vec<Frame>> {
        let mut out = vec![];
        while out.len() < k {
            let Some(f) = self.decoded() else { break };
            let i = match f.pts().or(f.timestamp()) {
                Some(p) => self.pts.partition_point(|x| *x < p),
                None => self.want,
            };
            if i < self.want {
                continue;
            }
            out.push(self.to_rgb(&f, i)?);
            self.want = i + 1;
        }
        Ok(out)
    }

    fn to_rgb(&mut self, f: &ff::frame::Video, index: usize) -> Result<Frame> {
        let (w, h) = (f.width(), f.height());
        let ok = self.scaler.as_ref().is_some_and(|s| s.input().width == w && s.input().height == h && s.input().format == f.format());
        if !ok {
            self.scaler = Some(ff::software::scaling::Context::get(
                f.format(), w, h, ff::format::Pixel::RGB24, w, h, ff::software::scaling::Flags::BILINEAR,
            )?);
        }
        let mut rgb = ff::frame::Video::empty();
        self.scaler.as_mut().unwrap().run(f, &mut rgb)?;
        let (w, h) = (w as usize, h as usize);
        let stride = rgb.stride(0);
        let data = rgb.data(0);
        let mut packed = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            packed.extend_from_slice(&data[row * stride..row * stride + 3 * w]);
        }
        Ok(Frame { index, width: w, height: h, rgb: packed })
    }
}

pub struct VideoInfo {
    pub n_frames: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Default)]
struct Shared {
    request: Option<usize>,
    stop: bool,
    opened: Option<Result<VideoInfo, String>>,
    latest: Option<Arc<Frame>>,
}

/// Decodes frames by index in its own thread; the newest request wins.
///
/// Per request k: cached -> return it; a little ahead of the decoder position -> decode forward;
/// otherwise seek to the keyframe at/before k and decode up to k, caching the run (so stepping
/// backwards inside the GOP is instant).
pub struct VideoDecoder {
    pub path: PathBuf,
    shared: Arc<(Mutex<Shared>, Condvar)>,
}

const CACHE_FRAMES: usize = 160;

impl VideoDecoder {
    pub fn new(path: PathBuf, index_dir: PathBuf, on_change: impl Fn() + Send + 'static) -> Self {
        let shared: Arc<(Mutex<Shared>, Condvar)> = Default::default();
        let sh = shared.clone();
        let p = path.clone();
        std::thread::Builder::new()
            .name("video-decoder".into())
            .spawn(move || {
                let set = |f: &dyn Fn(&mut Shared)| {
                    f(&mut sh.0.lock().unwrap());
                    on_change();
                };
                let opened = (|| -> Result<(Decoder, Option<usize>)> {
                    let pts = frame_pts(&p, &index_dir)?;
                    Ok((Decoder::open(&p, pts)?, keyframe_interval(&p)))
                })();
                let (mut dec, gop) = match opened {
                    Ok(v) => v,
                    Err(e) => {
                        set(&|s| s.opened = Some(Err(format!("could not open video: {e:#}"))));
                        return;
                    }
                };
                let n = dec.pts.len();
                let (w, h) = (dec.dec.width() as usize, dec.dec.height() as usize);
                set(&|s| s.opened = Some(Ok(VideoInfo { n_frames: n, width: w, height: h })));
                let mut cache: HashMap<usize, Arc<Frame>> = HashMap::new();
                let mut order: VecDeque<usize> = VecDeque::new();
                let mut next: Option<usize> = None;
                loop {
                    let k = {
                        let (m, cv) = &*sh;
                        let mut s = m.lock().unwrap();
                        while s.request.is_none() && !s.stop {
                            s = cv.wait(s).unwrap();
                        }
                        if s.stop {
                            return;
                        }
                        s.request.take().unwrap()
                    };
                    if k >= n {
                        continue;
                    }
                    if !cache.contains_key(&k) {
                        let g = gop.unwrap_or(1);
                        let ahead = next.map(|x| k as i64 - x as i64).unwrap_or(-1);
                        let res = (|| -> Result<Vec<Frame>> {
                            let start = if ahead >= 0 && ahead <= (k % g).max(4) as i64 {
                                next.unwrap()
                            } else {
                                let start = if gop.is_some() { k - k % g } else { k };
                                dec.seek_to_index(start)?;
                                start
                            };
                            let frames = dec.next_frames(k - start + 1)?;
                            next = Some(start + frames.len());
                            Ok(frames)
                        })();
                        match res {
                            Ok(frames) => {
                                for f in frames {
                                    let i = f.index;
                                    cache.insert(i, Arc::new(f));
                                    order.retain(|x| *x != i);
                                    order.push_back(i);
                                }
                                while order.len() > CACHE_FRAMES {
                                    if let Some(old) = order.pop_front() {
                                        cache.remove(&old);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("syncviewr: video decode error at frame {k}: {e:#}");
                                next = None;
                            }
                        }
                    }
                    if let Some(f) = cache.get(&k) {
                        order.retain(|x| *x != k);
                        order.push_back(k);
                        let f = f.clone();
                        set(&|s| s.latest = Some(f.clone()));
                    }
                }
            })
            .expect("spawn video decoder");
        VideoDecoder { path, shared }
    }

    pub fn request(&self, k: usize) {
        let (m, cv) = &*self.shared;
        m.lock().unwrap().request = Some(k);
        cv.notify_all();
    }

    /// Ok(info) once opened, Err(message) if opening failed, None while opening.
    pub fn take_opened(&self) -> Option<Result<VideoInfo, String>> {
        self.shared.0.lock().unwrap().opened.take()
    }

    pub fn take_frame(&self) -> Option<Arc<Frame>> {
        self.shared.0.lock().unwrap().latest.take()
    }
}

impl Drop for VideoDecoder {
    fn drop(&mut self) {
        let (m, cv) = &*self.shared;
        m.lock().unwrap().stop = true;
        cv.notify_all();
    }
}

/// Testing aid: decode frames by index (seeking like the viewer thread) and write each as raw RGB24
/// (`<dir>/<index>.rgb`, plus `<dir>/size.txt`).
pub fn dump_frames(path: &Path, indexes: &[usize], dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let pts = frame_pts(path, &std::env::temp_dir().join("syncviewr_test_index"))?;
    let gop = keyframe_interval(path).unwrap_or(1);
    let mut dec = Decoder::open(path, pts)?;
    for &k in indexes {
        let start = k - k % gop;
        dec.seek_to_index(start)?;
        let frames = dec.next_frames(k - start + 1)?;
        let f = frames.last().context("no frame decoded")?;
        if f.index != k {
            bail!("asked for frame {k}, decoder returned {}", f.index);
        }
        std::fs::write(dir.join(format!("{k}.rgb")), &f.rgb)?;
        std::fs::write(dir.join("size.txt"), format!("{} {} {}", f.width, f.height, dec.pts.len()))?;
    }
    Ok(())
}
