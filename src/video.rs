//! Frame-exact video decoding with FFmpeg (port of syncview's CpuDecoder + VideoDecoder thread).
//!
//! Frame index = rank of the packet's presentation timestamp (read from the container by a demux,
//! no decoding), so indexing stays frame-exact even when timestamps jitter. The sorted timestamps
//! are cached as `<cache>/video_index/<stem>.<hash>.npy`, the same file Python syncview writes.
//!
//! On macOS decoding uses VideoToolbox (the Mac's hardware decoder) when it handles the codec, and
//! falls back to software otherwise, or if it fails. Elsewhere it's software unless
//! `SYNCVIEWR_HWDEC=<FFmpeg device type>` (vaapi, cuda, d3d11va, …) asks for hardware, and FFmpeg
//! was built with it; `SYNCVIEWR_HWDEC=off` forces software on macOS.

use crate::npy;
use anyhow::{bail, Context, Result};
use ffmpeg_next as ff;
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString};
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

/// The hardware device type to try, if any (see the module docs).
fn hw_device_type() -> Option<ff::ffi::AVHWDeviceType> {
    use ff::ffi::AVHWDeviceType::*;
    match std::env::var("SYNCVIEWR_HWDEC").ok().as_deref().map(str::trim) {
        Some("" | "0" | "off" | "no" | "none" | "software") => None,
        Some(name) => {
            let c = CString::new(name).ok()?;
            let t = unsafe { ff::ffi::av_hwdevice_find_type_by_name(c.as_ptr()) };
            if t == AV_HWDEVICE_TYPE_NONE {
                eprintln!("syncviewr: SYNCVIEWR_HWDEC={name}: FFmpeg has no such device type; decoding in software");
                return None;
            }
            Some(t)
        }
        None if cfg!(target_os = "macos") => Some(AV_HWDEVICE_TYPE_VIDEOTOOLBOX),
        None => None,
    }
}

/// get_format callback: the hardware format stashed in `opaque` when offered, else FFmpeg's choice.
unsafe extern "C" fn pick_hw_format(ctx: *mut ff::ffi::AVCodecContext, fmts: *const ff::ffi::AVPixelFormat) -> ff::ffi::AVPixelFormat {
    unsafe {
        let want = (*ctx).opaque as isize as i32;
        let mut p = fmts;
        while *p != ff::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *p as i32 == want {
                return *p;
            }
            p = p.add(1);
        }
        ff::ffi::avcodec_default_get_format(ctx, fmts)
    }
}

/// Gives the (unopened) decoder context a hardware device of type `t`, if the codec supports one
/// and the device opens. Returns the device's name.
fn attach_hw(ctx: &mut ff::codec::context::Context, t: ff::ffi::AVHWDeviceType) -> Option<String> {
    unsafe {
        let p = ctx.as_mut_ptr();
        let codec = ff::ffi::avcodec_find_decoder((*p).codec_id);
        if codec.is_null() {
            return None;
        }
        let mut fmt = None;
        for i in 0.. {
            let c = ff::ffi::avcodec_get_hw_config(codec, i);
            if c.is_null() {
                break;
            }
            if (*c).device_type == t && (*c).methods & ff::ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32 != 0 {
                fmt = Some((*c).pix_fmt);
                break;
            }
        }
        let fmt = fmt?;
        let mut dev = std::ptr::null_mut();
        if ff::ffi::av_hwdevice_ctx_create(&mut dev, t, std::ptr::null(), std::ptr::null_mut(), 0) < 0 {
            return None;
        }
        (*p).hw_device_ctx = dev; // the context now owns our reference
        (*p).opaque = fmt as i32 as isize as *mut std::ffi::c_void;
        (*p).get_format = Some(pick_hw_format);
        Some(CStr::from_ptr(ff::ffi::av_hwdevice_get_type_name(t)).to_string_lossy().into_owned())
    }
}

fn is_hw_frame(f: &ff::frame::Video) -> bool {
    unsafe { !(*f.as_ptr()).hw_frames_ctx.is_null() }
}

struct Decoder {
    ictx: ff::format::context::Input,
    si: usize,
    dec: ff::decoder::Video,
    /// The hardware device decoding this stream, None for software.
    hw: Option<String>,
    scaler: Option<ff::software::scaling::Context>,
    pts: Vec<i64>,
    want: usize,
    pending: VecDeque<ff::frame::Video>,
    eof: bool,
    /// After a seek, frames past `want` that arrive before `want` itself are skipped (at most this
    /// many): some hardware decoders return a frame from before the seek.
    stale_guard: u32,
    /// Scale decoded frames to this size (default: as decoded).
    out_size: Option<(u32, u32)>,
}

impl Decoder {
    /// Opens with the hardware decoder if there is one that works, else in software.
    fn open(path: &Path, pts: Vec<i64>) -> Result<Self> {
        let Some(t) = hw_device_type() else { return Self::open_with(path, pts, None) };
        let mut sw = Self::open_with(path, pts.clone(), None)?;
        let hw = Self::open_with(path, pts, Some(t)).and_then(|mut d| {
            if d.hw.is_none() {
                return Ok(None);
            }
            // a hardware decoder can turn out garbage without reporting an error (seen with
            // NVIDIA's on some H.264): check its first frame against software's
            let (a, b) = (d.first_frame()?, sw.first_frame()?);
            let diff = a.rgb.iter().zip(&b.rgb).map(|(x, y)| x.abs_diff(*y) as u64).sum::<u64>() as f64 / a.rgb.len().max(1) as f64;
            if a.rgb.len() != b.rgb.len() || diff > 8.0 {
                bail!("its frames don't match software decoding (mean difference {diff:.0} levels)");
            }
            Ok(Some(d))
        });
        match hw {
            Ok(Some(d)) => Ok(d),
            Ok(None) => Ok(sw),
            Err(e) => {
                eprintln!("syncviewr: not using the hardware decoder: {e:#}; decoding in software");
                Ok(sw)
            }
        }
    }

    /// Frame 0 in RGB, leaving the decoder ready to decode from the start.
    fn first_frame(&mut self) -> Result<Frame> {
        self.seek_to_index(0)?;
        let f = self.decode_until(0, 0)?.pop().context("no frame decoded")?;
        self.seek_to_index(0)?;
        Ok(f)
    }

    fn open_with(path: &Path, pts: Vec<i64>, hw: Option<ff::ffi::AVHWDeviceType>) -> Result<Self> {
        if pts.is_empty() {
            bail!("no timestamped video packets found");
        }
        let ictx = ff::format::input(&path)?;
        let stream = ictx.streams().best(ff::media::Type::Video).context("no video stream")?;
        let si = stream.index();
        let mut ctx = ff::codec::context::Context::from_parameters(stream.parameters())?;
        let hw = hw.and_then(|t| attach_hw(&mut ctx, t));
        if hw.is_none() {
            // hardware decoders work a frame at a time; frame threads only add latency there
            ctx.set_threading(ff::codec::threading::Config { kind: ff::codec::threading::Type::Frame, count: 0, ..Default::default() });
        }
        let dec = ctx.decoder().video()?;
        let mut d = Decoder { ictx, si, dec, hw, scaler: None, pts, want: 0, pending: VecDeque::new(), eof: false, stale_guard: 0, out_size: None };
        d.seek_to_index(0)?;
        if d.hw.is_some() {
            // decode (and download) the first frame now, so a decoder that can't cope fails here
            let f = d.decoded()?.context("no frame decoded")?;
            if !is_hw_frame(&f) {
                d.hw = None; // FFmpeg fell back to software by itself
            } else {
                d.to_rgb(&f, 0)?;
            }
            d.pending.push_back(f);
        }
        Ok(d)
    }

    /// "videotoolbox", "software", …
    fn backend(&self) -> String {
        format!("{} ({})", self.dec.id().name(), self.hw.as_deref().unwrap_or("software"))
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
        self.stale_guard = 8;
        Ok(())
    }

    /// The next decoded frame, None at the end. Software decoding skips over damaged packets; a
    /// hardware decoder's errors are returned, so the caller can switch to software.
    fn decoded(&mut self) -> Result<Option<ff::frame::Video>> {
        loop {
            if let Some(f) = self.pending.pop_front() {
                return Ok(Some(f));
            }
            let mut f = ff::frame::Video::empty();
            match self.dec.receive_frame(&mut f) {
                Ok(()) => return Ok(Some(f)),
                Err(ff::Error::Other { errno: ff::util::error::EAGAIN }) | Err(ff::Error::Eof) => {}
                Err(e) if self.hw.is_some() => bail!("{} decoder: {e}", self.backend()),
                Err(_) => {}
            }
            if self.eof {
                return Ok(None);
            }
            // feed the next packet of our stream
            let mut fed = false;
            for (s, p) in self.ictx.packets() {
                if s.index() == self.si {
                    if let (Err(e), Some(_)) = (self.dec.send_packet(&p), &self.hw) {
                        bail!("{} decoder: {e}", self.backend());
                    }
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

    /// Decodes on (from `want`) to frame `last`, converting the frames from `rgb_from` on to RGB
    /// and returning those; earlier ones are only decoded, which is several times cheaper (and for
    /// a hardware decoder saves the copy out of GPU memory). Stops early at the end of the video.
    fn decode_until(&mut self, last: usize, rgb_from: usize) -> Result<Vec<Frame>> {
        let mut out = vec![];
        while self.want <= last {
            let Some(f) = self.decoded()? else { break };
            let i = match f.pts().or(f.timestamp()) {
                Some(p) => self.pts.partition_point(|x| *x < p),
                None => self.want,
            };
            if i < self.want {
                continue;
            }
            if i > self.want && self.stale_guard > 0 {
                // a frame from before the seek, still in a hardware decoder's queue
                self.stale_guard -= 1;
                continue;
            }
            self.stale_guard = 0;
            if i >= rgb_from {
                out.push(self.to_rgb(&f, i)?);
            }
            self.want = i + 1;
        }
        Ok(out)
    }

    fn to_rgb(&mut self, f: &ff::frame::Video, index: usize) -> Result<Frame> {
        // a hardware frame lives in GPU memory: download it (as NV12 or similar) first
        let sw;
        let f = if is_hw_frame(f) {
            let mut t = ff::frame::Video::empty();
            let r = unsafe { ff::ffi::av_hwframe_transfer_data(t.as_mut_ptr(), f.as_ptr(), 0) };
            if r < 0 {
                bail!("could not copy the frame from the {} decoder ({})", self.backend(), ff::Error::from(r));
            }
            sw = t;
            &sw
        } else {
            f
        };
        let (w, h) = (f.width(), f.height());
        let (ow, oh) = self.out_size.unwrap_or((w, h));
        let ok = self.scaler.as_ref().is_some_and(|s| {
            s.input().width == w && s.input().height == h && s.input().format == f.format() && s.output().width == ow && s.output().height == oh
        });
        if !ok {
            self.scaler = Some(ff::software::scaling::Context::get(
                f.format(), w, h, ff::format::Pixel::RGB24, ow, oh, ff::software::scaling::Flags::BILINEAR,
            )?);
        }
        let mut rgb = ff::frame::Video::empty();
        self.scaler.as_mut().unwrap().run(f, &mut rgb)?;
        let (w, h) = (ow as usize, oh as usize);
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
    /// Codec and decoder, e.g. "hevc (videotoolbox)".
    pub backend: String,
}

/// Decodes up to frame k (seeking unless the decoder is already a little before k) and returns
/// frames k - keep ..= k in RGB; updates `next`, the decoder's position.
fn decode_to(dec: &mut Decoder, k: usize, gop: Option<usize>, next: &mut Option<usize>, keep: usize) -> Result<Vec<Frame>> {
    let g = gop.unwrap_or(1);
    let ahead = next.map(|x| k as i64 - x as i64).unwrap_or(-1);
    if !(0..=(k % g).max(4) as i64).contains(&ahead) {
        *next = None;
        dec.seek_to_index(if gop.is_some() { k - k % g } else { k })?;
    }
    let frames = dec.decode_until(k, k.saturating_sub(keep))?;
    *next = Some(dec.want);
    Ok(frames)
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
                let backend = dec.backend();
                set(&|s| s.opened = Some(Ok(VideoInfo { n_frames: n, width: w, height: h, backend: backend.clone() })));
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
                        // stepping backwards: keep the whole run from the keyframe, so the next
                        // steps back are instant; otherwise just the last few frames
                        let keep = if cache.contains_key(&(k + 1)) { usize::MAX } else { 4 };
                        let mut res = decode_to(&mut dec, k, gop, &mut next, keep);
                        if let (Err(e), Some(_)) = (&res, &dec.hw) {
                            // e.g. the hardware session lost across sleep: carry on in software
                            eprintln!("syncviewr: {e:#}; switching to software decoding");
                            next = None;
                            res = Decoder::open_with(&p, dec.pts.clone(), None).and_then(|d| {
                                dec = d;
                                decode_to(&mut dec, k, gop, &mut next, keep)
                            });
                        }
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

/// Frames `first`, `first + 1`, … in order, scaled to `w` × `h` (for rendering clips).
pub struct FrameReader {
    dec: Decoder,
}

impl FrameReader {
    pub fn open(path: &Path, index_dir: &Path, first: usize, w: u32, h: u32) -> Result<Self> {
        let pts = frame_pts(path, index_dir)?;
        if first >= pts.len() {
            bail!("frame {first} is past the end of the video ({} frames)", pts.len());
        }
        let gop = keyframe_interval(path).unwrap_or(1);
        let mut dec = Decoder::open(path, pts)?;
        dec.out_size = Some((w, h));
        let start = first - first % gop;
        dec.seek_to_index(start)?;
        // decode up to just before `first`; next() then returns frame `first`
        if first > start {
            dec.decode_until(first - 1, first)?;
        }
        Ok(Self { dec })
    }

    /// The video's native size.
    pub fn native_size(path: &Path) -> Result<(u32, u32)> {
        let ictx = ff::format::input(&path)?;
        let s = ictx.streams().best(ff::media::Type::Video).context("no video stream")?;
        let ctx = ff::codec::context::Context::from_parameters(s.parameters())?;
        let v = ctx.decoder().video()?;
        Ok((v.width(), v.height()))
    }

    pub fn next(&mut self) -> Result<Frame> {
        let k = self.dec.want;
        self.dec.decode_until(k, k)?.pop().context("the video ended early")
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
        let frames = dec.decode_until(k, k)?;
        let f = frames.last().context("no frame decoded")?;
        if f.index != k {
            bail!("asked for frame {k}, decoder returned {}", f.index);
        }
        std::fs::write(dir.join(format!("{k}.rgb")), &f.rgb)?;
        std::fs::write(dir.join("size.txt"), format!("{} {} {}", f.width, f.height, dec.pts.len()))?;
    }
    Ok(())
}

/// Testing aid: times software and hardware decoding of `path`, playing forward and jumping to
/// random frames (decoding from the keyframe before each, as when scrubbing), including the
/// conversion to RGB the viewer does.
pub fn bench(path: &Path) -> Result<()> {
    use std::time::Instant;
    let pts = frame_pts(path, &std::env::temp_dir().join("syncviewr_test_index"))?;
    let n = pts.len();
    let gop = keyframe_interval(path);
    println!(
        "{}: {n} frames, keyframe every {} frames",
        path.display(),
        gop.map_or("(irregular)".to_string(), |g| g.to_string())
    );
    let mut kinds = vec![None];
    kinds.extend(hw_device_type().map(Some));
    for t in kinds {
        let t0 = Instant::now();
        let mut dec = Decoder::open_with(path, pts.clone(), t)?;
        if t.is_some() && dec.hw.is_none() {
            println!("  no hardware decoder for this video");
            continue;
        }
        let open_ms = t0.elapsed().as_secs_f64() * 1e3;
        let m = n.min(600);
        let mut next = None;
        let t0 = Instant::now();
        for k in 0..m {
            decode_to(&mut dec, k, gop, &mut next, 0)?;
        }
        let fps = m as f64 / t0.elapsed().as_secs_f64();
        // the same frames decoded only (no download or conversion to RGB)
        dec.seek_to_index(0)?;
        let t0 = Instant::now();
        let mut got = 0;
        while got < m && dec.decoded()?.is_some() {
            got += 1;
        }
        let decode_fps = got as f64 / t0.elapsed().as_secs_f64();
        dec.seek_to_index(0)?;
        let (mut busy, mut got) = (0.0, 0);
        while got < m.min(200) {
            let Some(f) = dec.decoded()? else { break };
            let t0 = Instant::now();
            dec.to_rgb(&f, 0)?;
            busy += t0.elapsed().as_secs_f64();
            got += 1;
        }
        let rgb_ms = busy * 1e3 / got.max(1) as f64;
        let mut x: u64 = 0x2545_f491_4f6c_dd1d;
        let mut jumps = vec![];
        for _ in 0..40 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let k = (x % n as u64) as usize;
            next = None;
            let t0 = Instant::now();
            let f = decode_to(&mut dec, k, gop, &mut next, 4)?;
            jumps.push(t0.elapsed().as_secs_f64() * 1e3);
            if f.last().map(|f| f.index) != Some(k) {
                bail!("asked for frame {k}, got {:?}", f.last().map(|f| f.index));
            }
        }
        let mean = jumps.iter().sum::<f64>() / jumps.len() as f64;
        let worst = jumps.iter().cloned().fold(0.0, f64::max);
        println!(
            "  {:<24} open {open_ms:>4.0} ms   play {fps:>4.0} frames/s (decode alone {decode_fps:>4.0}; to RGB {rgb_ms:.1} ms/frame)   random jump {mean:>4.0} ms mean, {worst:>4.0} ms worst",
            dec.backend()
        );
    }
    Ok(())
}

/// Frames in the video by the container's count, or duration × frame rate when it has none: an
/// estimate, for a quick check before the exact (packet-counting) index is built.
pub fn estimate_frames(path: &Path) -> Option<usize> {
    let ictx = ff::format::input(&path).ok()?;
    let s = ictx.streams().best(ff::media::Type::Video)?;
    if s.frames() > 0 {
        return Some(s.frames() as usize);
    }
    let r = s.avg_frame_rate();
    let d = ictx.duration();
    (r.1 > 0 && d > 0).then(|| (d as f64 / ff::ffi::AV_TIME_BASE as f64 * r.0 as f64 / r.1 as f64).round() as usize)
}

/// One frame from about a tenth of the way in, scaled to fit in `max` × `max` pixels.
pub fn thumbnail(path: &Path, max: u32) -> Result<Frame> {
    let mut ictx = ff::format::input(&path)?;
    let stream = ictx.streams().best(ff::media::Type::Video).context("no video stream")?;
    let si = stream.index();
    let mut dec = ff::codec::context::Context::from_parameters(stream.parameters())?.decoder().video()?;
    let d = ictx.duration();
    if d > 0 {
        let _ = ictx.seek(d / 10, ..d / 10 + 1);
    }
    let mut frame = ff::frame::Video::empty();
    let mut got = false;
    for (s, p) in ictx.packets() {
        if s.index() != si {
            continue;
        }
        let _ = dec.send_packet(&p);
        if dec.receive_frame(&mut frame).is_ok() {
            got = true;
            break;
        }
    }
    if !got {
        let _ = dec.send_eof();
        got = dec.receive_frame(&mut frame).is_ok();
    }
    if !got {
        bail!("no frame could be decoded");
    }
    let (w, h) = (frame.width(), frame.height());
    let k = (max as f64 / w as f64).min(max as f64 / h as f64).min(1.0);
    let (tw, th) = (((w as f64 * k).round() as u32).max(1), ((h as f64 * k).round() as u32).max(1));
    let mut sc = ff::software::scaling::Context::get(frame.format(), w, h, ff::format::Pixel::RGB24, tw, th, ff::software::scaling::Flags::AREA)?;
    let mut rgb = ff::frame::Video::empty();
    sc.run(&frame, &mut rgb)?;
    let (tw, th) = (tw as usize, th as usize);
    let stride = rgb.stride(0);
    let mut packed = Vec::with_capacity(tw * th * 3);
    for row in 0..th {
        packed.extend_from_slice(&rgb.data(0)[row * stride..row * stride + 3 * tw]);
    }
    Ok(Frame { index: 0, width: tw, height: th, rgb: packed })
}
