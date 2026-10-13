//! Minimal NumPy `.npy` reading and writing: 1-D, little-endian numeric arrays only,
//! which is all Open Ephys and the syncview cache use.

use anyhow::{bail, Context, Result};
use half::f16;
use memmap2::{Mmap, MmapMut};
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub struct Header {
    pub descr: String,
    pub len: usize,
    pub offset: usize,
}

pub fn parse_header(bytes: &[u8]) -> Result<Header> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not a .npy file");
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes(bytes[8..12].try_into()?) as usize, 12),
        v => bail!("unsupported .npy version {v}"),
    };
    let text = std::str::from_utf8(&bytes[start..start + hlen])?;
    let field = |name: &str| -> Result<&str> {
        let i = text.find(name).with_context(|| format!("no {name} in .npy header"))?;
        Ok(text[i + name.len()..].trim_start_matches([':', ' ']))
    };
    let descr = field("'descr'")?;
    let descr = descr.trim_start_matches('\'');
    let descr = descr[..descr.find('\'').context("bad descr")?].to_string();
    if field("'fortran_order'")?.starts_with("True") {
        bail!("fortran-order .npy not supported");
    }
    let shape = field("'shape'")?;
    let shape = &shape[1..shape.find(')').context("bad shape")?];
    let dims: Vec<usize> = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<Result<_, _>>()?;
    Ok(Header { descr, len: dims.iter().product(), offset: start + hlen })
}

/// Any integer or float 1-D array, converted to f64 / i64.
#[cfg(test)]
pub fn read_f64(path: &Path) -> Result<Vec<f64>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let h = parse_header(&bytes)?;
    convert(&bytes[h.offset..], &h.descr, h.len, path)
}

pub fn read_i64(path: &Path) -> Result<Vec<i64>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let h = parse_header(&bytes)?;
    let d = &bytes[h.offset..];
    macro_rules! ints {
        ($t:ty) => {{
            let w = std::mem::size_of::<$t>();
            (0..h.len).map(|i| <$t>::from_le_bytes(d[i * w..(i + 1) * w].try_into().unwrap()) as i64).collect()
        }};
    }
    Ok(match h.descr.as_str() {
        "<i8" => ints!(i64),
        "<u8" => ints!(u64),
        "<i4" => ints!(i32),
        "<u4" => ints!(u32),
        "<i2" => ints!(i16),
        "<u2" => ints!(u16),
        "|i1" => ints!(i8),
        "|u1" => ints!(u8),
        _ => convert(d, &h.descr, h.len, path)?.into_iter().map(|v| v as i64).collect(),
    })
}

fn convert(d: &[u8], descr: &str, n: usize, path: &Path) -> Result<Vec<f64>> {
    macro_rules! nums {
        ($t:ty) => {{
            let w = std::mem::size_of::<$t>();
            if d.len() < n * w {
                bail!("{} is truncated", path.display());
            }
            (0..n).map(|i| <$t>::from_le_bytes(d[i * w..(i + 1) * w].try_into().unwrap()) as f64).collect()
        }};
    }
    Ok(match descr {
        "<f8" => nums!(f64),
        "<f4" => nums!(f32),
        "<i8" => nums!(i64),
        "<u8" => nums!(u64),
        "<i4" => nums!(i32),
        "<u4" => nums!(u32),
        "<i2" => nums!(i16),
        "<u2" => nums!(u16),
        "|i1" => nums!(i8),
        "|u1" => nums!(u8),
        other => bail!("{}: unsupported dtype {other}", path.display()),
    })
}

/// A float32 array, memory-mapped (large files) or in memory (small ones).
pub enum F32Array {
    Mapped { map: Mmap, offset: usize, len: usize },
    Owned(Vec<f32>),
}

impl F32Array {
    pub fn as_slice(&self) -> &[f32] {
        match self {
            F32Array::Mapped { map, offset, len } => bytemuck::cast_slice(&map[*offset..*offset + 4 * *len]),
            F32Array::Owned(v) => v,
        }
    }
}

/// Files at least this big are memory-mapped; smaller ones are read into memory.
const MMAP_MIN_BYTES: u64 = 2 << 20;

pub fn load_f32(path: &Path) -> Result<F32Array> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let size = file.metadata()?.len();
    if size >= MMAP_MIN_BYTES {
        // SAFETY: cache files are written once to a temporary folder and renamed into place; never modified after.
        let map = unsafe { Mmap::map(&file)? };
        let h = parse_header(&map)?;
        if h.descr != "<f4" || h.offset % 4 != 0 || map.len() < h.offset + 4 * h.len {
            bail!("{}: expected a float32 array", path.display());
        }
        Ok(F32Array::Mapped { offset: h.offset, len: h.len, map })
    } else {
        let bytes = std::fs::read(path)?;
        let h = parse_header(&bytes)?;
        if h.descr != "<f4" {
            bail!("{}: expected a float32 array", path.display());
        }
        let d = &bytes[h.offset..h.offset + 4 * h.len];
        Ok(F32Array::Owned(d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()))
    }
}

fn header_bytes(descr: &str, len: usize) -> Vec<u8> {
    let mut dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({len},), }}");
    let total = 10 + dict.len() + 1;
    dict.push_str(&" ".repeat((64 - total % 64) % 64));
    dict.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out
}

/// Writes values as float16 (rounding to nearest; infinities and NaN kept).
pub fn write_f16(path: &Path, data: &[f32]) -> Result<()> {
    let mut f = std::io::BufWriter::new(File::create(path)?);
    f.write_all(&header_bytes("<f2", data.len()))?;
    for v in data {
        f.write_all(&f16::from_f32(*v).to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

pub fn write_i64(path: &Path, data: &[i64]) -> Result<()> {
    let mut f = std::io::BufWriter::new(File::create(path)?);
    f.write_all(&header_bytes("<i8", data.len()))?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

/// A new float16 .npy file of `len` NaNs, memory-mapped for writing.
pub struct F16MapMut {
    map: MmapMut,
    offset: usize,
    len: usize,
}

impl F16MapMut {
    pub fn as_mut_slice(&mut self) -> &mut [f16] {
        bytemuck::cast_slice_mut(&mut self.map[self.offset..self.offset + 2 * self.len])
    }
    pub fn as_slice(&self) -> &[f16] {
        bytemuck::cast_slice(&self.map[self.offset..self.offset + 2 * self.len])
    }
}

pub fn create_f16_mmap(path: &Path, len: usize) -> Result<F16MapMut> {
    let header = header_bytes("<f2", len);
    let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;
    file.set_len((header.len() + 2 * len) as u64)?;
    // SAFETY: the file was just created by us and nothing else maps it.
    let mut map = unsafe { MmapMut::map_mut(&file)? };
    map[..header.len()].copy_from_slice(&header);
    let mut out = F16MapMut { offset: header.len(), len, map };
    out.as_mut_slice().fill(f16::NAN);
    Ok(out)
}

/// The nearest float16, saturating at its largest finite value (±65504) instead of overflowing
/// to infinity. NaN stays NaN.
pub fn to_f16(v: f64) -> f16 {
    f16::from_f64(v.clamp(-65504.0, 65504.0))
}

/// A float32 or float16 array (memory-mapped or in memory), read as f32.
pub enum Samples {
    F32(F32Array),
    F16Mapped { map: Mmap, offset: usize, len: usize },
    F16Owned(Vec<f16>),
}

/// A borrowed view of `Samples`.
#[derive(Clone, Copy)]
pub enum SampleSlice<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
}

impl Samples {
    pub fn slice(&self) -> SampleSlice<'_> {
        match self {
            Samples::F32(a) => SampleSlice::F32(a.as_slice()),
            Samples::F16Mapped { map, offset, len } => SampleSlice::F16(bytemuck::cast_slice(&map[*offset..*offset + 2 * *len])),
            Samples::F16Owned(v) => SampleSlice::F16(v),
        }
    }
}

impl SampleSlice<'_> {
    pub fn len(&self) -> usize {
        match self {
            SampleSlice::F32(v) => v.len(),
            SampleSlice::F16(v) => v.len(),
        }
    }
    pub fn get(&self, i: usize) -> f32 {
        match self {
            SampleSlice::F32(v) => v[i],
            SampleSlice::F16(v) => v[i].to_f32(),
        }
    }
    /// (min, max) of [a, b), ignoring NaN; (inf, -inf) if empty.
    pub fn min_max(&self, a: usize, b: usize) -> (f32, f32) {
        let init = (f32::INFINITY, f32::NEG_INFINITY);
        let f = |(lo, hi): (f32, f32), v: f32| (lo.min(v), hi.max(v));
        match self {
            SampleSlice::F32(v) => v[a..b].iter().copied().fold(init, f),
            SampleSlice::F16(v) => v[a..b].iter().map(|x| x.to_f32()).fold(init, f),
        }
    }
}

/// A float32 or float16 .npy file.
pub fn load_samples(path: &Path) -> Result<Samples> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let size = file.metadata()?.len();
    let head = {
        let mut b = vec![0u8; size.min(4096) as usize];
        std::io::Read::read_exact(&mut &file, &mut b)?;
        parse_header(&b)?
    };
    match head.descr.as_str() {
        "<f4" => Ok(Samples::F32(load_f32(path)?)),
        "<f2" if size >= MMAP_MIN_BYTES => {
            // SAFETY: as in load_f32
            let map = unsafe { Mmap::map(&file)? };
            if head.offset % 2 != 0 || map.len() < head.offset + 2 * head.len {
                bail!("{} is truncated", path.display());
            }
            Ok(Samples::F16Mapped { map, offset: head.offset, len: head.len })
        }
        "<f2" => {
            let bytes = std::fs::read(path)?;
            let d = bytes.get(head.offset..head.offset + 2 * head.len).with_context(|| format!("{} is truncated", path.display()))?;
            Ok(Samples::F16Owned(d.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]])).collect()))
        }
        other => bail!("{}: expected float32 or float16, not {other}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float16_files() {
        let d = std::env::temp_dir().join(format!("syncviewr_npy_test_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        // small (read into memory) and large (memory-mapped) files, written both ways
        for n in [1000, (MMAP_MIN_BYTES / 2 + 10) as usize] {
            let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 300.0).collect();
            let want: Vec<f32> = v.iter().map(|x| f16::from_f32(*x).to_f32()).collect();
            write_f16(&d.join("a.npy"), &v).unwrap();
            let mut m = create_f16_mmap(&d.join("b.npy"), n).unwrap();
            for (o, x) in m.as_mut_slice().iter_mut().zip(&v) {
                *o = to_f16(*x as f64);
            }
            drop(m);
            for f in ["a.npy", "b.npy"] {
                let s = load_samples(&d.join(f)).unwrap();
                let s = s.slice();
                assert_eq!(s.len(), n);
                assert!((0..n).all(|i| s.get(i) == want[i]), "{f}, n = {n}");
                let lo = want[10..20].iter().copied().fold(f32::INFINITY, f32::min);
                let hi = want[10..20].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                assert_eq!(s.min_max(10, 20), (lo, hi));
            }
        }
        // out of range saturates, NaN survives
        assert_eq!(to_f16(1e9).to_f32(), 65504.0);
        assert_eq!(to_f16(-1e9).to_f32(), -65504.0);
        assert!(to_f16(f64::NAN).is_nan());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
