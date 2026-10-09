//! Minimal NumPy `.npy` reading and writing: 1-D, little-endian numeric arrays only,
//! which is all Open Ephys and the syncview cache use.

use anyhow::{bail, Context, Result};
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

pub fn write_f32(path: &Path, data: &[f32]) -> Result<()> {
    let mut f = std::io::BufWriter::new(File::create(path)?);
    f.write_all(&header_bytes("<f4", data.len()))?;
    f.write_all(bytemuck::cast_slice(data))?;
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

/// A new float32 .npy file of `len` NaNs, memory-mapped for writing.
pub struct F32MapMut {
    map: MmapMut,
    offset: usize,
    len: usize,
}

impl F32MapMut {
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        bytemuck::cast_slice_mut(&mut self.map[self.offset..self.offset + 4 * self.len])
    }
    pub fn as_slice(&self) -> &[f32] {
        bytemuck::cast_slice(&self.map[self.offset..self.offset + 4 * self.len])
    }
    pub fn flush(&self) -> Result<()> {
        Ok(self.map.flush()?)
    }
}

pub fn create_f32_mmap(path: &Path, len: usize) -> Result<F32MapMut> {
    let header = header_bytes("<f4", len);
    let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;
    file.set_len((header.len() + 4 * len) as u64)?;
    // SAFETY: the file was just created by us and nothing else maps it.
    let mut map = unsafe { MmapMut::map_mut(&file)? };
    map[..header.len()].copy_from_slice(&header);
    let mut out = F32MapMut { offset: header.len(), len, map };
    out.as_mut_slice().fill(f32::NAN);
    Ok(out)
}
