# syncviewR

A Rust port of [syncview](https://github.com/matthewperkins/syncview): view Open Ephys recordings
side by side with a behaviour video, frame-locked to the camera trigger. Built with
[egui](https://github.com/emilk/egui)/eframe for the interface, [wgpu](https://wgpu.rs) (Vulkan,
Metal, DirectX) for drawing the traces, and FFmpeg (`ffmpeg-next`) for video.

![syncviewR on macOS: video frame-locked to 16 channels of slow-wave and EMG traces, channel table, and
whole-session overview](docs/screenshot-macos.png)

MIT-licensed (see `LICENSE`). Written mostly by an AI model; see [Provenance](#provenance).

**Status: first milestone (core viewer).** It reads the same recordings, presets and cache folder
as the Python version. Not yet ported: the full channel editor (reference channel, band edges,
notch and Y range editing; reordering; saving presets), the Video… button, and clip export.

## Build

Needs Rust (1.85+) and FFmpeg's development libraries.

```bash
# Arch Linux
sudo pacman -S rust ffmpeg clang
# macOS
brew install rust ffmpeg pkg-config

cargo build --release          # -> target/release/syncviewr
```

## Run

```bash
syncviewr --rec "/path/to/Record Node 101/experiment1/recording1" \
          --video /path/to/BASLER_CAM_….mp4 \
          --preset presets.json          # optional; default: all electrode channels as EMG
```

| option | default | |
|---|---|---|
| `--rec FOLDER` | (required) | Open Ephys recording folder (contains `structure.oebin`) |
| `--video FILE` | none | video recorded during this recording |
| `--preset FILE.json` | built from the recording | same JSON as Python syncview |
| `--stream NAME` | `acquisition_board` | Open Ephys continuous stream |
| `--trigger-line N` | `1` | TTL line with one pulse per video frame |
| `--cache FOLDER` | `$SYNCVIEWR_CACHE`, else `~/.cache/syncviewr` (Linux), `~/Library/Caches/syncviewr` (macOS) | filtered traces |
| `--time S`, `--time-base S`, `--play` | | initial position, view width, start playing |

Controls are as in syncview: scroll = zoom time; sideways swipe, Shift+scroll or drag = pan;
⌘/Ctrl+scroll = scale one row's Y, double-click = reset it; click/drag the overview strip to jump;
←/→ one video frame (Shift: 10 % of the view), PgUp/PgDn, Home/End, Space play, [ / ] speed, +/- zoom.

**Sharing a cache with Python syncview.** Cache entries use the same keys and layout, so
`--cache` pointed at a Python syncview cache folder reuses its filtered traces and video indexes
(and vice versa).

## How it draws

Each row is one GPU line strip (`src/gpu.rs`), drawn inside egui's render pass through an
`egui_wgpu` paint callback. As in syncview, a row never has more points than pixel columns: the
whole session is filtered once into a min/max pyramid on disk (`src/cache.rs`), and each frame reads
the coarsest level with at least one block per pixel column. On high-DPI screens each strip is drawn
three times, offset by one physical pixel, to get ~2 px lines.

## Checked

- **Filters vs scipy/Python:** `cargo test` compares every mode (plus notch, odd order and
  high-pass-only variants) with Python syncview's output on 200 s of synthetic data: max relative
  difference 10⁻⁹ (slow mode) to 10⁻¹⁵. Run `tests/make_fixtures.py` with a Python that has
  syncview first to generate the references (55 MB, not in git). Cache keys match Python's exactly.
- **Whole-session cache:** against Python's cache for a 10-minute excerpt: same lengths and pyramid
  depth, identical statistics, differences at float32 rounding. Built 17 traces in 5 s (Python ≈ 33 s).
- **Video:** frames at 14 indexes (incl. both sides of keyframes and the last frame) are the same
  frames Python's decoder returns (mean grey difference 0.005 vs ≥ 0.33 between neighbours).
- **GUI:** Linux (Hyprland/Wayland, Vulkan, NVIDIA T400), 1648×983, 16 rows + video playing at 1×:
  58–60 frames/s (the display's refresh rate). macOS (Apple silicon, Metal, Retina; Homebrew Rust,
  FFmpeg and pkgconf): builds and runs with the 10-minute excerpt; trackpad and keys behave well.
- **Not yet checked:** Windows, multi-hour recordings, detailed trackpad/keyboard feel.

## Provenance

Written by **Claude** (Anthropic), model **Claude Opus 5.5** (`claude-opus-5-5`, training cutoff
June 2026), in Claude Code on 2026-10-07/08, directed by **Matthew Perkins**, who holds the
copyright. It is a port of the Python syncview written the same way; the invariants listed in that
README's Provenance section (sync rule, frame indexing by packet-timestamp rank, cache keys) apply
here too and are what the tests above check.
