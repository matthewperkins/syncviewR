"""Write reference outputs from Python syncview for the Rust tests (run with a Python that has syncview).

    python tests/make_fixtures.py   ->  tests/fixtures/*.npy + cases.json
"""
import json
from pathlib import Path

import numpy as np
from syncview.core.filters import process, spec_key

out = Path(__file__).parent / "fixtures"
out.mkdir(exist_ok=True)
fs, bit_volts, a = 10000.0, 0.195, 123457
rng = np.random.default_rng(1)
n = 2_000_000                                  # 200 s
t = np.arange(n) / fs
x = (rng.normal(0, 40, n) + 300 * np.sin(2 * np.pi * 0.08 * t) + 80 * np.sin(2 * np.pi * 60 * t)
     + np.cumsum(rng.normal(0, 2, n)))
bursts = rng.integers(0, n - 2000, 60)
for b in bursts:
    x[b:b + 2000] += rng.normal(0, 400, 2000)
raw = np.clip(np.round(x / bit_volts), -32768, 32767).astype("<i2")
np.save(out / "raw.npy", raw)
xs = raw.astype(np.float64) * bit_volts

cases = [
    dict(name="hilo", spec=dict(ch="CH1", mode="hilo")),
    dict(name="slow", spec=dict(ch="CH1", mode="slow")),
    dict(name="envelope", spec=dict(ch="CH1", mode="envelope")),
    dict(name="bandpower", spec=dict(ch="CH1", mode="bandpower")),
    dict(name="hilo_notch", spec=dict(ch="CH1", mode="hilo", band=[100, 3000], notch=[60.0, 120])),
    dict(name="slow_odd", spec=dict(ch="CH1", mode="slow", band=[0.05, 50.0], order=3)),
    dict(name="hilo_hp_only", spec=dict(ch="CH1", mode="hilo", band=[300.0, None])),
]
for c in cases:
    y, a_out, step = process(xs, a, fs, c["spec"])
    np.save(out / f"{c['name']}.npy", y.astype("<f8"))
    c.update(a_out=int(a_out), step=int(step), n=len(y))
keys = [dict(spec=s, rec=r, key=spec_key(s, r)) for s, r in [
    (dict(ch="CH13", mode="hilo"), "/data/Record Node 101/experiment1/recording1"),
    (dict(ch="CH5", ref="CH6", mode="slow", band=[0.05, 50]), "/data/rec"),
    (dict(ch="CH5", mode="bandpower", label="ignored", show=False), "/données/réc"),
    (dict(ch="CH1", mode="envelope", smooth_ms=10, notch=60.0, ylim=[0, 5]), "/x"),
    (dict(ch="CH2", mode="slow", band=[1e-5, 100.0], plot_fs=500), "/x"),
]]
json.dump(dict(fs=fs, bit_volts=bit_volts, a=a, cases=cases, keys=keys), open(out / "cases.json", "w"), indent=1)
print("wrote", out)
