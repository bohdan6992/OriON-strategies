"""Stage 2: one row per (ticker, date): pre-market features (04:00-09:20) + the 09:30 -> 10:00 targets.
Everything is computed with segmented numpy reductions over the sorted window parquet."""
import glob, sys, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
t0 = time.time()
parts = sorted(glob.glob(S + "win/part_*.parquet"))
tbl = pa.concat_tables([pq.read_table(p) for p in parts])
N = tbl.num_rows
print("rows", f"{N:,}", f"{time.time()-t0:.0f}s", flush=True)
enc = tbl["ticker"].combine_chunks().dictionary_encode()
tk = enc.indices.to_numpy(zero_copy_only=False).astype(np.int32)
tick_names = np.array(enc.dictionary.to_pylist(), dtype=object)
dt_ = pc.cast(pc.replace_substring(tbl["date"].combine_chunks(), "-", ""), pa.int32()).to_numpy(zero_copy_only=False)
mn = tbl["minute"].to_numpy().astype(np.int16)
def col(n): return tbl[n].to_numpy().astype(np.float32)
o, h, l, c, v, pcl = col("o"), col("h"), col("l"), col("c"), col("v"), col("prev_close")
bench, sig, beta, sig_d, beta_d = col("Bench%"), col("sigma"), col("beta"), col("sigma_day"), col("beta_day")
del tbl, enc

newg = np.ones(N, bool)
newg[1:] = (tk[1:] != tk[:-1]) | (dt_[1:] != dt_[:-1])
starts = np.flatnonzero(newg)
G = len(starts)
print("ticker-days", f"{G:,}", f"{time.time()-t0:.0f}s", flush=True)
# monotone minutes inside a group?
bad = ((mn[1:] <= mn[:-1]) & ~newg[1:]).sum()
print("non-increasing minutes inside a group (should be 0):", bad, flush=True)

idx = np.arange(N, dtype=np.int64)
big = N + 10

def cnt(mask): return np.add.reduceat(mask.astype(np.int32), starts)
def first_idx(mask): return np.minimum.reduceat(np.where(mask, idx, big), starts)
def last_idx(mask): return np.maximum.reduceat(np.where(mask, idx, -1), starts)
def take(arr, ix, dtype=np.float64):
    out = np.full(len(ix), np.nan, dtype)
    ok = (ix >= 0) & (ix < N)
    out[ok] = arr[ix[ok]]
    return out

PM = (mn >= 240) & (mn <= 560)            # 04:00 - 09:20
PRE = (mn >= 561) & (mn <= 569)           # 09:21 - 09:29
ENT = (mn == 570)
RTH30 = (mn >= 570) & (mn <= 599)         # 09:30 - 09:59
F = {}
F["tk"] = tk[starts]; F["date"] = dt_[starts]
F["prev_close"] = pcl[starts].astype(np.float64)
F["sigma"] = sig[starts].astype(np.float64); F["beta"] = beta[starts].astype(np.float64)
F["sigma_day"] = sig_d[starts].astype(np.float64); F["beta_day"] = beta_d[starts].astype(np.float64)

# ---- premarket 04:00-09:20
F["n_pm"] = cnt(PM)
li, fi = last_idx(PM), first_idx(PM)
F["t_first"] = take(mn, np.where(fi >= big, -1, fi)); F["t_last"] = take(mn, li)
F["p_first"] = take(o, np.where(fi >= big, -1, fi)); F["p_last"] = take(c, li)
vm = np.where(PM, v, 0).astype(np.float64)
F["pm_vol"] = np.add.reduceat(vm, starts)
F["pm_dollar"] = np.add.reduceat(vm * c, starts)
F["pm_hi"] = np.maximum.reduceat(np.where(PM, h, -np.inf).astype(np.float64), starts)
F["pm_lo"] = np.minimum.reduceat(np.where(PM, l, np.inf).astype(np.float64), starts)
for k in ("pm_hi", "pm_lo"):
    F[k][~np.isfinite(F[k])] = np.nan
# time of the high / low (first bar that printed it)
gid = np.repeat(np.arange(G), np.diff(np.append(starts, N)))
hi_b = PM & (h.astype(np.float64) >= np.nan_to_num(F["pm_hi"], nan=np.inf)[gid] - 1e-9)
lo_b = PM & (l.astype(np.float64) <= np.nan_to_num(F["pm_lo"], nan=-np.inf)[gid] + 1e-9)
F["t_hi"] = take(mn, np.where(first_idx(hi_b) >= big, -1, first_idx(hi_b)))
F["t_lo"] = take(mn, np.where(first_idx(lo_b) >= big, -1, first_idx(lo_b)))
# price checkpoints (last print at or before ...)
for name, X in (("p_0600", 360), ("p_0700", 420), ("p_0800", 480), ("p_0830", 510), ("p_0900", 540), ("p_0910", 550)):
    F[name] = take(c, last_idx((mn >= 240) & (mn <= X)))
F["up_bars"] = cnt(PM & (c > o)); F["dn_bars"] = cnt(PM & (c < o))
mv = np.where(PM & (o > 0), np.abs(c / np.where(o > 0, o, 1) - 1) * 100, 0)
F["max_bar_move"] = np.maximum.reduceat(mv.astype(np.float64), starts)
# bench (sector ETF) % vs its prev close, last non-null <= 09:20 / <= 09:29 / <= 09:59
bn = ~np.isnan(bench)
F["bench_0920"] = take(bench, last_idx(PM & bn)); F["bench_pre"] = take(bench, last_idx((mn <= 569) & (mn >= 240) & bn))
F["bench_exit"] = take(bench, last_idx((mn <= 599) & (mn >= 240) & bn))
# ---- 09:21-09:29 and the entry bar
F["n_pre"] = cnt(PRE); F["p_0929"] = take(c, last_idx(PRE | PM))
ei = last_idx(ENT)
F["has_open"] = (ei >= 0).astype(np.int8)
F["o_0930"] = take(o, ei); F["h_0930"] = take(h, ei); F["l_0930"] = take(l, ei); F["c_0930"] = take(c, ei); F["v_0930"] = take(v, ei)
# ---- exits
xi = last_idx(RTH30)                         # last print before 10:00
F["t_exit"] = take(mn, xi); F["p_exit"] = take(c, xi)
oi = last_idx(mn == 600)
F["o_1000"] = take(o, oi)
for name, X in (("p_0945", 584), ("p_0935", 574), ("p_0940", 579), ("p_0950", 589)):
    F[name] = take(c, last_idx((mn >= 570) & (mn <= X)))
F["rth_vol"] = np.add.reduceat(np.where(RTH30, v, 0).astype(np.float64), starts)
F["rth_dollar"] = np.add.reduceat(np.where(RTH30, v, 0).astype(np.float64) * c, starts)
F["rth_hi"] = np.maximum.reduceat(np.where(RTH30, h, -np.inf).astype(np.float64), starts)
F["rth_lo"] = np.minimum.reduceat(np.where(RTH30, l, np.inf).astype(np.float64), starts)
F["n_rth"] = cnt(RTH30)
for k in ("rth_hi", "rth_lo"):
    F[k][~np.isfinite(F[k])] = np.nan

df = pd.DataFrame(F)
df["ticker"] = tick_names[df["tk"].to_numpy()]
df.to_parquet(S + "td_raw.parquet", index=False)
print("saved td_raw", df.shape, f"{time.time()-t0:.0f}s", flush=True)
print(df.describe().T[["count", "mean", "min", "max"]].round(3).to_string())
