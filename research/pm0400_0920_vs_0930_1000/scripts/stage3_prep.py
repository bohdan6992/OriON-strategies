import numpy as np, pandas as pd, time, sys
sys.stdout.reconfigure(encoding="utf-8")
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
t0 = time.time()
d = pd.read_parquet(S + "td_raw.parquet")
n0 = len(d)
d = d.sort_values(["tk", "date"], kind="mergesort").reset_index(drop=True)
d["dts"] = pd.to_datetime(d["date"].astype(str), format="%Y%m%d")
d["dow"] = d["dts"].dt.weekday

# ---------------- validity
ok = (d.prev_close > 0) & (d.o_0930 > 0) & (d.p_exit > 0) & (d.has_open == 1)
print("rows", n0, "| with open bar & exit & prev_close:", int(ok.sum()))
d = d[ok].copy()
d["fresh"] = (d.t_exit >= 595).astype(np.int8)

# ---------------- pre-market features (all in % of prev_close unless said otherwise)
pc = d.prev_close
d["has_pm"] = (d.n_pm > 0).astype(np.int8)
d["pm_ret"] = (d.p_last / pc - 1) * 100
d["pm_hi_ret"] = (d.pm_hi / pc - 1) * 100
d["pm_lo_ret"] = (d.pm_lo / pc - 1) * 100
d["pm_range"] = (d.pm_hi - d.pm_lo) / pc * 100
rng = (d.pm_hi - d.pm_lo)
d["pos_in_range"] = np.where(rng > 0, (d.p_last - d.pm_lo) / rng.where(rng > 0), 0.5)
d.loc[d.n_pm == 0, "pos_in_range"] = np.nan
d["r_0400_0700"] = (d.p_0700 / pc - 1) * 100
d["r_0700_0830"] = (d.p_0830 / d.p_0700 - 1) * 100
d["r_0830_0920"] = (d.p_last / d.p_0830 - 1) * 100
d["r_0900_0920"] = (d.p_last / d.p_0900 - 1) * 100
d["r_first_last"] = (d.p_last / d.p_first - 1) * 100
d["vwap_pm"] = d.pm_dollar / d.pm_vol.where(d.pm_vol > 0)
d["vwap_dev"] = (d.p_last / d.vwap_pm - 1) * 100
d["pre_open_ret"] = (d.p_0929 / d.p_last - 1) * 100          # 09:21-09:29 drift, known at 09:30
d["open_gap"] = (d.o_0930 / pc - 1) * 100                      # the open vs yesterday close
d["gap_into_open"] = (d.o_0930 / d.p_last - 1) * 100          # open vs last pre-market print
d["lpd"] = np.log10(1 + d.pm_dollar)
d["lpv"] = np.log10(1 + d.pm_vol)
d["bar_imb"] = (d.up_bars - d.dn_bars) / (d.up_bars + d.dn_bars).where((d.up_bars + d.dn_bars) > 0)
d["t_hi_late"] = (d.t_hi >= 500).astype(float).where(d.n_pm > 0)
d["t_lo_late"] = (d.t_lo >= 500).astype(float).where(d.n_pm > 0)
d["pm_ret_z"] = d.pm_ret / d.sigma
d["pm_ret_zd"] = d.pm_ret / d.sigma_day
d["bench_pm"] = d.bench_0920
d["ex_bench_pm"] = d.pm_ret - d.beta_day * d.bench_0920       # stock gap net of its sector ETF gap
d["price"] = pc

# ---------------- own-history relative activity (strictly PAST days, no look-ahead)
g = d.groupby("tk", sort=False)
d["pm_dollar_med20"] = g["pm_dollar"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).median())
d["rvol"] = d.pm_dollar / d.pm_dollar_med20.where(d.pm_dollar_med20 > 0)
d["abs_pm_ret"] = d.pm_ret.abs()
d["abs_pm_med20"] = g["abs_pm_ret"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).median())
d["gap_rel"] = d.abs_pm_ret / d.abs_pm_med20.where(d.abs_pm_med20 > 0)
d["n_hist"] = g.cumcount()

# ---------------- targets (%)
d["R"] = (d.p_exit / d.o_0930 - 1) * 100
d["R_o"] = (d.o_1000 / d.o_0930 - 1) * 100
d["R15"] = (d.p_0945 / d.o_0930 - 1) * 100
d["absR"] = d.R.abs()
d["range30"] = (d.rth_hi - d.rth_lo) / d.o_0930 * 100
d["up_exc"] = (d.rth_hi / d.o_0930 - 1) * 100
d["dn_exc"] = (d.rth_lo / d.o_0930 - 1) * 100
d["R_bench"] = (d.bench_exit - d.bench_pre)
d["R_bn"] = d.R - d.beta_day * d.R_bench

# ---------------- hygiene: crazy rows
bad = (d.pm_ret.abs() > 60) | (d.R.abs() > 60) | (d.open_gap.abs() > 80)
print("dropped as anomalous (|gap|>60% or |R|>60%):", int(bad.sum()))
d = d[~bad].copy()

# ---------------- daily market factor from liquid names
liq = (d.price >= 10) & (d.pm_dollar >= 5e5) & (d.fresh == 1)
mk = d[liq].groupby("date")["R"].agg(mkt_R="mean", n_liq="size")
d = d.merge(mk, on="date", how="left")
d["R_ex"] = d.R - d.mkt_R
liq = (d.price >= 10) & (d.pm_dollar >= 5e5) & (d.fresh == 1)      # recomputed: merge() reset the index
mk2 = d[liq & d.has_pm.astype(bool)].groupby("date")["pm_ret"].agg(mkt_pm="mean")
d = d.merge(mk2, on="date", how="left")
liq = (d.price >= 10) & (d.pm_dollar >= 5e5) & (d.fresh == 1)
d["pm_ex"] = d.pm_ret - d.mkt_pm                              # stock gap net of the market gap

# ---------------- splits
dates = np.sort(d.date.unique())
ndate = len(dates)
cut = dates[int(ndate * 0.6)]
d["period"] = np.where(d.date < cut, "disc", "val")
print("sessions", ndate, "| discovery <", cut, "| validation >=", cut)
d.to_parquet(S + "td.parquet", index=False)
print("saved", d.shape, f"{time.time()-t0:.0f}s")

# ---------------- quick profile
U = d[(d.price >= 5)]
print("\nrows price>=5:", len(U), " with premarket:", int(U.has_pm.sum()), " fresh target:", int(U.fresh.sum()))
print("R pct quantiles (all):", d.R.quantile([.001, .01, .05, .25, .5, .75, .95, .99, .999]).round(2).tolist())
for name, m in (("price>=5", d.price >= 5), ("price>=10 & pm$>=250k & fresh", (d.price >= 10) & (d.pm_dollar >= 2.5e5) & (d.fresh == 1)),
                ("price>=10 & pm$>=2M & fresh", (d.price >= 10) & (d.pm_dollar >= 2e6) & (d.fresh == 1))):
    x = d[m]
    print(f"{name:34s} n={len(x):>9,} mean R={x.R.mean():+.4f} med={x.R.median():+.4f} sd={x.R.std():.3f} up%={(x.R > 0).mean():.3f} |R|mean={x.absR.mean():.3f}")
print("\nmarket factor mkt_R per date: mean", round(mk.mkt_R.mean(), 4), "sd", round(mk.mkt_R.std(), 4), "| share of R variance explained by the date mean (liquid):",
      round(1 - d[liq].R_ex.var() / d[liq].R.var(), 3))
print("liquid names per date (median):", int(mk.n_liq.median()))
