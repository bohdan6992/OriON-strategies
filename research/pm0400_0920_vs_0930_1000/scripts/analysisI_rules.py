import numpy as np, pandas as pd, sys, math, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 110)
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.has_pm == 1) & (d.fresh == 1) & d.R.notna() & d.pm_ex.notna()].copy()
d["pnl_short"] = -d.R.clip(-30, 30)                                 # % gained by SHORTING at the 09:30 open and covering ~10:00
d["pnl_short_0935"] = -((d.p_exit / d.p_0935 - 1) * 100).clip(-30, 30)
d["pnl_short_0945"] = -((d.p_0945 / d.o_0930 - 1) * 100).clip(-30, 30)
U = d[(d.price >= 10) & (d.pm_dollar >= 5e5)]

def evaluate(name, m, cost_bps=(0, 10, 20, 30)):
    x = U[m]
    out = {"rule": name, "n": len(x), "events/day": len(x) / x.date.nunique()}
    for per in ("disc", "val"):
        y = x[x.period == per]
        out[f"mean_{per}"] = y.pnl_short.mean(); out[f"win_{per}"] = (y.pnl_short > 0).mean(); out[f"n_{per}"] = len(y)
    # daily equal-weight series, validation
    v = x[x.period == "val"]
    dly = v.groupby("date").pnl_short.mean()
    out["val_days"] = len(dly); out["val_daily_t"] = dly.mean() / (dly.std() / math.sqrt(len(dly)))
    out["val_p01"] = v.pnl_short.quantile(0.01); out["val_p99"] = v.pnl_short.quantile(0.99)
    out["val_0935entry"] = v.pnl_short_0935.mean()
    out["val_0945exit"] = v.pnl_short_0945.mean()
    for c in cost_bps:
        out[f"val_net_{c}bp"] = v.pnl_short.mean() - 2 * c / 100          # c bp each way (entry + exit)
    return out

rules = [
    ("gap(net mkt) > 3%",                              U.pm_ex > 3),
    ("gap > 5%",                                       U.pm_ex > 5),
    ("gap > 8%",                                       U.pm_ex > 8),
    ("gap > 10%",                                      U.pm_ex > 10),
    ("gap > 15%",                                      U.pm_ex > 15),
    ("gap > 5% & gap > 5 sigma",                       (U.pm_ex > 5) & (U.pm_ex / U.sigma > 5)),
    ("gap > 5% & pre-mkt close MID range",             (U.pm_ex > 5) & (U.pos_in_range > 0.25) & (U.pos_in_range < 0.75)),
    ("gap > 5% & rvol >= 3",                           (U.pm_ex > 5) & (U.rvol >= 3)),
    ("gap > 5% & n_prints >= 40",                      (U.pm_ex > 5) & (U.n_pm >= 40)),
    ("gap > 5% & ticker gaps rarely (<=9 events total)", (U.pm_ex > 5) & (U.tk.map(U[U.pm_ex > 5].groupby("tk").size()).fillna(0) <= 9)),
    ("RAW gap (not net of market) > 5%",               U.pm_ret > 5),
]
R = pd.DataFrame([evaluate(n, m) for n, m in rules]).set_index("rule")
print("SHORT the gap-up at 09:30, cover ~10:00. All numbers in % per trade; universe price>=10 & pre-mkt $vol>=500k")
print(R[["n", "events/day", "mean_disc", "win_disc", "mean_val", "win_val", "val_daily_t", "val_p01", "val_p99"]].round(3).to_string())
print("\nsensitivity (validation): entry after the first minutes, shorter holding, and round-trip costs of c bp EACH way")
print(R[["val_0935entry", "val_0945exit", "val_net_0bp", "val_net_10bp", "val_net_20bp", "val_net_30bp"]].round(3).to_string())

# horizon profile of the fade for gap>5%: where in 09:30-10:00 does it happen?
g = U[U.pm_ex > 5]
prof = {}
for lab, col in (("09:35", "p_0935"), ("09:40", "p_0940"), ("09:45", "p_0945"), ("09:50", "p_0950"), ("10:00", "p_exit")):
    prof[lab] = (-(g[col] / g.o_0930 - 1) * 100).clip(-30, 30).mean()
print("\nmean short PnL measured from the 09:30 open to:", {k: round(v, 3) for k, v in prof.items()})

# the other replicating cells from the 3-dim scan: how big are they really?
c = pd.read_parquet(S + "cells_B.parquet")
big = c[(c.mv_or >= 0.15) & (c.nv >= 1000)].sort_values("mv_or", ascending=False)
print(f"\nscan cells in B that replicated: {len(c)}; with validation effect >= 0.15% and >=1000 events: {len(big)}; "
      f"median validation effect of all replicating cells {c.mv_or.median():.3f}%, 90th pct {c.mv_or.quantile(.9):.3f}%")
