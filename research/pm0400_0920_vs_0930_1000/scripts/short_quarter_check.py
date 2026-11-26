import numpy as np, pandas as pd, sys, math, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.has_pm == 1) & (d.fresh == 1) & d.R.notna() & d.pm_ex.notna() & d.p_0935.notna()].copy()
U = d[(d.price >= 10) & (d.pm_dollar >= 5e5)].copy()
U["s_open"] = -U.R.clip(-30, 30)
U["s_0935"] = -((U.p_exit / U.p_0935 - 1) * 100).clip(-30, 30)
U["q"] = pd.to_datetime(U.date.astype(str), format="%Y%m%d").dt.to_period("Q").astype(str)
U["dn"] = U.dow.map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"})

def daily_t(x, col):
    g = x.groupby("date")[col].mean()
    return g.mean(), g.mean() / (g.std() / math.sqrt(len(g))), len(g)

print("BASELINE short_0935 all liquid events: mean %.3f | short_open %.3f" % (U.s_0935.mean(), U.s_open.mean()))
print("\n-- by day of week (market-wide drift?), daily-clustered t")
for k, x in U.groupby("dn"):
    m, t, n = daily_t(x, "s_0935"); m2, t2, _ = daily_t(x, "s_open")
    print(f"{k}: days {n:3d} short_0935 {m:+.3f} (t {t:+.1f}) | short_open {m2:+.3f} (t {t2:+.1f})")
print("\n-- Thursday short_0935 by quarter")
for q, x in U[U.dn == "Thu"].groupby("q"):
    m, t, n = daily_t(x, "s_0935"); print(f"  {q} days {n:2d} mean {m:+.3f} t {t:+.1f}")
print("\n-- ALL days short_0935 by quarter")
for q, x in U.groupby("q"):
    m, t, n = daily_t(x, "s_0935"); print(f"  {q} days {n:2d} mean {m:+.3f} t {t:+.1f}")

# demeaned by same-day average: is there any STOCK-level short edge beyond the day effect?
U["s_dm"] = U.s_0935 - U.groupby("date").s_0935.transform("mean")
print("\n-- stock-level (same-day demeaned) short_0935 for gap>5 net:")
for nm, m in (("gap>3", U.pm_ex > 3), ("gap>5", U.pm_ex > 5), ("gap>10", U.pm_ex > 10), ("gap>5 mid-range", (U.pm_ex > 5) & U.pos_in_range.between(.25, .75))):
    x = U[m]; v = x[x.period == "val"]; dd = x[x.period == "disc"]
    mv, tv, nv = daily_t(v, "s_dm")
    print(f"  {nm:16s} n={len(x):5d} disc {dd.s_dm.mean():+.3f} val {mv:+.3f} (daily t {tv:+.1f}) | raw val short_0935 {v.s_0935.mean():+.3f}")
print("\n-- gap>5 raw short_0935 by quarter")
x = U[U.pm_ex > 5]
for q, y in x.groupby("q"):
    print(f"  {q} n={len(y):4d} raw {y.s_0935.mean():+.3f} demeaned {y.s_dm.mean():+.3f}")
