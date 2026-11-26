import numpy as np, pandas as pd, sys, math, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.has_pm == 1) & (d.fresh == 1) & d.R.notna() & d.pm_ex.notna()].copy()
# three entry assumptions: the 09:30 open print, the close of the first minute, and the last print before 09:35
d["R_open"] = d.R
d["R_1min"] = (d.p_exit / d.c_0930 - 1) * 100
d["R_0935"] = (d.p_exit / d.p_0935 - 1) * 100
d["open_range"] = (d.h_0930 - d.l_0930) / d.o_0930 * 100
d["open_vs_pm"] = d.gap_into_open
d["half"] = np.where(d.date < d.date.quantile(0.5, interpolation="nearest"), "H1", "H2")
qs = d.date.drop_duplicates().sort_values().values
d["q"] = pd.cut(d.date, [qs[0] - 1] + [qs[int(len(qs) * f)] for f in (0.25, 0.5, 0.75)] + [qs[-1]], labels=["Q1", "Q2", "Q3", "Q4"])

def cl_t(v, dates):
    """date-clustered t of the mean"""
    v = np.asarray(v, float); m = v.mean(); r = v - m
    s = pd.Series(r).groupby(np.asarray(dates)).sum()
    se = math.sqrt((s ** 2).sum()) / len(v)
    return m / se if se > 0 else np.nan

def line(lbl, x, col):
    v = x[col].clip(-30, 30)
    return f"{lbl:26s} n={len(x):6d} mean={v.mean():+.3f}% med={x[col].median():+.3f}% down%={(x[col]<0).mean():.3f} t={cl_t(v, x.date):+.2f}"

U = d[(d.price >= 10) & (d.pm_dollar >= 5e5)]
print(f"universe B (price>=10, pm$>=500k): {len(U):,} events, {U.date.nunique()} dates")
print("\n=== 1. size of the pre-market gap (net of market gap) -> move 09:30-10:00, entered AT THE OPEN PRINT.  negative = fades")
bins = [-100, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 100]
U = U.assign(gb=pd.cut(U.pm_ex, bins))
for gb, x in U.groupby("gb"):
    print(line(f"gap {gb}", x, "R_open"), "|", "disc" , f"{x[x.period=='disc'].R_open.clip(-30,30).mean():+.3f}", "val", f"{x[x.period=='val'].R_open.clip(-30,30).mean():+.3f}")

print("\n=== 2. the gap-up >5% group: entry-price sensitivity (is it just the opening print?)")
G = U[U.pm_ex > 5]
for col, nm in (("R_open", "enter at 09:30 open print"), ("R_1min", "enter at close of 09:30 bar (~09:31)"), ("R_0935", "enter at last print <= 09:35")):
    print(line(nm, G, col))
print("   open print vs last pre-market print (the opening jump / dip):  mean", round(G.open_vs_pm.mean(), 3), "%   09:30 bar range mean", round(G.open_range.mean(), 3), "%")

print("\n=== 3. stability: by calendar quarter-of-sample and by period (gap>5%, open entry)")
for k, x in G.groupby("q"):
    print(line(f"quarter {k}", x, "R_open"))
for k, x in G.groupby("period"):
    print(line(f"period {k}", x, "R_open"))

print("\n=== 4. which gap-ups fade?  gap>5% split by structure (entry at open; also the 09:35 entry)")
def sp(name, m):
    x = G[m];
    print(f"{name:44s} n={len(x):5d} | open {x.R_open.clip(-30,30).mean():+.3f}% t={cl_t(x.R_open.clip(-30,30), x.date):+.2f} | 09:35 {x.R_0935.clip(-30,30).mean():+.3f}% t={cl_t(x.R_0935.clip(-30,30), x.date):+.2f} | down%={(x.R_open<0).mean():.3f}")
sp("pre-market close in TOP 25% of range", G.pos_in_range >= 0.75)
sp("pre-market close MID range", (G.pos_in_range > 0.25) & (G.pos_in_range < 0.75))
sp("pre-market close LOW 25% of range", G.pos_in_range <= 0.25)
sp("high set early (before 08:20)", G.t_hi < 500)
sp("high set LATE (after 08:20)", G.t_hi >= 500)
sp("rvol >= 3 (unusual volume)", G.rvol >= 3)
sp("rvol < 1", G.rvol < 1)
sp("price 10-30", (G.price >= 10) & (G.price < 30))
sp("price 30-100", (G.price >= 30) & (G.price < 100))
sp("price >= 100", G.price >= 100)
sp("pm$ < 2M", G.pm_dollar < 2e6)
sp("pm$ >= 10M", G.pm_dollar >= 1e7)
sp("n_prints >= 40", G.n_pm >= 40)
sp("n_prints < 10", G.n_pm < 10)
sp("gap 5-8%", (G.pm_ex > 5) & (G.pm_ex <= 8))
sp("gap 8-15%", (G.pm_ex > 8) & (G.pm_ex <= 15))
sp("gap > 15%", G.pm_ex > 15)
sp("gap > 2 sigma", G.pm_ex / G.sigma > 2)
sp("report/news day proxy: gap>5 & gap>5sigma", G.pm_ex / G.sigma > 5)

print("\n=== 5. mirror: gap DOWN <-5% (does it bounce?)")
Gd = U[U.pm_ex < -5]
for col, nm in (("R_open", "enter at open print"), ("R_1min", "enter close 09:30 bar"), ("R_0935", "enter <=09:35")):
    print(line(nm, Gd, col))
for k, x in Gd.groupby("period"): print(line(f"period {k}", x, "R_open"))

print("\n=== 6. same question in the wider universe A (price>=5, pm$>=100k) and in the very liquid one C (pm$>=5M)")
for nm, m in (("A", (d.price >= 5) & (d.pm_dollar >= 1e5)), ("C pm$>=5M", (d.price >= 10) & (d.pm_dollar >= 5e6))):
    x = d[m & (d.pm_ex > 5)]
    print(f"universe {nm}: gap>5%")
    for col, lbl in (("R_open", " open"), ("R_0935", " 09:35")):
        print("  ", line(lbl, x, col))
    for k, y in x.groupby("period"): print("     ", line(f"period {k}", y, "R_open"))
