import numpy as np, pandas as pd, sys, math, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.fresh == 1) & d.R.notna()].copy()
d["Rc"] = d.R.clip(-30, 30); d["absRc"] = d.Rc.abs()
d["big2"] = (d.absRc >= 2).astype(int); d["big3"] = (d.absRc >= 3).astype(int); d["big5"] = (d.absRc >= 5).astype(int)

# ------------------------------------------------------------------ 0. rare vs serial gappers (are fades a leveraged-ETF thing?)
dd = d[(d.has_pm == 1) & d.pm_ex.notna() & (d.price >= 10) & (d.pm_dollar >= 5e5)]
G = dd[dd.pm_ex > 5].copy()
cnt = G.groupby("tk").size(); G["tk_events"] = G.tk.map(cnt)
print("0. gap-up>5% events split by how often THE TICKER does it (rare = news-type, serial = habitual, mostly leveraged ETFs)")
for nm, m in (("rare   (1-2 events in the whole sample)", G.tk_events <= 2), ("occasional (3-9)", (G.tk_events >= 3) & (G.tk_events <= 9)),
              ("frequent (10-29)", (G.tk_events >= 10) & (G.tk_events <= 29)), ("serial (>=30)", G.tk_events >= 30)):
    x = G[m]
    print(f"   {nm:42s} events {len(x):6d} tickers {x.tk.nunique():5d} | open mean {x.Rc.mean():+.3f}% down%={(x.Rc<0).mean():.3f} | disc {x[x.period=='disc'].Rc.mean():+.3f}  val {x[x.period=='val'].Rc.mean():+.3f}")

# ------------------------------------------------------------------ 1. conditions that predict a BIG move (either direction)
x = d[(d.has_pm == 1) & (d.price >= 10) & (d.pm_dollar >= 5e5)].copy()
base = x.big2.mean()
print(f"\n1. base rate in the liquid universe: P(|move 09:30-10:00| >= 2%) = {base:.3f},  >=3%: {x.big3.mean():.3f},  >=5%: {x.big5.mean():.3f}   (n={len(x):,})")
def tab(name, col, edges):
    c = pd.cut(x[col], edges)
    g = x.groupby(c).agg(n=("big2", "size"), p2=("big2", "mean"), p3=("big3", "mean"), p5=("big5", "mean"), absmean=("absRc", "mean"))
    gd = x[x.period == "disc"].groupby(pd.cut(x[x.period == "disc"][col], edges)).big2.mean(); gv = x[x.period == "val"].groupby(pd.cut(x[x.period == "val"][col], edges)).big2.mean()
    g["p2_disc"] = gd; g["p2_val"] = gv
    print(f"\n  by {name}"); print(g.round(3).to_string())
tab("pre-market range % (high-low)/prev_close", "pm_range", [0, 1, 2, 3, 5, 8, 12, 20, 1000])
tab("largest single 1-min move in pre-market %", "max_bar_move", [0, 0.5, 1, 2, 3, 5, 10, 1000])
tab("|gap| net of market %", "pm_ex", [-1000, -10, -5, -3, -1, 1, 3, 5, 10, 1000])
tab("number of pre-market prints", "n_pm", [0, 5, 10, 20, 40, 80, 160, 1000])
tab("relative pre-market volume (rvol)", "rvol", [0, 0.5, 1, 2, 4, 8, 1e9])

# ------------------------------------------------------------------ 2. lists: tickers that move a lot OFTEN; is it persistent?
st = x.groupby(["tk", "period"]).agg(n=("big2", "size"), p2=("big2", "mean"), p3=("big3", "mean"), amean=("absRc", "mean")).unstack("period")
st.columns = [f"{a}_{b}" for a, b in st.columns]; st = st.dropna()
st = st[(st.n_disc >= 40) & (st.n_val >= 20)]
print(f"\n2. persistence of how OFTEN a ticker moves >=2% in 09:30-10:00 (tickers with >=40 disc / >=20 val liquid days): {len(st)}")
print(f"   corr(P(|R|>=2%) disc, val) = {st.p2_disc.corr(st.p2_val):+.3f} | rank corr {st.p2_disc.rank().corr(st.p2_val.rank()):+.3f} | mean |R| {st.amean_disc.corr(st.amean_val):+.3f}")
st["rank_d"] = st.p2_disc.rank(ascending=False)
for k in (25, 50, 100, 200):
    top = st[st.rank_d <= k]
    print(f"   top {k:3d} by disc frequency: disc P(>=2%)={top.p2_disc.mean():.3f} -> val {top.p2_val.mean():.3f} | all tickers val {st.p2_val.mean():.3f} | top-{k} val mean|R| {top.amean_val.mean():.2f}% vs all {st.amean_val.mean():.2f}%")

# ------------------------------------------------------------------ 3. combined: a ticker that is a habitual mover AND shows pre-market activity today
x2 = x.merge(st[["p2_disc"]].rename(columns={"p2_disc": "hist_p2"}), left_on="tk", right_index=True, how="inner")
v = x2[x2.period == "val"]
print("\n3. validation only: tickers' discovery frequency of big moves x today's pre-market range -> P(|R|>=2%) today")
v = v.assign(hist=pd.qcut(v.hist_p2, 4, labels=["rarely", "sometimes", "often", "very often"]), pmr=pd.cut(v.pm_range, [0, 2, 5, 10, 1000], labels=["<2%", "2-5%", "5-10%", ">10%"]))
print(v.pivot_table(index="hist", columns="pmr", values="big2", aggfunc="mean").round(3).to_string())
print(v.pivot_table(index="hist", columns="pmr", values="big2", aggfunc="size").to_string())

# ------------------------------------------------------------------ 4. is the direction of the big mover predictable by the habitual direction?
print("\n4. for habitual movers: does the sign of the big move follow the pre-market direction (gap sign)?  (validation only, |R|>=2%)")
bb = v[(v.big2 == 1) & v.pm_ex.notna()]
for nm, m in (("gap up >=2%", bb.pm_ex >= 2), ("gap down <=-2%", bb.pm_ex <= -2), ("|gap|<1%", bb.pm_ex.abs() < 1)):
    y = bb[m]; print(f"   {nm:16s} n={len(y):6d} share of big moves that were UP: {(y.Rc>0).mean():.3f}")
