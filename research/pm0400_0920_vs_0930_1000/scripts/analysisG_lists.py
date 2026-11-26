import numpy as np, pandas as pd, sys, math, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.has_pm == 1) & (d.fresh == 1) & d.R.notna() & d.pm_ex.notna()].copy()
d["Rc"] = d.R.clip(-30, 30)
d["R_0935"] = ((d.p_exit / d.p_0935 - 1) * 100).clip(-30, 30)
U = d[(d.price >= 10) & (d.pm_dollar >= 5e5)]
G = U[U.pm_ex > 5].copy()
print(f"gap-up >5% events: {len(G):,}, tickers: {G.tk.nunique()}, per ticker median {G.groupby('tk').size().median():.0f}, max {G.groupby('tk').size().max()}")

# ----------------------------------------------------------------------------------------------------------------- 1. heterogeneity
print("\n=== 1. Is the fade ticker-specific?  Cochran Q: do per-ticker means differ more than chance allows?")
for per in ("disc", "val", "all"):
    x = G if per == "all" else G[G.period == per]
    st = x.groupby("tk").Rc.agg(["size", "mean", "var"]).dropna()
    st = st[st["size"] >= 5]
    s2 = x.Rc.var()
    se2 = s2 / st["size"]
    M = np.average(st["mean"], weights=1 / se2)
    Q = (((st["mean"] - M) ** 2) / se2).sum(); df = len(st) - 1
    tau2 = max(0.0, (Q - df) / (np.sum(1 / se2) - np.sum((1 / se2) ** 2) / np.sum(1 / se2)))
    print(f"  {per:5s}: tickers(>=5 events) {len(st):4d} | pooled mean {M:+.3f}% | Q={Q:.0f} vs df={df} (ratio {Q/df:.2f}; 1.00 = fully homogeneous) | between-ticker sd tau={math.sqrt(tau2):.3f}% (within-event sd {math.sqrt(s2):.2f}%)")

# ----------------------------------------------------------------------------------------------------------------- 2. persistence
print("\n=== 2. persistence of the ticker's own fade, discovery -> validation (tickers with >=4 events in each)")
a = G[G.period == "disc"].groupby("tk").Rc.agg(["size", "mean"]); b = G[G.period == "val"].groupby("tk").Rc.agg(["size", "mean"])
j = a.join(b, lsuffix="_d", rsuffix="_v", how="inner"); j = j[(j.size_d >= 4) & (j.size_v >= 4)]
print(f"  tickers {len(j)} | corr(mean disc, mean val) = {j.mean_d.corr(j.mean_v):+.3f}   rank corr {j.mean_d.rank().corr(j.mean_v.rank()):+.3f}")
rs = np.random.RandomState(0); nulls = []
for _ in range(500):
    p = rs.permutation(len(j)); nulls.append(j.mean_d.corr(pd.Series(j.mean_v.values[p], index=j.index)))
print(f"  permutation null sd {np.std(nulls):.3f}")

# ----------------------------------------------------------------------------------------------------------------- 3. tradable lists built on discovery
print("\n=== 3. lists picked on DISCOVERY only, traded on VALIDATION (gap-up>5% events; short at open, cover ~10:00)")
base_v = G[G.period == "val"]
print(f"  baseline: every gap-up>5% in validation: n={len(base_v):,} mean {base_v.Rc.mean():+.3f}%  (a short earns {-base_v.Rc.mean():+.3f}%)")
d_ev = G[G.period == "disc"].groupby("tk").Rc.agg(["size", "mean", "std"])
d_ev["t"] = d_ev["mean"] / (d_ev["std"] / np.sqrt(d_ev["size"]))
M = G[G.period == "disc"].Rc.mean(); s2 = G[G.period == "disc"].Rc.var()
for k in (4, 6, 8):
    c = d_ev[d_ev["size"] >= k].copy()
    # shrink toward the pooled mean with the between-ticker variance found above (small) -> empirical-Bayes score
    tau2 = 0.05
    w = tau2 / (tau2 + s2 / c["size"]); c["eb"] = M + w * (c["mean"] - M)
    for nm, sel in (("most fading (lowest EB mean)", c.eb.nsmallest(30).index), ("least fading / continuing", c.eb.nlargest(30).index), ("random 30", c.sample(30, random_state=1).index)):
        v = base_v[base_v.tk.isin(sel)]
        print(f"  tickers with >= {k} disc events ({len(c)}), {nm:30s}: val events {len(v):4d} mean {v.Rc.mean():+.3f}%  down%={(v.Rc<0).mean():.3f}")

# ----------------------------------------------------------------------------------------------------------------- 4. serial gappers: who does it often
print("\n=== 4. serial gappers (gap-up >5% net of market) - the tickers that produce the event most often, with their fade record")
allg = G.groupby("tk").agg(n=("Rc", "size"), mean_open=("Rc", "mean"), mean_0935=("R_0935", "mean"), down=("Rc", lambda s: (s < 0).mean()), price=("price", "median"), pm_usd=("pm_dollar", "median"))
vv = G[G.period == "val"].groupby("tk").Rc.agg(n_val="size", mean_val="mean")
dd = G[G.period == "disc"].groupby("tk").Rc.agg(n_disc="size", mean_disc="mean")
tab = allg.join(dd).join(vv)
tab["name"] = d.groupby("tk").ticker.first()
top = tab.sort_values("n", ascending=False).head(40)
print(top[["name", "n", "mean_open", "mean_0935", "down", "n_disc", "mean_disc", "n_val", "mean_val", "price", "pm_usd"]].round(3).to_string())
tab.reset_index().to_parquet(S + "serial_gappers.parquet", index=False)
