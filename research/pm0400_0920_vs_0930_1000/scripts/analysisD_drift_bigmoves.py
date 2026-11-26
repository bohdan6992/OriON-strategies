import numpy as np, pandas as pd, sys, math, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.fresh == 1) & d.R.notna() & (d.price >= 10)].copy()          # all liquid-ish days, with OR without a pre-market print
lo, hi = d.R_ex.quantile([0.001, 0.999]); d["Rw"] = d.R_ex.clip(lo, hi)
d["up"] = (d.R_ex > 0).astype(int)

# ------------------------------------------------------------------ D0: does a ticker have a persistent own drift (always up / always down 09:30-10:00)?
x = d[d.pm_dollar >= 2.5e5]
st = x.groupby(["tk", "period"]).agg(n=("Rw", "size"), mean=("Rw", "mean"), sd=("Rw", "std"), up=("up", "mean")).unstack("period")
st.columns = [f"{a}_{b}" for a, b in st.columns]
st = st.dropna()
st = st[(st.n_disc >= 60) & (st.n_val >= 30)]
st["t_disc"] = st.mean_disc / (st.sd_disc / np.sqrt(st.n_disc)); st["t_val"] = st.mean_val / (st.sd_val / np.sqrt(st.n_val))
print(f"D0 own drift: {len(st)} tickers (>=60 disc / >=30 val days, pm$>=250k, price>=10)")
print(f"   corr(mean R_ex disc, val)={st.mean_disc.corr(st.mean_val):+.3f}   corr(P(up) disc, val)={st.up_disc.corr(st.up_val):+.3f}   corr(t disc,val)={st.t_disc.corr(st.t_val):+.3f}")
# how many tickers look 'always up' / 'always down' in discovery, and do they stay that way?
for lab, sel in (("UP list   (disc t>=2.5)", st.t_disc >= 2.5), ("DOWN list (disc t<=-2.5)", st.t_disc <= -2.5)):
    s = st[sel]
    sgn = 1 if "UP" in lab else -1
    print(f"   {lab}: {len(s):3d} tickers | disc mean {s.mean_disc.mean()*sgn:+.3f}% -> val mean {s.mean_val.mean()*sgn:+.3f}% (in the list's direction) | val P(up) {s.up_val.mean():.3f} | share of tickers still same sign in val: {(np.sign(s.mean_val)==sgn).mean():.2f}")
print("   expected number of tickers with |t|>=2.5 by chance (two-sided p=.0124): %.0f of %d" % (0.0124 * len(st), len(st)))
print("   observed with t>=2.5: %d, t<=-2.5: %d" % ((st.t_disc >= 2.5).sum(), (st.t_disc <= -2.5).sum()))

# ------------------------------------------------------------------ D1: among BIG movers, is the sign predictable from the pre-market tape?
x = d[(d.has_pm == 1) & (d.pm_dollar >= 5e5)].copy()
big = x[x.R_ex.abs() >= 2].copy()
big["bigup"] = (big.R_ex > 0).astype(int)
print(f"\nD1 big movers (|R_ex|>=2%): {len(big):,} of {len(x):,} liquid days ({len(big)/len(x):.1%}); share of those that were UP: {big.bigup.mean():.3f}")
FE = ["pm_ret", "pm_hi_ret", "pm_lo_ret", "pm_range", "pos_in_range", "r_0400_0700", "r_0700_0830", "r_0830_0920", "r_0900_0920", "vwap_dev",
      "lpd", "n_pm", "rvol", "gap_rel", "bar_imb", "t_hi_late", "t_lo_late", "max_bar_move", "price", "sigma", "ex_bench_pm", "t_hi", "t_lo", "pm_ret_z", "pm_ex"]
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
tr, va = big[big.period == "disc"], big[big.period == "val"]
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=2.0,
                                     early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=0)
clf.fit(tr[FE], tr.bigup)
p = clf.predict_proba(va[FE])[:, 1]
print(f"   GBM direction among big movers: OOS AUC={roc_auc_score(va.bigup, p):.4f}  (train {len(tr):,}, val {len(va):,})")
dec = pd.qcut(pd.Series(p), 10, labels=False, duplicates="drop")
print("   realised UP share by predicted decile:", [round(float(v), 3) for v in va.bigup.groupby(dec.values).mean().values])
# the single most 'obvious' rules, stated plainly
def rule(name, m):
    s = big[m]; sd, sv = s[s.period == "disc"], s[s.period == "val"]
    print(f"   {name:46s} disc n={len(sd):6d} up={sd.bigup.mean():.3f} | val n={len(sv):6d} up={sv.bigup.mean():.3f}")
rule("gap up >=2% (pm_ex)", big.pm_ex >= 2)
rule("gap down <=-2%", big.pm_ex <= -2)
rule("closes pre-market in top 10% of its range", big.pos_in_range >= 0.9)
rule("closes pre-market in bottom 10% of its range", big.pos_in_range <= 0.1)
rule("last 50 min (08:30-09:20) rising >1%", big.r_0830_0920 >= 1)
rule("last 50 min falling <-1%", big.r_0830_0920 <= -1)
rule("above pre-market VWAP by >1%", big.vwap_dev >= 1)
rule("below pre-market VWAP by >1%", big.vwap_dev <= -1)
rule("pre-market HIGH set in last 40 min & gap up", (big.t_hi >= 520) & (big.pm_ex >= 1))
rule("pre-market LOW set in last 40 min & gap down", (big.t_lo >= 520) & (big.pm_ex <= -1))
