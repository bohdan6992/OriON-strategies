import numpy as np, pandas as pd, sys, math, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.has_pm == 1) & (d.fresh == 1) & d.R.notna()].copy()

# ---- (1) is there a market-level pattern? (686 dates): market gap at 09:20 vs market move 09:30-10:00
dm = d[(d.price >= 10) & (d.pm_dollar >= 5e5)].groupby("date").agg(mkt_pm=("pm_ret", "mean"), mkt_R=("R", "mean"), per=("period", "first"))
for p in ("disc", "val", None):
    z = dm if p is None else dm[dm.per == p]
    print(f"MARKET level ({p or 'all'}): n={len(z)} corr(market gap 09:20, market move 09:30-10:00)={z.mkt_pm.corr(z.mkt_R):+.3f}  "
          f"corr(sign)={np.corrcoef(np.sign(z.mkt_pm), np.sign(z.mkt_R))[0,1]:+.3f}  P(same sign)={(np.sign(z.mkt_pm)==np.sign(z.mkt_R)).mean():.3f}")

FE = ["pm_ret", "pm_hi_ret", "pm_lo_ret", "pm_range", "pos_in_range", "r_0400_0700", "r_0700_0830", "r_0830_0920", "r_0900_0920", "r_first_last",
      "vwap_dev", "pre_open_ret", "gap_into_open", "open_gap", "lpd", "lpv", "n_pm", "rvol", "gap_rel", "bar_imb", "t_hi_late", "t_lo_late",
      "max_bar_move", "price", "sigma", "ex_bench_pm", "t_first", "t_last", "t_hi", "t_lo", "n_pre", "pm_ret_z"]
UNIV = {"A price>=5, pm$>=100k": (d.price >= 5) & (d.pm_dollar >= 1e5),
        "B price>=10, pm$>=500k": (d.price >= 10) & (d.pm_dollar >= 5e5)}

def pct_rank_by_date(x, cols):
    g = x.groupby("date")
    out = x[["date"]].copy()
    for c in cols:
        out[c] = g[c].rank(pct=True)
    return out.drop(columns="date")

def perdate_spread(v, score, target, q=0.1):
    z = pd.DataFrame({"date": v.date.values, "s": score, "y": target})
    z["r"] = z.groupby("date").s.rank(pct=True)
    top = z[z.r > 1 - q].groupby("date").y.mean(); bot = z[z.r <= q].groupby("date").y.mean()
    return (top - bot).dropna(), top, bot

for name, m in UNIV.items():
    x = d[m].copy()
    # within-universe, within-date relative move: nothing day-level can predict it
    x["R_rel"] = x.R - x.groupby("date").R.transform("mean")
    lo, hi = x.R_rel.quantile([0.001, 0.999]); x["R_relw"] = x.R_rel.clip(lo, hi)
    X = pct_rank_by_date(x, FE)
    tr = (x.period == "disc").values; va = ~tr
    print(f"\n========== {name}: train {tr.sum():,} / validation {va.sum():,}   (features = within-day percentiles, target = move minus same-universe same-day mean)")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=400, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=0)
    clf.fit(X[tr], (x.R_rel[tr] > 0).astype(int))
    p = clf.predict_proba(X[va])[:, 1]
    auc = roc_auc_score((x.R_rel[va] > 0).astype(int), p)
    sp, top, bot = perdate_spread(x[va], p, x.R_relw[va].values)
    print(f"DIRECTION classifier: iters={clf.n_iter_}  OOS AUC={auc:.4f}")
    print(f"   top-decile {top.mean():+.4f}%  bottom-decile {bot.mean():+.4f}%  spread {sp.mean():+.4f}%  t={sp.mean()/(sp.std()/math.sqrt(len(sp))):.2f} over {len(sp)} dates")
    b = pd.qcut(pd.Series(p), 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"b": b.values, "p": p, "up": (x.R_rel[va] > 0).values.astype(int)}).groupby("b").agg(p=("p", "mean"), up=("up", "mean"))
    print("   predicted-p decile -> realised up rate:", [f"{a:.3f}->{r:.3f}" for a, r in zip(cal.p, cal.up)])
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=400, l2_regularization=1.0,
                                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=0, loss="absolute_error")
    reg.fit(X[tr], x.R_relw[tr]); pr = reg.predict(X[va])
    icd = pd.DataFrame({"date": x.date[va].values, "p": pr, "y": x.R_relw[va].values}).groupby("date").apply(lambda q: q.p.rank().corr(q.y.rank()))
    print(f"SIGNED regression: OOS mean daily rank-IC={icd.mean():+.4f}  t={icd.mean()/(icd.std()/math.sqrt(icd.count())):.2f}")
    sp2, top2, bot2 = perdate_spread(x[va], pr, x.R_relw[va].values)
    print(f"   top-decile {top2.mean():+.4f}%  bottom {bot2.mean():+.4f}%  spread {sp2.mean():+.4f}%  t={sp2.mean()/(sp2.std()/math.sqrt(len(sp2))):.2f}")
    # by halves of validation, so a single regime cannot carry the result
    vd = np.sort(x.date[va].unique()); half = vd[len(vd)//2]
    for lab, sel in (("val first half", x.date[va].values < half), ("val second half", x.date[va].values >= half)):
        s_, _, _ = perdate_spread(x[va][sel], pr[sel], x.R_relw[va].values[sel])
        print(f"   spread {lab}: {s_.mean():+.4f}%  t={s_.mean()/(s_.std()/math.sqrt(len(s_))):.2f}")
    # importances
    from sklearn.inspection import permutation_importance
    idx = np.random.RandomState(1).choice(np.flatnonzero(va), size=min(60000, va.sum()), replace=False)
    pi = permutation_importance(clf, X.iloc[idx], (x.R_rel.iloc[idx] > 0).astype(int), n_repeats=3, random_state=0, scoring="roc_auc", n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=FE).sort_values(ascending=False).head(8)
    print("   top permutation importances (AUC drop):", {k: round(v, 4) for k, v in imp.items()})
