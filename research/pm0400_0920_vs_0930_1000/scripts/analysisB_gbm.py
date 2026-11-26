import numpy as np, pandas as pd, sys, math, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.has_pm == 1) & (d.fresh == 1) & d.R_ex.notna()].copy()
lo, hi = d.R_ex.quantile([0.001, 0.999]); d["R_exw"] = d.R_ex.clip(lo, hi)
FE = ["pm_ret", "pm_ex", "pm_ret_z", "ex_bench_pm", "pm_hi_ret", "pm_lo_ret", "pm_range", "pos_in_range", "r_0400_0700", "r_0700_0830",
      "r_0830_0920", "r_0900_0920", "r_first_last", "vwap_dev", "pre_open_ret", "gap_into_open", "open_gap", "lpd", "lpv", "n_pm", "rvol", "gap_rel",
      "bar_imb", "t_hi_late", "t_lo_late", "max_bar_move", "price", "sigma", "mkt_pm", "t_first", "t_last", "t_hi", "t_lo", "n_pre", "dow"]
UNIV = {"A price>=5, pm$>=100k": (d.price >= 5) & (d.pm_dollar >= 1e5),
        "B price>=10, pm$>=500k": (d.price >= 10) & (d.pm_dollar >= 5e5)}

def perdate_spread(v, score, target, q=0.1):
    """top-decile minus bottom-decile of the model score, per date; returns series"""
    z = pd.DataFrame({"date": v.date.values, "s": score, "y": target})
    z["r"] = z.groupby("date").s.rank(pct=True)
    top = z[z.r > 1 - q].groupby("date").y.mean(); bot = z[z.r <= q].groupby("date").y.mean()
    return (top - bot).dropna(), top, bot

for name, m in UNIV.items():
    x = d[m]
    tr, va = x[x.period == "disc"], x[x.period == "val"]
    # 5-session embargo is irrelevant here: targets are same-day and the cut is by date
    print(f"\n========== {name}: train {len(tr):,} / validation {len(va):,}")
    # ----- direction: classifier on sign(R_ex)
    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=400, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=0)
    clf.fit(tr[FE], (tr.R_ex > 0).astype(int))
    p = clf.predict_proba(va[FE])[:, 1]
    auc = roc_auc_score((va.R_ex > 0).astype(int), p)
    sp, top, bot = perdate_spread(va, p, va.R_exw.values)
    print(f"DIRECTION  classifier: iters={clf.n_iter_}  OOS AUC={auc:.4f}   (0.5 = nothing)   {time.time()-t0:.0f}s")
    print(f"   top-decile mean R_ex {top.mean():+.4f}%   bottom-decile {bot.mean():+.4f}%   long-short {sp.mean():+.4f}% per day-bucket, t={sp.mean()/(sp.std()/math.sqrt(len(sp))):.2f} over {len(sp)} dates")
    # calibration of p: realized up-rate by predicted-p bucket
    b = pd.qcut(pd.Series(p), 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"b": b.values, "p": p, "up": (va.R_ex > 0).values.astype(int)}).groupby("b").agg(p=("p", "mean"), up=("up", "mean"), n=("up", "size"))
    print("   predicted-p decile -> realised up rate:", [f"{a:.3f}->{r:.3f}" for a, r in zip(cal.p, cal.up)])
    # ----- regression on the signed move
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=400, l2_regularization=1.0,
                                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=0, loss="absolute_error")
    reg.fit(tr[FE], tr.R_exw)
    pr = reg.predict(va[FE])
    ic = pd.DataFrame({"date": va.date.values, "a": pd.Series(pr).rank().values, "b": va.R_exw.rank().values})
    icd = pd.DataFrame({"date": va.date.values, "p": pr, "y": va.R_exw.values}).groupby("date").apply(lambda q: q.p.rank().corr(q.y.rank()))
    print(f"SIGNED regression: OOS mean daily rank-IC={icd.mean():+.4f}  t={icd.mean()/(icd.std()/math.sqrt(icd.count())):.2f}")
    sp2, top2, bot2 = perdate_spread(va, pr, va.R_exw.values)
    print(f"   top-decile {top2.mean():+.4f}%  bottom {bot2.mean():+.4f}%  spread {sp2.mean():+.4f}%  t={sp2.mean()/(sp2.std()/math.sqrt(len(sp2))):.2f}")
    # ----- magnitude: regression on |R| (log) and probability of a >=2% move
    mag = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=400, l2_regularization=1.0,
                                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, random_state=0)
    mag.fit(tr[FE], np.log1p(tr.absR))
    pm_ = mag.predict(va[FE])
    icm = pd.DataFrame({"date": va.date.values, "p": pm_, "y": va.absR.values}).groupby("date").apply(lambda q: q.p.rank().corr(q.y.rank()))
    print(f"MAGNITUDE |R|:     OOS mean daily rank-IC={icm.mean():+.4f}  t={icm.mean()/(icm.std()/math.sqrt(icm.count())):.2f}")
    bq = pd.qcut(pd.Series(pm_), 10, labels=False, duplicates="drop")
    cm = pd.DataFrame({"b": bq.values, "absR": va.absR.values, "big": (va.absR >= 2).values.astype(int)}).groupby("b").agg(meanabs=("absR", "mean"), p_ge2=("big", "mean"), n=("big", "size"))
    print("   predicted-magnitude decile -> mean |R| / P(|R|>=2%):", [f"{a:.2f}/{b_:.2f}" for a, b_ in zip(cm.meanabs, cm.p_ge2)])
    # permutation importance (cheap, on a 60k validation sample) for the direction classifier
    from sklearn.inspection import permutation_importance
    sm = va.sample(min(60000, len(va)), random_state=1)
    pi = permutation_importance(clf, sm[FE], (sm.R_ex > 0).astype(int), n_repeats=3, random_state=0, scoring="roc_auc", n_jobs=8)
    imp = pd.Series(pi.importances_mean, index=FE).sort_values(ascending=False).head(8)
    print("   direction: top permutation importances (AUC drop):", {k: round(v, 4) for k, v in imp.items()})
