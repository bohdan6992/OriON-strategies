import numpy as np, pandas as pd, sys, math
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.has_pm == 1) & (d.fresh == 1) & d.R_ex.notna()].copy()
# winsorise the target a little so a handful of 40% prints do not dominate means (kept raw for the sign / hit-rate tests)
lo, hi = d.R_ex.quantile([0.001, 0.999]); d["R_exw"] = d.R_ex.clip(lo, hi)
UNIV = {
    "A  price>=5, pm$>=100k": (d.price >= 5) & (d.pm_dollar >= 1e5),
    "B  price>=10, pm$>=500k": (d.price >= 10) & (d.pm_dollar >= 5e5),
    "C  price>=10, pm$>=2M": (d.price >= 10) & (d.pm_dollar >= 2e6),
}
FEATS = ["pm_ret", "pm_ex", "pm_ret_z", "ex_bench_pm", "pm_hi_ret", "pm_lo_ret", "pm_range", "pos_in_range", "r_0400_0700", "r_0700_0830",
         "r_0830_0920", "r_0900_0920", "r_first_last", "vwap_dev", "pre_open_ret", "gap_into_open", "open_gap", "lpd", "n_pm", "rvol", "gap_rel",
         "bar_imb", "t_hi_late", "t_lo_late", "max_bar_move", "price"]

def ic_table(x, target):
    """per-date Spearman(feature, target); mean IC and t over dates, discovery vs validation"""
    rows = []
    for f in FEATS:
        z = x[["date", "period", f, target]].dropna()
        if len(z) < 1000: continue
        z = z.assign(rf=z.groupby("date")[f].rank(), rt=z.groupby("date")[target].rank())
        g = z.groupby("date")
        n = g.size(); ok = n[n >= 30].index
        z = z[z.date.isin(ok)]
        ic = z.groupby("date").apply(lambda q: np.corrcoef(q.rf, q.rt)[0, 1])
        per = z.groupby("date").period.first()
        r = {"feature": f, "n": len(z)}
        for p in ("disc", "val"):
            v = ic[per == p].dropna()
            r[f"IC_{p}"] = v.mean(); r[f"t_{p}"] = v.mean() / (v.std() / math.sqrt(len(v))) if len(v) > 2 else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("feature")

for name, m in UNIV.items():
    x = d[m]
    print(f"\n================ universe {name}: n={len(x):,}, dates={x.date.nunique()}, tickers={x.tk.nunique()}")
    print(f"mean R_ex={x.R_ex.mean():+.4f}  up%={(x.R_ex>0).mean():.3f}  sd={x.R_ex.std():.3f}")
    t1 = ic_table(x, "R_exw")
    print("--- rank-IC vs SIGNED move (R_ex).  IC>0 = continuation, IC<0 = reversal")
    print(t1.round(4).to_string())
    t2 = ic_table(x.assign(absR=x.absR), "absR")
    print("--- rank-IC vs MAGNITUDE |R| (does the feature predict that it moves, any direction)")
    print(t2.round(4).to_string())
