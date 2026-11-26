import numpy as np, pandas as pd, sys, math, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.has_pm == 1) & (d.fresh == 1) & d.R.notna() & d.pm_ex.notna()].copy()
lo, hi = d.R_ex.quantile([0.001, 0.999]); d["Rw"] = d.R_ex.clip(lo, hi)

UNIV = {"A price>=5, pm$>=100k": (d.price >= 5) & (d.pm_dollar >= 1e5),
        "B price>=10, pm$>=500k": (d.price >= 10) & (d.pm_dollar >= 5e5)}
EVENTS = {                                   # name: (mask builder, min discovery events, min validation events)
    "all days":                 (lambda x: np.ones(len(x), bool), 80, 40),
    "|gap|>=1%":                (lambda x: x.pm_ex.abs() >= 1, 30, 12),
    "|gap|>=2%":                (lambda x: x.pm_ex.abs() >= 2, 15, 6),
    "|gap|>=3%":                (lambda x: x.pm_ex.abs() >= 3, 10, 4),
    "|gap|>=2 sigma":           (lambda x: x.pm_ex.abs() / x.sigma >= 2, 15, 6),
    "|gap|>=1% & rvol>=2":      (lambda x: (x.pm_ex.abs() >= 1) & (x.rvol >= 2), 12, 5),
}

def tstat(v):
    n = len(v)
    return v.mean() / (v.std(ddof=1) / math.sqrt(n)) if n > 2 and v.std(ddof=1) > 0 else np.nan

def per_ticker(x, F, nmin_d, nmin_v):
    g = x.assign(F=F).groupby(["tk", "period"]).F
    st = g.agg(["size", "mean", "std"]).unstack("period")
    st.columns = [f"{a}_{b}" for a, b in st.columns]
    st = st.dropna(subset=["size_disc", "size_val"])
    st = st[(st.size_disc >= nmin_d) & (st.size_val >= nmin_v)].copy()
    st["t_disc"] = st.mean_disc / (st.std_disc / np.sqrt(st.size_disc))
    st["t_val"] = st.mean_val / (st.std_val / np.sqrt(st.size_val))
    return st

def cluster_t(sel, F, dates):
    """date-clustered t for the mean of F over the selected events"""
    if len(F) < 5: return np.nan
    m = F.mean(); r = F - m
    dd = pd.Series(r.values).groupby(np.asarray(dates)).sum()
    se = math.sqrt((dd ** 2).sum()) / len(F)
    return m / se if se > 0 else np.nan

def run(x, evmask, nd, nv, tcut=2.0, seed=None):
    """returns persistence corr, and the validation outcome of the 'follow' list and the 'fade' list chosen on discovery"""
    xe = x[evmask].copy()
    if seed is not None:                                  # permutation null: shuffle the move across tickers WITHIN each date
        rs = np.random.RandomState(seed)
        xe["Rw"] = xe.groupby("date").Rw.transform(lambda s: s.values[rs.permutation(len(s))])
    F = np.sign(xe.pm_ex).values * xe.Rw.values                # follow-the-gap return
    st = per_ticker(xe, F, nd, nv)
    out = {"n_tick": len(st), "persist_t": st.t_disc.corr(st.t_val), "persist_mean": st.mean_disc.corr(st.mean_val)}
    for lab, sel_t, sgn in (("follow", st.index[st.t_disc >= tcut], 1), ("fade", st.index[st.t_disc <= -tcut], -1)):
        v = xe[(xe.period == "val") & xe.tk.isin(sel_t)]
        Fv = sgn * np.sign(v.pm_ex).values * v.Rw.values
        out[f"{lab}_n_tick"] = len(sel_t); out[f"{lab}_n_ev"] = len(v)
        out[f"{lab}_val_mean"] = Fv.mean() if len(Fv) else np.nan
        out[f"{lab}_val_hit"] = (Fv > 0).mean() if len(Fv) else np.nan
        out[f"{lab}_val_t"] = cluster_t(sel_t, pd.Series(Fv), v.date.values) if len(Fv) else np.nan
    return out

NPERM = 60
for uname, um in UNIV.items():
    x = d[um]
    print(f"\n################ universe {uname}: {len(x):,} ticker-days, {x.tk.nunique()} tickers")
    rows = []
    for ename, (fn, nd, nv) in EVENTS.items():
        t0 = time.time()
        em = np.asarray(fn(x))
        r = run(x, em, nd, nv)
        # permutation null for the validation mean of the two lists + persistence
        null = pd.DataFrame([run(x, em, nd, nv, seed=s) for s in range(NPERM)])
        r["ename"] = ename
        for lab in ("follow", "fade"):
            r[f"{lab}_p"] = (null[f"{lab}_val_mean"].fillna(-9) >= r[f"{lab}_val_mean"]).mean() if not np.isnan(r[f"{lab}_val_mean"]) else np.nan
            r[f"{lab}_null_mean_tick"] = null[f"{lab}_n_tick"].mean()
        r["persist_t_null_sd"] = null.persist_t.std()
        rows.append(r)
        print(f"  [{ename}] done {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows).set_index("ename")
    print("\n-- persistence of the per-ticker follow-the-gap t-stat, discovery vs validation (null sd from shuffling ~ %.3f)" % R.persist_t_null_sd.mean())
    print(R[["n_tick", "persist_t", "persist_mean"]].round(3).to_string())
    for lab in ("follow", "fade"):
        print(f"\n-- list '{lab}' picked on discovery (t>=2), traded on validation. gain in % per event, in the picked direction")
        print(R[[f"{lab}_n_tick", f"{lab}_null_mean_tick", f"{lab}_n_ev", f"{lab}_val_mean", f"{lab}_val_hit", f"{lab}_val_t", f"{lab}_p"]].round(4).to_string())
