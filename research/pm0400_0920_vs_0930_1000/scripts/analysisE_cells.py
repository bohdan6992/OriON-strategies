import numpy as np, pandas as pd, sys, math, itertools, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.has_pm == 1) & (d.fresh == 1) & d.R.notna() & d.pm_ex.notna()].copy()

def cut(s, edges, labels):
    c = pd.cut(s, [-np.inf] + edges + [np.inf], labels=False).astype("float")
    c = c.fillna(len(labels)).astype(int)                      # NaN -> its own bin
    return c.values, labels + ["na"]

def build(x):
    D = {}
    D["gap"] = cut(x.pm_ex, [-5, -2, -0.5, 0.5, 2, 5], ["<=-5", "-5..-2", "-2..-.5", "-.5..+.5", "+.5..2", "2..5", ">5"])
    D["pos_in_range"] = cut(x.pos_in_range, [0.25, 0.75], ["low25", "mid", "top25"])
    D["last50min"] = cut(x.r_0830_0920, [-1, 1], ["fall>1%", "flat", "rise>1%"])
    D["rvol"] = cut(x.rvol, [1, 3], ["<1x", "1-3x", ">=3x"])
    D["hi/lo_late"] = ((((x.t_hi >= 500).astype(int) * 2 + (x.t_lo >= 500).astype(int)).values), ["hi_early,lo_early", "hi_early,LO_LATE", "HI_LATE,lo_early", "HI&LO_late"])
    D["vs_vwap"] = cut(x.vwap_dev, [-1, 1], ["<-1%", "~vwap", ">+1%"])
    D["n_prints"] = cut(x.n_pm, [10, 40], ["<10", "10-40", ">=40"])
    D["price"] = cut(x.price, [30, 100], ["10-30", "30-100", ">100"])
    D["pm_$vol"] = cut(x.pm_dollar, [2e6, 1e7], ["<2M", "2-10M", ">10M"])
    D["early_0400_0700"] = cut(x.r_0400_0700, [-1, 1], ["fall>1%", "flat", "rise>1%"])
    return D

def cells(D, y, per_disc, minD, minV, order=3):
    """vectorised over all 1..order-dim combinations; returns dataframe of cells with disc/val stats"""
    names = list(D); out = []
    for k in range(1, order + 1):
        for combo in itertools.combinations(names, k):
            codes = np.zeros(len(y), np.int64); labs = []; stride = 1
            for nme in combo:
                c, l = D[nme]; codes += c * stride; stride *= len(l); labs.append(l)
            size = stride
            nd = np.bincount(codes[per_disc], minlength=size); nv = np.bincount(codes[~per_disc], minlength=size)
            sd = np.bincount(codes[per_disc], weights=y[per_disc], minlength=size); sv = np.bincount(codes[~per_disc], weights=y[~per_disc], minlength=size)
            qd = np.bincount(codes[per_disc], weights=y[per_disc] ** 2, minlength=size); qv = np.bincount(codes[~per_disc], weights=y[~per_disc] ** 2, minlength=size)
            ok = np.flatnonzero((nd >= minD) & (nv >= minV))
            if len(ok) == 0: continue
            md = sd[ok] / nd[ok]; mv = sv[ok] / nv[ok]
            vd = np.maximum(qd[ok] / nd[ok] - md ** 2, 1e-12); vv = np.maximum(qv[ok] / nv[ok] - mv ** 2, 1e-12)
            td = md / np.sqrt(vd / nd[ok]); tv = mv / np.sqrt(vv / nv[ok])
            out.append(pd.DataFrame({"combo": [combo] * len(ok), "code": ok, "nd": nd[ok], "nv": nv[ok], "md": md, "mv": mv, "td": td, "tv": tv,
                                     "k": k}))
    R = pd.concat(out, ignore_index=True)
    return R

def describe(R, D):
    def lab(row):
        parts = []; code = int(row.code)
        for nme in row.combo:
            _, l = D[nme]; code, r = divmod(code, len(l)); parts.append(f"{nme}={l[r]}")
        return " & ".join(parts)
    return R.apply(lab, axis=1)

def summarise(R, tcut=3.5):
    sel = R[R.td.abs() >= tcut].copy()
    sgn = np.sign(sel.md)
    sel["mv_or"] = sel.mv * sgn; sel["tv_or"] = sel.tv * sgn; sel["md_or"] = sel.md * sgn
    rep = sel[sel.tv_or >= 2]
    return sel, rep

for uname, um in (("B price>=10, pm$>=500k", (d.price >= 10) & (d.pm_dollar >= 5e5)), ("A price>=5, pm$>=100k", (d.price >= 5) & (d.pm_dollar >= 1e5))):
    x = d[um].copy()
    x["y"] = x.R - x.groupby("date").R.transform("mean")                   # within-universe within-day relative move
    lo_, hi_ = x.y.quantile([0.001, 0.999]); x["y"] = x.y.clip(lo_, hi_)
    D = build(x); y = x.y.values; per_disc = (x.period == "disc").values
    t0 = time.time()
    R = cells(D, y, per_disc, minD=800, minV=400)
    sel, rep = summarise(R)
    print(f"\n########## universe {uname}: {len(x):,} events; cells tested (>=800 disc & >=400 val events): {len(R):,}   [{time.time()-t0:.0f}s]")
    print(f"discovery |t|>=3.5: {len(sel)} cells | replicate in validation (same sign, t>=2): {len(rep)} ({len(rep)/max(len(sel),1):.0%}) | mean effect disc {sel.md_or.mean():+.3f}% -> val {sel.mv_or.mean():+.3f}%")
    # PLACEBO: shuffle the move across tickers within each date -> same pipeline, no real effect possible
    pl = []
    for s in range(12):
        rs = np.random.RandomState(100 + s)
        ys = x.assign(y=x.y).groupby("date").y.transform(lambda v: v.values[rs.permutation(len(v))]).values
        Rp = cells(D, ys, per_disc, 800, 400); sp, rp = summarise(Rp)
        pl.append((len(sp), len(rp), sp.mv_or.mean() if len(sp) else np.nan))
    pl = pd.DataFrame(pl, columns=["sel", "rep", "mv_or"])
    print(f"PLACEBO (12 shuffles): selected {pl.sel.mean():.1f}+-{pl.sel.std():.1f}  replicating {pl.rep.mean():.1f}+-{pl.rep.std():.1f}  val effect {pl.mv_or.mean():+.3f}%")
    # strongest replicating cells
    rep = rep.copy(); rep["desc"] = describe(rep, D)
    rep["half"] = 0
    top = rep.sort_values("tv_or", ascending=False).head(18)
    print(top[["desc", "nd", "nv", "md_or", "td", "mv_or", "tv_or"]].round(3).to_string(index=False))
    rep.assign(sign=np.sign(rep.md)).to_parquet(S + f"cells_{uname[0]}.parquet", index=False)
