import numpy as np, pandas as pd, sys, math, itertools, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 140)
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
d = pd.read_parquet(S + "td.parquet")
d = d[(d.has_pm == 1) & (d.fresh == 1) & d.R.notna() & d.pm_ex.notna() & d.p_0935.notna()].copy()
U = d[(d.price >= 10) & (d.pm_dollar >= 5e5)].copy()
U["short_open"] = -U.R.clip(-30, 30)                                            # short at 09:30 open, cover ~10:00
U["short_0935"] = -((U.p_exit / U.p_0935 - 1) * 100).clip(-30, 30)               # realistic: short after the first minutes
U["pullback"] = U.pm_hi_ret - U.pm_ret                                          # depth of the retreat from the pre-market high (pp)
U["ex_bench"] = U.ex_bench_pm
print("events", len(U), "| period split:", U.groupby("period").size().to_dict())

def cut(s, edges, labels):
    c = pd.cut(s, [-np.inf] + edges + [np.inf], labels=False).astype("float")
    return c.fillna(len(labels)).astype(int).values, labels + ["na"]

D = {
    "gap":       cut(U.pm_ex, [-3, 1, 3, 5, 8, 15], ["<=-3", "-3..1", "1..3", "3..5", "5..8", "8..15", ">15"]),
    "gap_vs_sector": cut(U.ex_bench, [-2, 2, 5], ["below sector", "in line", "above sector 2-5", ">5 above sector"]),
    "pullback":  cut(U.pullback, [0.5, 2, 5], ["<0.5 at high", "0.5-2", "2-5", ">5 off high"]),
    "pos_range": cut(U.pos_in_range, [0.25, 0.75], ["low", "mid", "top"]),
    "rvol":      cut(U.rvol, [1, 3], ["<1x", "1-3x", ">3x"]),
    "vs_vwap":   cut(U.vwap_dev, [-1, 1], ["<-1%", "~vwap", ">1%"]),
    "t_high":    cut(U.t_hi, [360, 500], ["<06:00", "06:00-08:20", ">08:20"]),
    "last50":    cut(U.r_0830_0920, [-1, 1], ["fall", "flat", "rise"]),
    "early":     cut(U.r_0400_0700, [-1, 1], ["fall", "flat", "rise"]),
    "prints":    cut(U.n_pm, [10, 40, 100], ["<10", "10-40", "40-100", ">100"]),
    "price":     cut(U.price, [20, 50, 150], ["10-20", "20-50", "50-150", ">150"]),
    "pm_usd":    cut(U.pm_dollar, [2e6, 1e7], ["<2M", "2-10M", ">10M"]),
    "mkt_gap":   cut(U.mkt_pm, [-0.5, 0.5], ["market gap down", "flat", "market gap up"]),
    "gap_z":     cut(U.pm_ret_z, [1, 2, 4], ["<1 sigma", "1-2", "2-4", ">4 sigma"]),
    "gap_rel":   cut(U.gap_rel, [1, 3], ["usual", "1-3x usual", ">3x usual"]),
    "dow":       cut(U.dow.astype(float), [0.5, 1.5, 2.5, 3.5], ["Mon", "Tue", "Wed", "Thu", "Fri"]),
}
per_disc = (U.period == "disc").values

def scan(y, minD=600, minV=300, order=3):
    out = []; names = list(D)
    for k in range(1, order + 1):
        for combo in itertools.combinations(names, k):
            codes = np.zeros(len(y), np.int64); stride = 1
            for nme in combo:
                c, l = D[nme]; codes += c * stride; stride *= len(l)
            w = lambda m, a=None: np.bincount(codes[m], weights=a[m] if a is not None else None, minlength=stride)
            nd, nv = w(per_disc), w(~per_disc)
            sd, sv, qd, qv = w(per_disc, y), w(~per_disc, y), w(per_disc, y ** 2), w(~per_disc, y ** 2)
            ok = np.flatnonzero((nd >= minD) & (nv >= minV))
            if not len(ok): continue
            md, mv = sd[ok] / nd[ok], sv[ok] / nv[ok]
            vd = np.maximum(qd[ok] / nd[ok] - md ** 2, 1e-12); vv = np.maximum(qv[ok] / nv[ok] - mv ** 2, 1e-12)
            out.append(pd.DataFrame({"combo": [combo] * len(ok), "code": ok, "nd": nd[ok], "nv": nv[ok], "md": md, "mv": mv,
                                     "td": md / np.sqrt(vd / nd[ok]), "tv": mv / np.sqrt(vv / nv[ok]), "k": k}))
    return pd.concat(out, ignore_index=True)

def describe(R):
    def lab(r):
        code = int(r.code); parts = []
        for nme in r.combo:
            l = D[nme][1]; code, i = divmod(code, len(l)); parts.append(f"{nme}={l[i]}")
        return " & ".join(parts)
    return R.apply(lab, axis=1)

TCUT = 4.0
for tgt in ("short_0935", "short_open"):
    y = U[tgt].values
    t0 = time.time(); R = scan(y)
    sel = R[R.td >= TCUT]                                  # SHORT-profitable on discovery only
    rep = sel[(sel.tv >= 2.0) & (sel.mv >= 0.15)]
    print(f"\n########## target {tgt}: cells tested {len(R):,} [{time.time()-t0:.0f}s]  | disc t>={TCUT}: {len(sel)} | replicate (val t>=2 and val mean>=0.15%): {len(rep)} ({len(rep)/max(len(sel),1):.0%})")
    # placebo: shuffle the outcome across tickers within each date
    pl = []
    for s in range(8):
        rs = np.random.RandomState(300 + s)
        ys = pd.Series(y).groupby(U.date.values).transform(lambda v: v.values[rs.permutation(len(v))]).values
        Rp = scan(ys); sp = Rp[Rp.td >= TCUT]; pl.append((len(sp), len(sp[(sp.tv >= 2) & (sp.mv >= 0.15)])))
    pl = np.array(pl); print(f"   placebo ({len(pl)} shuffles): selected {pl[:,0].mean():.1f}  replicating {pl[:,1].mean():.1f}")
    rep = rep.copy(); rep["desc"] = describe(rep)
    rep.sort_values("mv", ascending=False).to_parquet(S + f"short_cells_{tgt}.parquet", index=False)
    print(rep.sort_values("tv", ascending=False).head(25)[["desc", "nd", "nv", "md", "td", "mv", "tv"]].round(3).to_string(index=False))
