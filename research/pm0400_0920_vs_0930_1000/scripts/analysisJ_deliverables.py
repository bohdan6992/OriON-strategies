import numpy as np, pandas as pd, sys, math, os, itertools, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = "C:/Users/wsu/AppData/Local/Temp/claude/42361bea28bb7ab2/78a15a39-9e9a-4ac7-972b-918884fe43af/scratchpad/"
OUT = "C:/Users/wsu/Desktop/CenturiON/OriON-strategies/research/pm0400_0920_vs_0930_1000/"
os.makedirs(OUT, exist_ok=True)

d = pd.read_parquet(S + "td.parquet")
d["Rc"] = d.R.clip(-30, 30)
d = d[(d.fresh == 1) & d.R.notna()].copy()
d["big2"] = (d.Rc.abs() >= 2).astype(int); d["big3"] = (d.Rc.abs() >= 3).astype(int)
pm = d[(d.has_pm == 1) & d.pm_ex.notna()]
U = pm[(pm.price >= 10) & (pm.pm_dollar >= 5e5)].copy()                  # the liquid universe used for every conclusion
sessions = np.sort(d.date.unique()); last60 = sessions[-60:]
print("universe B events", len(U), "| sessions", len(sessions), "| split date (first validation session)", d[d.period == "val"].date.min())

# ------------------------------------------------------------------ 1. habitual movers
g = U.groupby("tk")
mv = g.agg(ticker=("ticker", "first"), liquid_days=("big2", "size"), p_move2_all=("big2", "mean"), p_move3_all=("big3", "mean"),
           mean_abs_move=("Rc", lambda s: s.abs().mean()), up_share_of_big=("Rc", lambda s: (s[s.abs() >= 2] > 0).mean()),
           median_price=("price", "median"), median_pm_usd=("pm_dollar", "median"), median_pm_range=("pm_range", "median"),
           median_pm_prints=("n_pm", "median"), median_sigma=("sigma", "median"))
mv["p_move2_disc"] = U[U.period == "disc"].groupby("tk").big2.mean(); mv["n_disc"] = U[U.period == "disc"].groupby("tk").size()
mv["p_move2_val"] = U[U.period == "val"].groupby("tk").big2.mean(); mv["n_val"] = U[U.period == "val"].groupby("tk").size()
l60 = U[U.date.isin(last60)].groupby("tk").big2.agg(["mean", "size"])
mv["p_move2_last60"] = l60["mean"]; mv["n_last60"] = l60["size"]
mv = mv[mv.liquid_days >= 40].copy()
mv["rank_all"] = mv.p_move2_all.rank(ascending=False, method="first")
mv["rank_by_disc_only"] = mv.p_move2_disc.where(mv.n_disc >= 30).rank(ascending=False, method="first")
mv = mv.sort_values("rank_all")
mv.drop(columns=[]).round(4).to_csv(OUT + "movers_ticker_list.csv", index=False)
print("movers list:", len(mv), "tickers (>=40 liquid days)")

# honest out-of-sample table for the list
st = mv.dropna(subset=["p_move2_disc", "p_move2_val"])
st = st[(st.n_disc >= 40) & (st.n_val >= 20)]
oos = []
for k in (25, 50, 100, 200, 400):
    top = st.loc[st.p_move2_disc.rank(ascending=False) <= k]
    oos.append({"list": f"top {k} by discovery frequency", "p_move2_disc": top.p_move2_disc.mean(), "p_move2_val": top.p_move2_val.mean(), "all_tickers_val": st.p_move2_val.mean()})
pd.DataFrame(oos).round(4).to_csv(OUT + "movers_list_out_of_sample.csv", index=False)
print(pd.DataFrame(oos).round(3).to_string(index=False)); print("persistence corr disc->val:", round(st.p_move2_disc.corr(st.p_move2_val), 3), "tickers", len(st))

# ------------------------------------------------------------------ 2. conditions for a big move (probabilities), validation only
hist = U[U.period == "disc"].groupby("tk").big2.agg(["mean", "size"]); hist = hist[hist["size"] >= 20]["mean"].rename("hist_p2")
V = U[U.period == "val"].merge(hist, left_on="tk", right_index=True, how="inner")
V["ticker_habit"] = pd.cut(V.hist_p2, [0, 0.10, 0.20, 0.35, 0.55, 1.0], labels=["<10%", "10-20%", "20-35%", "35-55%", ">55%"])
V["pm_range_today"] = pd.cut(V.pm_range, [0, 2, 3, 5, 8, 12, 1000], labels=["<=2%", "2-3%", "3-5%", "5-8%", "8-12%", ">12%"])
P = V.groupby(["ticker_habit", "pm_range_today"]).agg(n=("big2", "size"), p_move_ge2=("big2", "mean"), p_move_ge3=("big3", "mean"), mean_abs_move=("Rc", lambda s: s.abs().mean())).reset_index()
P.round(4).to_csv(OUT + "conditions_big_move.csv", index=False)
print(P.pivot(index="ticker_habit", columns="pm_range_today", values="p_move_ge2").round(2).to_string())

# ------------------------------------------------------------------ 3. gap curve + fade rules
edges = [-100, -10, -5, -3, -1, 1, 3, 5, 8, 10, 15, 100]
U["gap_bucket"] = pd.cut(U.pm_ex, edges)
curve = U.groupby("gap_bucket").agg(n=("Rc", "size"), mean_move=("Rc", "mean"), median_move=("R", "median"), p_down=("Rc", lambda s: (s < 0).mean())).reset_index()
curve["mean_disc"] = U[U.period == "disc"].groupby("gap_bucket").Rc.mean().values; curve["mean_val"] = U[U.period == "val"].groupby("gap_bucket").Rc.mean().values
curve["gap_bucket"] = curve.gap_bucket.astype(str)
curve.round(4).to_csv(OUT + "gap_size_curve.csv", index=False)

U["short_open"] = -U.Rc; U["short_0935"] = -((U.p_exit / U.p_0935 - 1) * 100).clip(-30, 30)
rare = U[U.pm_ex > 5].groupby("tk").size()
rules = [("gap > 3%", U.pm_ex > 3), ("gap > 5%", U.pm_ex > 5), ("gap > 8%", U.pm_ex > 8), ("gap > 10%", U.pm_ex > 10), ("gap > 15%", U.pm_ex > 15),
         ("gap > 5% and gap > 5 sigma", (U.pm_ex > 5) & (U.pm_ex / U.sigma > 5)),
         ("gap > 5% and pre-market close in the middle of its range", (U.pm_ex > 5) & (U.pos_in_range > .25) & (U.pos_in_range < .75)),
         ("gap > 5% and ticker has <=9 such days in 2.7y", (U.pm_ex > 5) & (U.tk.map(rare).fillna(0) <= 9))]
rows = []
for nm, m in rules:
    x = U[m]; v = x[x.period == "val"]; dsc = x[x.period == "disc"]
    dly = v.groupby("date").short_open.mean()
    rows.append({"rule (SHORT at 09:30 open, cover at 10:00)": nm, "events": len(x), "events_per_day": round(len(x) / x.date.nunique(), 1),
                 "gross_disc_%": dsc.short_open.mean(), "gross_val_%": v.short_open.mean(), "win_rate_val": (v.short_open > 0).mean(),
                 "val_daily_t": dly.mean() / (dly.std() / math.sqrt(len(dly))), "entry_0935_val_%": v.short_0935.mean(),
                 "net_val_10bp_each_way_%": v.short_open.mean() - 0.2, "net_val_20bp_each_way_%": v.short_open.mean() - 0.4,
                 "worst_1pct_trade_val_%": v.short_open.quantile(0.01)})
pd.DataFrame(rows).round(3).to_csv(OUT + "gap_fade_rules.csv", index=False)

# ------------------------------------------------------------------ 4. serial gappers (informational)
G = U[U.pm_ex > 5]
sg = G.groupby("tk").agg(ticker=("ticker", "first"), events=("Rc", "size"), mean_move_open_to_1000=("Rc", "mean"), p_down=("Rc", lambda s: (s < 0).mean()),
                         median_price=("price", "median"), median_pm_usd=("pm_dollar", "median"))
sg = sg[sg.events >= 10].sort_values("events", ascending=False)
sg.round(4).to_csv(OUT + "serial_gappers_info_only.csv", index=False)

# ------------------------------------------------------------------ 5. the figure
fig, ax = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True)
a = ax[0, 0]; xs = np.arange(len(curve))
a.axhline(0, color="#999", lw=.8); a.plot(xs, curve.mean_disc, "o-", color="#4f7cff", label="discovery"); a.plot(xs, curve.mean_val, "s-", color="#f59e0b", label="validation")
a.set_xticks(xs); a.set_xticklabels([s.replace("(", "").replace("]", "") for s in curve.gap_bucket], rotation=45, ha="right", fontsize=8)
a.set_title("Move 09:30-10:00 (%) vs pre-market gap size (%, net of market)"); a.legend(frameon=False); a.grid(alpha=.2); a.set_ylabel("mean move, % from the open print")
a = ax[0, 1]
r = pd.cut(U.pm_range, [0, 1, 2, 3, 5, 8, 12, 20, 1000]); t2 = U.groupby(r).big2.mean(); td_ = U[U.period == "disc"].groupby(pd.cut(U[U.period == "disc"].pm_range, [0, 1, 2, 3, 5, 8, 12, 20, 1000])).big2.mean(); tv_ = U[U.period == "val"].groupby(pd.cut(U[U.period == "val"].pm_range, [0, 1, 2, 3, 5, 8, 12, 20, 1000])).big2.mean()
xs2 = np.arange(len(t2)); a.plot(xs2, td_.values, "o-", color="#4f7cff", label="discovery"); a.plot(xs2, tv_.values, "s-", color="#f59e0b", label="validation")
a.set_xticks(xs2); a.set_xticklabels([str(i).replace("(", "").replace("]", "") for i in t2.index], rotation=45, ha="right", fontsize=8); a.set_ylim(0, 1)
a.set_title("P(|move 09:30-10:00| >= 2%) vs pre-market range (%)"); a.legend(frameon=False); a.grid(alpha=.2)
a = ax[1, 0]
a.scatter(st.p_move2_disc, st.p_move2_val, s=9, alpha=.5, color="#2fbf9b"); a.plot([0, 1], [0, 1], color="#999", lw=.8)
a.set_title(f"Ticker's own frequency of >=2% moves: discovery vs validation  (corr {st.p_move2_disc.corr(st.p_move2_val):.2f}, {len(st)} tickers)")
a.set_xlabel("discovery"); a.set_ylabel("validation"); a.grid(alpha=.2)
a = ax[1, 1]
labels = ["gap>3%", "gap>5%", "gap>10%", "gap>5% mid-range"]; masks = [U.pm_ex > 3, U.pm_ex > 5, U.pm_ex > 10, (U.pm_ex > 5) & (U.pos_in_range > .25) & (U.pos_in_range < .75)]
w = .27; xs3 = np.arange(len(labels))
for k, (nm, col, c) in enumerate((("enter at open print", "short_open", "#4f7cff"), ("enter ~09:35", "short_0935", "#f59e0b"))):
    vals = [U[m & (U.period == "val")][col].mean() for m in masks]; a.bar(xs3 + (k - .5) * w, vals, w, label=nm, color=c)
vals = [U[m & (U.period == "val")].short_open.mean() - 0.4 for m in masks]; a.bar(xs3 + 1.5 * w, vals, w, label="open entry, 20bp each way", color="#e0707f")
a.axhline(0, color="#999", lw=.8); a.set_xticks(xs3); a.set_xticklabels(labels); a.set_title("SHORT the gap-up: validation PnL per trade (%)"); a.legend(frameon=False, fontsize=8); a.grid(alpha=.2, axis="y")
fig.suptitle("final.parquet 2024-02..2026-11: pre-market 04:00-09:20 vs the 09:30->10:00 move (liquid universe: price>=10, pre-market $vol>=500k)", fontsize=12, fontweight="bold")
fig.savefig(OUT + "pm_pattern_summary.png", dpi=130, facecolor="white")
print("saved", os.listdir(OUT))
