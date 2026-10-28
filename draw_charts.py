#!/usr/bin/env python3
"""
draw_charts.py - data charts for the SAR token-economics white paper.

Imports the cost model from sar_token_economics.py (same folder), so every bar
is computed from the same code that produces the paper's figures.
Requires: matplotlib.   Run:  python3 draw_charts.py
Writes:   chart1_savings_bridge.png, chart2_cost_composition.png,
          chart3_caching_vs_compaction.png, chart4_robustness.png
"""
import itertools
from dataclasses import replace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sar_token_economics as m

plt.rcParams["font.family"] = "DejaVu Sans"
NAVY, GREY, AMBER = "#1F2A44", "#5B6577", "#C77F00"
GREEN, BLUE, PURPLE, LIGHT = "#4C9A5F", "#3C78C3", "#7A5BB8", "#C9D3E3"
sc = m.Scenario()
RUNS_YEAR = sc.runs_per_day * sc.days_per_year
NOTE = "Modelled outcome, 8,000 assessments/day. Assumptions: paper Section 2.3. Reproduce: sar_token_economics.py"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9AA3B2")
    ax.tick_params(colors=NAVY, labelsize=10)
    ax.yaxis.grid(True, color="#E6E9EF", lw=0.8); ax.set_axisbelow(True)


def finish(fig, ax, title, sub, path, note=None):
    fig.text(0.06, 0.955, title, fontsize=15.5, fontweight="bold", color=NAVY, ha="left", va="top")
    fig.text(0.06, 0.895, sub, fontsize=10.5, color=GREY, ha="left", va="top")
    fig.text(0.06, 0.015, note or NOTE, fontsize=8, color=GREY, ha="left", style="italic")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig); print("saved", path)


# ---- exact cost + component breakdown (mirrors m.run_pattern, proves totals match) ----
def breakdown(scn, steps):
    comp = dict(uncached=0.0, cached=0.0, output=0.0, wasted=0.0)
    def call(model, ctx, out):
        r = m.PRICES[model]
        c_tok = min(scn.static_tokens * scn.cache_hit, ctx) if scn.cache else 0.0
        return ((ctx - c_tok) * r["inp"] / 1e6, c_tok * r["cached"] / 1e6, out * r["out"] / 1e6)
    for pattern in itertools.product((False, True), repeat=len(steps)):
        p = 1.0
        for st, f in zip(steps, pattern):
            p *= st.fail_p if f else 1 - st.fail_p
        if p == 0:
            continue
        ctx = scn.static_tokens + scn.goal_tokens
        for st, failed in zip(steps, pattern):
            model, out = st.model, st.out_tokens
            if failed:
                if scn.bill_failed_attempts:
                    comp["wasted"] += p * sum(call(model, ctx, out))
                if model == m.MINI:
                    model = m.PREMIUM
                out = round(out * scn.fallback_out_mult)
            u, c, o = call(model, ctx, out)
            comp["uncached"] += p * u; comp["cached"] += p * c; comp["output"] += p * o
            ctx += out
    return comp


A_cfg = (replace(sc, cache=False), m.premium_pipeline(True))
B_cfg = (replace(sc, cache=False), m.routed_pipeline())
C_cfg = (replace(sc, cache=True), m.routed_pipeline())
cost = {k: m.exact_distribution(*cfg).mean_cost for k, cfg in zip("ABC", (A_cfg, B_cfg, C_cfg))}
comp = {k: breakdown(*cfg) for k, cfg in zip("ABC", (A_cfg, B_cfg, C_cfg))}
for k in "ABC":   # proof: component sums equal the exact expected cost
    assert abs(sum(comp[k].values()) - cost[k]) < 1e-12, k

# ======================= Chart 1: savings bridge (waterfall) =======================
ann = {k: v * RUNS_YEAR / 1000 for k, v in cost.items()}           # $ thousands per year
fig, ax = plt.subplots(figsize=(11, 6.6)); fig.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.97)
style(ax)
x = [0, 1, 2, 3]
ax.bar(0, ann["A"], color=NAVY, width=0.6)
ax.bar(1, ann["A"] - ann["B"], bottom=ann["B"], color=GREEN, width=0.6)
ax.bar(2, ann["B"] - ann["C"], bottom=ann["C"], color=BLUE, width=0.6)
ax.bar(3, ann["C"], color=NAVY, width=0.6)
for a, b in ((0.3, 0.7), (1.3, 1.7), (2.3, 2.7)):
    lvl = ann["A"] if a == 0.3 else (ann["B"] if a == 1.3 else ann["C"])
    ax.plot([a, b], [lvl, lvl], color=GREY, lw=1, ls=":")
lab = [(0, ann["A"], f"${ann['A']:,.0f}K", NAVY),
       (1, ann["A"], f"-${ann['A']-ann['B']:,.0f}K\n({(1-cost['B']/cost['A'])*100:.1f}%)", GREEN),
       (2, ann["B"], f"-${ann['B']-ann['C']:,.0f}K\n({(1-cost['C']/cost['B'])*100:.1f}% of remainder)", BLUE),
       (3, ann["C"], f"${ann['C']:,.0f}K", NAVY)]
for xi, y, s, c in lab:
    ax.text(xi, y + 7, s, ha="center", va="bottom", fontsize=11, fontweight="bold", color=c)
ax.set_xticks(x); ax.set_xticklabels(["A. Premium by\ndefault", "Smart routing\n(Tier 1 / Tier 2)", "Prompt caching\n(static prefix)", "C. Routed\nand cached"], fontsize=10.5)
ax.set_ylabel("Annual model spend, $ thousands", color=NAVY); ax.set_ylim(0, ann["A"] * 1.18)
sav = ann["A"] - ann["C"]
finish(fig, ax, f"Routing and caching together cut annual spend by {(1-cost['C']/cost['A'])*100:.1f}%",
       f"One workflow, 2.92 million assessments a year: ${sav:,.0f}K saved, roughly half from each lever", "chart1_savings_bridge.png")

# ======================= Chart 2: where each dollar goes =======================
fig, ax = plt.subplots(figsize=(11, 6.6)); fig.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.97)
style(ax)
segs = [("uncached", "Input tokens, full price", NAVY), ("cached", "Input tokens, cached read", GREEN),
        ("output", "Output tokens", BLUE), ("wasted", "Wasted first attempts (billed)", AMBER)]
names = ["A. Premium\nby default", "B. Routed", "C. Routed\nand cached"]
bottoms = [0, 0, 0]
for key, label, col in segs:
    vals = [comp[k][key] * 1000 for k in "ABC"]
    ax.bar(range(3), vals, bottom=bottoms, color=col, width=0.55, label=label)
    for i, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 4:
            ax.text(i, b + v / 2, f"${v:,.1f}", ha="center", va="center", color="white",
                    fontsize=10 if v > 8 else 8.5, fontweight="bold")
    bottoms = [b + v for b, v in zip(bottoms, vals)]
for i, t in enumerate(bottoms):
    ax.text(i, t + 2, f"${t:,.2f}", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=NAVY)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=10.5)
ax.set_ylabel("Cost per 1,000 assessments, $", color=NAVY); ax.set_ylim(0, bottoms[0] * 1.15)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
in_share = (comp["A"]["uncached"]) / cost["A"] * 100
finish(fig, ax, "Input tokens are the bill: caching and routing attack exactly that",
       f"Input is {in_share:.1f}% of baseline cost; in design C, cheap cached reads and Tier 1 steps shrink it, and failures cost only ${comp['C']['wasted']*1000:,.2f} per 1,000", "chart2_cost_composition.png")

# ======================= Chart 3: caching vs compaction on long runs =======================
steps_list = (4, 10, 20, 40)
res = {}
for n in steps_list:
    _, plain = m.long_run(n, sc, cache=False)
    _, c1 = m.long_run(n, sc, cache=True)
    _, c2 = m.long_run(n, sc, cache=False, compact_every=10)
    _, c3 = m.long_run(n, sc, cache=True, compact_every=10)
    res[n] = [(1 - c1 / plain) * 100, (1 - c2 / plain) * 100, (1 - c3 / plain) * 100]
fig, ax = plt.subplots(figsize=(11, 6.6)); fig.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.97)
style(ax)
w = 0.24
for j, (lab_, col) in enumerate((("Caching only", GREEN), ("Compaction only", PURPLE), ("Both", NAVY))):
    vals = [res[n][j] for n in steps_list]
    xs = [i + (j - 1) * w for i in range(4)]
    ax.bar(xs, vals, width=w, color=col, label=lab_)
    for xv, v in zip(xs, vals):
        ax.text(xv, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=col)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{n} steps" for n in steps_list], fontsize=10.5)
ax.set_ylabel("Cost saving vs plain premium agent, %", color=NAVY); ax.set_ylim(0, 75)
ax.legend(frameon=False, fontsize=10, loc="upper left")
finish(fig, ax, "Caching leads on short runs; compaction overtakes it on long ones",
       "Premium-model agent, +500 output tokens per step, compaction every 10 steps; crossover falls between 20 and 40 steps", "chart3_caching_vs_compaction.png",
       note="Modelled outcome; percentages are per run and independent of volume. Assumptions: paper Section 5. Reproduce: sar_token_economics.py")

# ======================= Chart 4: robustness =======================
a_cost = cost["A"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.6), sharey=True)
fig.subplots_adjust(top=0.80, bottom=0.17, left=0.07, right=0.98, wspace=0.08)
hits = (1.0, 0.9, 0.75, 0.5, 0.0)
v1 = [(1 - m.exact_distribution(replace(sc, cache=True, cache_hit=h), m.routed_pipeline()).mean_cost / a_cost) * 100 for h in hits]
fails = (0.05, 0.15, 0.30, 0.45, 0.60)
def routed_p(p):
    steps = tuple(replace(s, fail_p=p) if s.model == m.MINI else s for s in m.routed_pipeline())
    return (1 - m.exact_distribution(replace(sc, cache=True), steps).mean_cost / a_cost) * 100
v2 = [routed_p(p) for p in fails]
for ax, vals, labels, ttl, base in ((ax1, v1, [f"{int(h*100)}%" if h > 0 else "0%\n(routing only)" for h in hits], "Share of static prompt served from cache", 0),
                                    (ax2, v2, [f"{int(p*100)}%" for p in fails], "Tier 1 first-attempt failure rate", 1)):
    style(ax)
    cols = [NAVY if (i == base) else LIGHT for i in range(5)]
    ax.bar(range(5), vals, color=cols, width=0.62, edgecolor=NAVY, lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color=NAVY)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=10.5); ax.set_xlabel(ttl, color=NAVY, fontsize=10.5)
ax1.set_ylabel("Saving vs premium by default, %", color=NAVY); ax1.set_ylim(0, 85)
ax2.tick_params(left=False)
finish(fig, ax1, "The saving survives much worse assumptions than the base case",
       "Dark bar = base case. Even with a 60% Tier 1 failure rate, or half the prompt uncached, the saving stays above 50%", "chart4_robustness.png",
       note="Modelled outcome; percentages are per run and independent of volume. Assumptions: paper Section 4.4. Reproduce: sar_token_economics.py")
