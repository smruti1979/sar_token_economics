#!/usr/bin/env python3
"""
draw_architecture.py - Fig. 1 of the SAR white paper (v1.1, adds the tool layer).

Requires: matplotlib.   Run:  python3 draw_architecture.py
Writes:   SAR_Control_Plane_Architecture.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"
NAVY, GREY, BLUE, AMBER = "#1F2A44", "#5B6577", "#3C78C3", "#C77F00"
GREEN, PURPLE, TEAL, TEAL_D = "#4C9A5F", "#7A5BB8", "#0F8B8D", "#0B5F61"

W, H = 2300, 2010
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def box(x, y, w, h, fc="white", ec=NAVY, lw=2.5, ls="-", r=18):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))


def text(x, y, s, size=15, color=NAVY, weight="normal", ha="center", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va="center",
            style=style, zorder=5)


def arrow(x0, y0, x1, y1, color=NAVY, lw=4.5, ls="-", both=False, scale=22):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=scale))


def line(xs, ys, color=NAVY, lw=3.5, ls="-"):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=3, solid_capstyle="butt")


# ---------------- title ----------------
text(W / 2, 72, "Token-Aware Agentic Control Plane for Cospas-Sarsat Alert Intelligence", 33, NAVY, "bold")
text(W / 2, 127, "Reference architecture: the AI agent, and every tool it can call, run beside the alert path, never in it",
     18, GREY)

# ---------------- safety-critical path ----------------
text(113, 175, "SAFETY-CRITICAL ALERT PATH  (unchanged, human-decided)", 15.5, GREY, "bold", "left")
box(88, 198, 2120, 184, "#EDF2F9", "#9DB4D8", 2.5, r=28)
box(138, 235, 518, 110); text(397, 268, "406 MHz beacon alert", 17, NAVY, "bold")
text(397, 311, "EPIRB · ELT · PLB via satellites", 14.5)
box(889, 235, 518, 110); text(1148, 268, "Mission Control Centre", 17, NAVY, "bold")
text(1148, 311, "MCC · alert + location data", 14.5)
box(1640, 235, 518, 110, NAVY, NAVY); text(1899, 268, "Rescue Coordination Centre", 17, "white", "bold")
text(1899, 311, "human controller decides", 14.5, "#D5DCEB")
arrow(656, 290, 889, 290); arrow(1407, 290, 1640, 290)
text(778, 263, "alert data", 13, GREY, style="italic"); text(1523, 263, "immediate", 13, GREY, style="italic")
arrow(1148, 345, 1148, 467, BLUE, 3.5, "--", scale=20)
text(1180, 370, "copy for assessment", 13.5, BLUE, ha="left", style="italic")
arrow(1887, 467, 1887, 345, BLUE, 3.5, "--", scale=20)
text(1918, 370, "advisory brief", 13.5, BLUE, ha="left", style="italic")

# ---------------- advisory path ----------------
box(88, 418, 2120, 1262, "#FAFAF6", "#8A93A6", 2.5, "--", 28)
text(113, 447, "ADVISORY ASSESSMENT PATH  (off the critical path, fail-open)", 15.5, GREY, "bold", "left")

box(285, 467, 1233, 98)
text(901, 500, "1   Router evaluation layer", 18, NAVY, "bold")
text(901, 540, "intent  ·  severity indicators  ·  predicted token length  ·  confidence signals", 14.5)
box(1665, 467, 443, 98)
text(1887, 500, "Case brief + cost record", 17, NAVY, "bold"); text(1887, 540, "advisory only, sources cited", 14.5, GREY)

arrow(532, 565, 532, 650, scale=20); arrow(1148, 565, 1148, 650, scale=20)
line([1518, 1591], [516, 516]); arrow(1591, 516, 1591, 650, scale=20)

# tiers
tiers = [
    (285, "#E2F2E4", GREEN, "TIER 1  ·  LITE", "Steps 1 and 3", "Parse alert  ·  Structure payload",
     "no tool access", "e.g. GPT-4o Mini, Gemini 1.5 Flash"),
    (901, "#E0EBF9", BLUE, "TIER 2  ·  CORE", "Steps 2 and 4", "Correlate (tool loop)  ·  Verify",
     "model picks tools within its budget", "e.g. Claude 3.5 Sonnet, GPT-4o"),
    (1517, "#ECE5F7", PURPLE, "TIER 3  ·  EXPERT", "Rare: high-severity or ambiguous", "Contradictory evidence",
     "may call tools, higher budget", "e.g. o1 / o3, DeepSeek-R1  (not priced)"),
]
for x, fc, ec, t1, t2, t3, t4, t5 in tiers:
    box(x, 650, 493, 220, fc, ec, 3.5, r=20)
    cx = x + 246
    text(cx, 686, t1, 18, ec, "bold"); text(cx, 728, t2, 15.5, NAVY, "bold"); text(cx, 767, t3, 14.5)
    text(cx, 802, t4, 12.5, GREY, style="italic"); text(cx, 840, t5, 12.5, GREY, style="italic")
arrow(778, 760, 895, 760, AMBER, 4.5, scale=22); arrow(1394, 760, 1511, 760, AMBER, 4.5, scale=22)
text(839, 733, "esc.", 13, AMBER, "bold"); text(1455, 733, "esc.", 13, AMBER, "bold")

# validated results bus
for x in (532, 1148, 1763):
    line([x, x], [870, 913])
line([532, 2182], [913, 913]); line([2182, 2182], [913, 516]); arrow(2182, 516, 2108, 516, scale=20)
text(1395, 898, "validated step results", 13.5, GREY, style="italic")

# tool gateway
box(901, 968, 1109, 128, "#E4F3F3", TEAL, 3.5, r=20)
text(1455, 1000, "6   Tool gateway  ·  read-only, allowlisted per tier", 17, TEAL_D, "bold")
text(1455, 1040, "schema-checked arguments  ·  per-run tool-call budget  ·  results trimmed and redacted", 13.5, NAVY)
text(1455, 1072, "results are data, never instructions  ·  every call logged to the case record", 13.5, NAVY)
arrow(1010, 870, 1010, 968, TEAL, 3.5, both=True, scale=18)
arrow(1880, 870, 1880, 968, TEAL, 3.5, both=True, scale=18)
text(1022, 946, "tool call / result", 12.5, TEAL_D, ha="left", style="italic")
text(1892, 946, "tool call / result", 12.5, TEAL_D, ha="left", style="italic")

# Tier 1 note
box(285, 968, 493, 282, "#F3F5F8", "#9AA3B2", 2, "--", 18)
text(531, 1000, "WHY TIER 1 HAS NO TOOLS", 13.5, GREY, "bold")
for i, s in enumerate(["Parsing and structuring run on", "supplied data and deterministic",
                       "decoders, so a misread position", "or beacon ID cannot originate",
                       "in a language model."]):
    text(531, 1050 + i * 36, s, 14.5)

# data sources
srcs = [("Beacon registry", "IBRD owner record"), ("Alert history", "prior activations"),
        ("Other detections", "satellites · MCCs"), ("Vessel / aircraft", "AIS · ADS-B position"),
        ("Weather · drift", "sea state, search area")]
for i, (a, b) in enumerate(srcs):
    x = 901 + i * 226
    arrow(x + 102, 1096, x + 102, 1138, TEAL, 2.5, both=True, scale=14)
    box(x, 1138, 204, 100, "white", TEAL, 2.2, r=14)
    text(x + 102, 1171, a, 13.5, NAVY, "bold"); text(x + 102, 1208, b, 12, GREY)
text(1455, 1274, "No write tools: nothing here can close, suppress, delay or contact anyone on the agent's own authority.",
     13, TEAL_D, style="italic")

# ---------------- control plane services ----------------
box(163, 1322, 1970, 316, "#F4F1EA", "#B8AE96", 2.5, r=22)
text(212, 1353, "CONTROL PLANE SERVICES  ·  applied to every model call and tool call", 15.5, "#6B6450", "bold", "left")
cards = [
    (212, "2   Prompt cache manager", ["Static prefix + tool schemas first", "Frozen, versioned prefix",
                                       "Cached reads are 50% to 90% cheaper"]),
    (696, "3   Context compactor", ["Tool results kept as evidence digests", "Protected fields never compacted:",
                                    "beacon ID, positions, decisions"]),
    (1181, "4   Escalation controller", ["Validates every Tier 1 output", "Re-routes failures upward",
                                         "Retry ceiling stops reasoning loops"]),
    (1665, "5   FinOps telemetry + guardrails", ["Cost record on every model and tool call",
                                                 "Run budget, step and tool-call caps", "Fail open to a human at the limit"]),
]
for x, title, lines_ in cards:
    box(x, 1385, 423, 196, "white", "#B8AE96", 2.2, r=16)
    text(x + 211, 1421, title, 15.5, NAVY, "bold")
    for i, s in enumerate(lines_):
        text(x + 211, 1471 + i * 41, s, 12.5, "#3A4256")

# ---------------- legend and fail-open ----------------
text(113, 1742, "HOW TO READ", 15.5, GREY, "bold", "left")
legend = [(1790, NAVY, "-", "Safety-critical alert path (unchanged)", False),
          (1840, BLUE, "--", "Advisory assessment flow", False),
          (1890, AMBER, "-", "Escalation: schema failure, low confidence, high severity", False),
          (1940, TEAL, "-", "Tool call and result (read-only, via the gateway)", True)]
for y, c, ls, label, both in legend:
    ax.plot([114, 236], [y, y], color=c, lw=5 if ls == "-" else 3.5, ls=(0, (4, 3)) if ls == "--" else "-", zorder=3)
    text(262, y, label, 14.5, NAVY, ha="left")
box(1000, 1745, 1208, 205, "#FFF4DC", AMBER, 3, r=20)
text(1050, 1780, "FAIL OPEN", 18, "#8A5A00", "bold", "left")
for i, s in enumerate(["When a token budget, step cap, tool-call ceiling or retry ceiling is reached, the agent stops",
                       "spending and hands the human controller the raw alert plus any completed analysis.",
                       "The agent never closes, suppresses, delays or marks an alert false:",
                       "every alert is real until a person confirms otherwise."]):
    text(1050, 1820 + i * 30, s, 13.5, "#5A3F00", ha="left")

fig.savefig("SAR_Control_Plane_Architecture.png", dpi=100, facecolor="white")
print("saved SAR_Control_Plane_Architecture.png")
