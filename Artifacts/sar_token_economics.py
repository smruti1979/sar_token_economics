#!/usr/bin/env python3
"""
sar_token_economics.py
======================
Reproducible cost model behind the white paper
"Token-Aware Agentic Architecture for Cospas-Sarsat Alert Intelligence".

Every number in the paper is produced by this file. No third-party packages
are required (Python 3.9+ standard library only).

WHAT IT PROVES
--------------
1. Per-run, per-day and annual cost of three architectures:
      A  Premium-by-default  (all steps on the premium model, no caching)
      B  Routed              (Tier 1 / Tier 2 split + escalation, no caching)
      C  Routed + cached     (B + prompt caching of the static prefix)
2. Exact expected cost by ENUMERATING every failure outcome (no sampling
   noise), cross-checked by a 200,000-run Monte Carlo.
3. Tail behaviour (P99), the "escalation tax", cache-hit and failure-rate
   sensitivity, and the break-even Tier 1 failure rate.
4. Long-running agents: cumulative input tokens and cost with and without
   caching and context compaction.
5. The TOOL LAYER (v1.1): Step 2 becomes a bounded, model-directed tool loop.
   Cost of the agentic design versus the original workflow, the effect of the
   tool gateway (trimmed results + evidence digests), the tool-call ceiling,
   and sensitivity to the number of tool rounds and to result size.

USAGE
-----
    python3 sar_token_economics.py             # full report
    python3 sar_token_economics.py --selftest  # sanity checks on the maths
    python3 sar_token_economics.py --verify    # assert paper figures match
    python3 sar_token_economics.py --json      # machine-readable output
    python3 sar_token_economics.py --runs-per-day 20000 --static-tokens 8000

All scenario inputs are ASSUMPTIONS (see Scenario). Replace them with values
measured on your own pilot before using the output for budgeting.

PRICING (USD per 1M tokens) uses 2024-era list prices taken from each provider's
published pricing pages. Those prices change often: verify against current
provider documentation before relying on the results.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from dataclasses import dataclass, replace
from typing import List, Sequence, Tuple

# --------------------------------------------------------------------------
# 1. PRICING  (2024-era list prices; update before budgeting)
# --------------------------------------------------------------------------
PRICES = {
    "gpt-4o-mini": {"inp": 0.15, "cached": 0.075, "out": 0.60},      # 50% cache discount
    "claude-3-5-sonnet": {"inp": 3.00, "cached": 0.30, "out": 15.00},  # 90% cache discount
}
MINI = "gpt-4o-mini"
PREMIUM = "claude-3-5-sonnet"


# --------------------------------------------------------------------------
# 2. SCENARIO  (illustrative assumptions for a SAR alert-intelligence agent)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    name: str
    model: str          # primary model for this step
    out_tokens: int     # output tokens on a clean first attempt
    fail_p: float       # probability the first attempt fails validation


@dataclass(frozen=True)
class Scenario:
    static_tokens: int = 6000        # procedures, schemas, false-alert playbooks
    goal_tokens: int = 900           # decoded alert + registry record + history
    runs_per_day: int = 8000         # triage runs per day (illustrative)
    days_per_year: int = 365
    cache: bool = False
    cache_hit: float = 1.0           # share of the static prefix served from cache
    fallback_out_mult: float = 1.4   # escalated step emits 40% more output
    bill_failed_attempts: bool = True  # conservative: the wasted attempt is billed


def routed_pipeline() -> Tuple[Step, ...]:
    """Tier 1 on parsing/structuring, Tier 2 on planning/verification."""
    return (
        Step("1 Parse & normalise alert", MINI, 300, 0.15),
        Step("2 Correlate & assess", PREMIUM, 700, 0.05),
        Step("3 Structure case payload", MINI, 500, 0.15),
        Step("4 Verify before operator", PREMIUM, 300, 0.05),
    )


def premium_pipeline(failure_free: bool = True) -> Tuple[Step, ...]:
    return tuple(
        Step(s.name, PREMIUM, s.out_tokens, 0.0 if failure_free else 0.05)
        for s in routed_pipeline()
    )


# --------------------------------------------------------------------------
# 3. CORE COST FUNCTIONS
# --------------------------------------------------------------------------
def call_cost(model: str, ctx: int, out: int, static: int, cache: bool, hit: float) -> Tuple[float, float]:
    """USD cost of ONE model call, and the number of cached input tokens."""
    r = PRICES[model]
    cached = static * hit if cache else 0.0
    cached = min(cached, ctx)
    uncached = ctx - cached
    usd = (uncached * r["inp"] + cached * r["cached"] + out * r["out"]) / 1_000_000
    return usd, cached


@dataclass
class RunResult:
    cost: float
    input_tokens: float
    cached_tokens: float
    output_tokens: float
    escalations: int


def run_pattern(sc: Scenario, steps: Sequence[Step], pattern: Sequence[bool]) -> RunResult:
    """Cost of one run given which steps failed their first attempt."""
    ctx = sc.static_tokens + sc.goal_tokens
    total = t_in = t_cached = t_out = 0.0
    esc = 0
    for st, failed in zip(steps, pattern):
        model, out = st.model, st.out_tokens
        if failed:
            esc += 1
            if sc.bill_failed_attempts:  # wasted first attempt: same input, original output
                usd, c = call_cost(model, ctx, out, sc.static_tokens, sc.cache, sc.cache_hit)
                total += usd; t_in += ctx; t_cached += c; t_out += out
            if model == MINI:
                model = PREMIUM          # escalate the exact payload to Tier 2
            out = round(out * sc.fallback_out_mult)
        usd, c = call_cost(model, ctx, out, sc.static_tokens, sc.cache, sc.cache_hit)
        total += usd; t_in += ctx; t_cached += c; t_out += out
        ctx += out                       # state carrying: output joins next step's input
    return RunResult(total, t_in, t_cached, t_out, esc)


@dataclass
class Distribution:
    outcomes: List[Tuple[float, RunResult]]  # (probability, result)

    @property
    def mean_cost(self) -> float:
        return sum(p * r.cost for p, r in self.outcomes)

    @property
    def mean_escalations(self) -> float:
        return sum(p * r.escalations for p, r in self.outcomes)

    def mean(self, attr: str) -> float:
        return sum(p * getattr(r, attr) for p, r in self.outcomes)

    def quantile(self, q: float) -> float:
        acc = 0.0
        for p, r in sorted(self.outcomes, key=lambda x: x[1].cost):
            acc += p
            if acc >= q - 1e-12:
                return r.cost
        return max(r.cost for _, r in self.outcomes)

    @property
    def max_cost(self) -> float:
        return max(r.cost for _, r in self.outcomes)


def exact_distribution(sc: Scenario, steps: Sequence[Step], runner=None) -> Distribution:
    """Enumerate all 2^n failure patterns -> exact expectation, no sampling noise."""
    runner = runner or run_pattern
    outs = []
    for pattern in itertools.product((False, True), repeat=len(steps)):
        p = 1.0
        for st, f in zip(steps, pattern):
            p *= st.fail_p if f else (1.0 - st.fail_p)
        if p > 0:
            outs.append((p, runner(sc, steps, pattern)))
    return Distribution(outs)


def monte_carlo(sc: Scenario, steps: Sequence[Step], n: int, seed: int = 42, runner=None) -> Tuple[float, float]:
    """Independent cross-check: sample failures instead of enumerating them."""
    runner = runner or run_pattern
    rng = random.Random(seed)
    costs = []
    for _ in range(n):
        pattern = [rng.random() < st.fail_p for st in steps]
        costs.append(runner(sc, steps, pattern).cost)
    costs.sort()
    return sum(costs) / n, costs[int(0.99 * n) - 1]



# --------------------------------------------------------------------------
# 3b. TOOL LAYER (agentic Step 2)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolScenario:
    """Assumptions for the model-directed tool loop inside Step 2.

    Step 2 (Correlate and assess) becomes: `rounds` tool-calling turns, then one
    final assessment call. Every turn re-reads the whole context, so each tool
    result is paid for again on every later call (state carrying).
    """
    schema_tokens: int = 1200          # tool definitions: static, cacheable, in the Step 2 prefix
    rounds: int = 3                    # tool-calling turns in Step 2 (base case)
    calls_per_round: int = 1           # >1 = parallel tool calls in one turn
    call_out_tokens: int = 100         # model output per tool-call request
    result_tokens: int = 500           # gateway-normalised result returned to the model
    raw_result_tokens: int = 1500      # the same result without the gateway
    digest_tokens: int = 100           # evidence digest per call carried to Steps 3 and 4
    gateway: bool = True               # False = naive agent: raw results, full transcript carried on
    ceiling_rounds: int = 8            # per-run tool-call ceiling (fail open to a human at the limit)


def run_pattern_tools(sc: Scenario, steps: Sequence[Step], pattern: Sequence[bool],
                      ts: ToolScenario) -> RunResult:
    """Like run_pattern, but Step 2 is a bounded tool loop (see ToolScenario)."""
    ctx = sc.static_tokens + sc.goal_tokens
    total = t_in = t_cached = t_out = 0.0
    esc = 0

    def bill(model, c, o, static):
        nonlocal total, t_in, t_cached, t_out
        usd, cached = call_cost(model, c, o, static, sc.cache, sc.cache_hit)
        total += usd; t_in += c; t_cached += cached; t_out += o

    for idx, (st, failed) in enumerate(zip(steps, pattern)):
        model, out = st.model, st.out_tokens
        if idx == 1:                                     # Step 2: model-directed tool loop
            static = sc.static_tokens + ts.schema_tokens
            res = ts.result_tokens if ts.gateway else ts.raw_result_tokens
            per_round_ctx = ts.calls_per_round * (ts.call_out_tokens + res)
            c2 = ctx + ts.schema_tokens                  # tool definitions join the prefix
            for _ in range(ts.rounds):
                bill(model, c2, ts.calls_per_round * ts.call_out_tokens, static)
                c2 += per_round_ctx
            if failed:                                   # final assessment fails validation
                esc += 1
                if sc.bill_failed_attempts:
                    bill(model, c2, out, static)
                out = round(out * sc.fallback_out_mult)
            bill(model, c2, out, static)
            n_calls = ts.rounds * ts.calls_per_round
            # gateway: only compact digests go on to Steps 3 and 4; naive: the whole transcript
            carried = n_calls * ts.digest_tokens if ts.gateway else ts.rounds * per_round_ctx
            ctx += out + carried
            continue
        if failed:
            esc += 1
            if sc.bill_failed_attempts:
                bill(model, ctx, out, sc.static_tokens)
            if model == MINI:
                model = PREMIUM
            out = round(out * sc.fallback_out_mult)
        bill(model, ctx, out, sc.static_tokens)
        ctx += out
    return RunResult(total, t_in, t_cached, t_out, esc)


def tool_distribution(sc: Scenario, steps: Sequence[Step], ts: ToolScenario) -> "Distribution":
    return exact_distribution(sc, steps, runner=lambda s, st, pat: run_pattern_tools(s, st, pat, ts))

# --------------------------------------------------------------------------
# 4. LONG-RUNNING AGENTS (context growth, compaction, caching)
# --------------------------------------------------------------------------
def long_run(n_steps: int, sc: Scenario, cache: bool, compact_every: int = 0,
             out_per_step: int = 500, summary_tokens: int = 1000) -> Tuple[int, float]:
    """Cumulative input tokens and USD cost of an n-step premium-model agent."""
    base = sc.static_tokens + sc.goal_tokens
    ctx, t_in, cost = base, 0, 0.0
    for k in range(1, n_steps + 1):
        usd, _ = call_cost(PREMIUM, ctx, out_per_step, sc.static_tokens, cache, sc.cache_hit)
        cost += usd
        t_in += ctx
        ctx += out_per_step
        if compact_every and k % compact_every == 0:
            ctx = base + summary_tokens  # history replaced by a compact summary
    return t_in, cost


# --------------------------------------------------------------------------
# 5. REPORT
# --------------------------------------------------------------------------
def compute(sc: Scenario, mc_runs: int, ts: ToolScenario = None) -> dict:
    ts = ts or ToolScenario()
    A = exact_distribution(replace(sc, cache=False), premium_pipeline(True))
    A_fair = exact_distribution(replace(sc, cache=False), premium_pipeline(False))
    B = exact_distribution(replace(sc, cache=False), routed_pipeline())
    C = exact_distribution(replace(sc, cache=True), routed_pipeline())
    C_clean = exact_distribution(replace(sc, cache=True),
                                 tuple(replace(s, fail_p=0.0) for s in routed_pipeline()))
    src_conv = exact_distribution(replace(sc, cache=True, bill_failed_attempts=False), routed_pipeline())
    mcB = monte_carlo(replace(sc, cache=False), routed_pipeline(), mc_runs)
    mcC = monte_carlo(replace(sc, cache=True), routed_pipeline(), mc_runs)

    runs_year = sc.runs_per_day * sc.days_per_year
    a_cost, b_cost, c_cost = A.mean_cost, B.mean_cost, C.mean_cost

    # input share of cost for the premium baseline
    base_ctx = sc.static_tokens + sc.goal_tokens
    in_cost = out_cost = 0.0
    ctx = base_ctx
    for s in premium_pipeline(True):
        in_cost += ctx * PRICES[PREMIUM]["inp"] / 1e6
        out_cost += s.out_tokens * PRICES[PREMIUM]["out"] / 1e6
        ctx += s.out_tokens

    res = {
        "scenario": sc.__dict__,
        "per_run_usd": {"A": a_cost, "A_with_premium_failures": A_fair.mean_cost,
                        "B": b_cost, "C": c_cost, "C_failure_free": C_clean.mean_cost,
                        "C_source_convention": src_conv.mean_cost},
        "per_1000_runs_usd": {k: v * 1000 for k, v in
                              {"A": a_cost, "B": b_cost, "C": c_cost}.items()},
        "per_day_usd": {k: v * sc.runs_per_day for k, v in
                        {"A": a_cost, "B": b_cost, "C": c_cost}.items()},
        "annual_usd": {k: v * runs_year for k, v in {"A": a_cost, "B": b_cost, "C": c_cost}.items()},
        "savings_pct": {"routing_alone": (1 - b_cost / a_cost) * 100,
                        "cache_on_top_of_routing": (1 - c_cost / b_cost) * 100,
                        "combined": (1 - c_cost / a_cost) * 100,
                        "combined_vs_fair_baseline": (1 - c_cost / A_fair.mean_cost) * 100,
                        "failure_free_ceiling": (1 - C_clean.mean_cost / a_cost) * 100},
        "annual_saving_usd": (a_cost - c_cost) * runs_year,
        "escalation_tax_pct": (c_cost / C_clean.mean_cost - 1) * 100,
        "escalations_per_run": C.mean_escalations,
        "baseline_input_share_pct": in_cost / (in_cost + out_cost) * 100,
        "tokens_per_run": {"A_input": A.mean("input_tokens"), "C_input": C.mean("input_tokens"),
                           "C_cached": C.mean("cached_tokens"), "C_output": C.mean("output_tokens")},
        "cache_share_of_input_pct": C.mean("cached_tokens") / C.mean("input_tokens") * 100,
        "tail": {"A_p99": A.quantile(0.99), "C_p99": C.quantile(0.99), "C_max": C.max_cost,
                 "C_p99_over_mean": C.quantile(0.99) / c_cost,
                 "A_30x_run": a_cost * 30, "C_30x_run": c_cost * 30},
        "monte_carlo": {"B_mean": mcB[0], "C_mean": mcC[0], "C_p99": mcC[1],
                        "B_error_pct": (mcB[0] / b_cost - 1) * 100,
                        "C_error_pct": (mcC[0] / c_cost - 1) * 100, "runs": mc_runs},
    }

    # sensitivity: cache hit rate
    res["sens_cache_hit"] = []
    for hit in (1.0, 0.9, 0.75, 0.5, 0.0):
        d = exact_distribution(replace(sc, cache=True, cache_hit=hit), routed_pipeline())
        res["sens_cache_hit"].append({"hit": hit, "per_run": d.mean_cost,
                                      "saving_vs_A_pct": (1 - d.mean_cost / a_cost) * 100})

    # sensitivity: Tier 1 failure rate
    res["sens_tier1_failure"] = []
    for p in (0.05, 0.15, 0.30, 0.45, 0.60):
        steps = tuple(replace(s, fail_p=p) if s.model == MINI else s for s in routed_pipeline())
        d = exact_distribution(replace(sc, cache=True), steps)
        res["sens_tier1_failure"].append({"p": p, "per_run": d.mean_cost,
                                          "saving_vs_A_pct": (1 - d.mean_cost / a_cost) * 100})

    # break-even Tier 1 failure rate: routing ALONE (B, no cache) vs premium-by-default (A, no cache)
    def routed_at(p, cache):
        steps = tuple(replace(s, fail_p=p) if s.model == MINI else s for s in routed_pipeline())
        return exact_distribution(replace(sc, cache=cache), steps).mean_cost
    res["routing_cost_at_100pct_tier1_failure"] = routed_at(1.0, False)
    lo, hi = 0.0, 1.0
    if routed_at(1.0, False) < a_cost:
        lo = 1.0                       # routing wins even if Tier 1 always fails
    else:
        for _ in range(60):
            mid = (lo + hi) / 2
            if routed_at(mid, False) < a_cost:
                lo = mid
            else:
                hi = mid
    res["breakeven_tier1_failure_pct"] = lo * 100

    # long-running agents
    res["long_run"] = []
    for n in (4, 10, 20, 40):
        tin_plain, c_plain = long_run(n, sc, cache=False)
        _, c_cache = long_run(n, sc, cache=True)
        tin_comp, c_comp = long_run(n, sc, cache=False, compact_every=10)
        _, c_both = long_run(n, sc, cache=True, compact_every=10)
        res["long_run"].append({"steps": n, "input_tokens_plain": tin_plain,
                                "input_tokens_compacted": tin_comp,
                                "input_reduction_pct": (1 - tin_comp / tin_plain) * 100,
                                "cost_plain": c_plain, "cost_cache": c_cache,
                                "cost_compact": c_comp, "cost_both": c_both,
                                "cache_saving_pct": (1 - c_cache / c_plain) * 100,
                                "compact_saving_pct": (1 - c_comp / c_plain) * 100,
                                "both_saving_pct": (1 - c_both / c_plain) * 100})
    res["long_run_growth"] = {"steps_ratio": 10,
                              "token_ratio_4_to_40": res["long_run"][3]["input_tokens_plain"]
                              / res["long_run"][0]["input_tokens_plain"]}

    # ---- tool layer (agentic Step 2) ----
    nc, ca = replace(sc, cache=False), replace(sc, cache=True)
    naive = replace(ts, gateway=False)
    td = lambda scn, steps, t: tool_distribution(scn, steps, t)
    A1 = td(nc, premium_pipeline(True), naive)          # naive premium agent: raw results, no cache
    A1g = td(nc, premium_pipeline(True), ts)            # premium agent + gateway only
    B1 = td(nc, routed_pipeline(), ts)                  # routed + gateway, no cache
    C1n = td(ca, routed_pipeline(), naive)              # routed + cached, no gateway
    C1 = td(ca, routed_pipeline(), ts)                  # routed + cached + gateway (proposed)
    mcC1 = monte_carlo(ca, routed_pipeline(), mc_runs, runner=lambda s_, st, pat: run_pattern_tools(s_, st, pat, ts))
    a1, a1g, b1, c1n, c1 = (d.mean_cost for d in (A1, A1g, B1, C1n, C1))

    def sweep(t):
        return {"A1_naive": td(nc, premium_pipeline(True), replace(t, gateway=False)).mean_cost,
                "C1_naive": td(ca, routed_pipeline(), replace(t, gateway=False)).mean_cost,
                "C1": td(ca, routed_pipeline(), t).mean_cost}
    rounds_sweep = [{"rounds": k, **sweep(replace(ts, rounds=k))} for k in range(0, 11)]
    k_below_A0 = max((x["rounds"] for x in rounds_sweep if x["C1"] < a_cost), default=-1)
    k_zero = rounds_sweep[0]["C1"]
    par = td(ca, routed_pipeline(), replace(ts, rounds=1, calls_per_round=ts.rounds)).mean_cost
    ceil_c = td(ca, routed_pipeline(), replace(ts, rounds=ts.ceiling_rounds))
    run30 = td(ca, routed_pipeline(), replace(ts, rounds=30)).mean_cost
    run30n = td(ca, routed_pipeline(), replace(ts, rounds=30, gateway=False)).mean_cost
    res["agentic"] = {
        "tool_scenario": ts.__dict__,
        "per_run_usd": {"A1_naive": a1, "A1_gateway_only": a1g, "B1": b1, "C1_naive": c1n, "C1": c1,
                        "C1_p99": C1.quantile(0.99), "C1_max": C1.max_cost, "C1_at_ceiling": ceil_c.mean_cost,
                        "C1_runaway_30_naive": run30n, "C1_runaway_30_gateway": run30,
                        "C1_parallel": par, "C1_zero_rounds": k_zero},
        "annual_usd": {"A1_naive": a1 * runs_year, "C1": c1 * runs_year},
        "per_1000_runs_usd": {"A1_naive": a1 * 1000, "C1": c1 * 1000},
        "saving_pct": {"C1_vs_A1_naive": (1 - c1 / a1) * 100,
                       "gateway_on_A1": (1 - a1g / a1) * 100,
                       "routing_after_gateway": (1 - b1 / a1g) * 100,
                       "caching_after_routing": (1 - c1 / b1) * 100,
                       "gateway_on_C": (1 - c1 / c1n) * 100,
                       "parallel_vs_sequential": (1 - par / c1) * 100,
                       "C1_vs_A0_workflow": (1 - c1 / a_cost) * 100},
        "annual_saving_usd": (a1 - c1) * runs_year,
        "tool_tax_pct": (c1 / c_cost - 1) * 100,
        "tool_tax_usd_per_1000": (c1 - c_cost) * 1000,
        "rounds_C1_stays_below_A0": k_below_A0,
        "tail": {"C1_p99_over_mean": C1.quantile(0.99) / c1, "ceiling_over_mean": ceil_c.mean_cost / c1,
                 "runaway_naive_over_mean": run30n / c1},
        "cache_share_of_input_pct": C1.mean("cached_tokens") / C1.mean("input_tokens") * 100,
        "tokens_per_run": {"C1_input": C1.mean("input_tokens"), "C1_cached": C1.mean("cached_tokens"),
                           "C1_output": C1.mean("output_tokens")},
        "monte_carlo": {"C1_mean": mcC1[0], "C1_error_pct": (mcC1[0] / c1 - 1) * 100},
        "rounds_sweep": rounds_sweep,
        "result_size_sweep": [{"result_tokens": r, "C1": td(ca, routed_pipeline(), replace(ts, result_tokens=r)).mean_cost}
                              for r in (250, 500, 1000, 1500, 3000)],
        "cache_hit_sweep": [{"hit": h, "C1": td(replace(sc, cache=True, cache_hit=h), routed_pipeline(), ts).mean_cost}
                            for h in (1.0, 0.75, 0.5, 0.0)],
    }
    return res


def print_report(r: dict) -> None:
    sc = r["scenario"]
    line = "=" * 78
    print(line); print("SAR ALERT-INTELLIGENCE AGENT : TOKEN ECONOMICS (reproducible)".center(78)); print(line)
    print(f"Assumptions: static prompt {sc['static_tokens']:,} tok | goal {sc['goal_tokens']:,} tok | "
          f"{sc['runs_per_day']:,} runs/day | {sc['days_per_year']} days")
    print(f"Failed attempts billed: {sc['bill_failed_attempts']} | escalation output x{sc['fallback_out_mult']}")
    print("\n[1] THREE ARCHITECTURES (exact expectation over all failure outcomes)")
    print(f"{'Architecture':<34}{'per run':>10}{'per 1,000':>12}{'per day':>12}{'per year':>14}")
    names = {"A": "A  Premium by default", "B": "B  Routed", "C": "C  Routed + cached"}
    for k in "ABC":
        print(f"{names[k]:<34}${r['per_run_usd'][k]:>9.5f}${r['per_1000_runs_usd'][k]:>10.2f}"
              f"${r['per_day_usd'][k]:>10,.0f}${r['annual_usd'][k]:>12,.0f}")
    s = r["savings_pct"]
    print(f"\nRouting alone                : {s['routing_alone']:.1f}% reduction")
    print(f"Caching on top of routing    : {s['cache_on_top_of_routing']:.1f}% further reduction")
    print(f"Combined (A -> C)            : {s['combined']:.1f}%  = ${r['annual_saving_usd']:,.0f} per year")
    print(f"Combined vs fair baseline*   : {s['combined_vs_fair_baseline']:.1f}%   (*baseline given the same 5% premium-failure rate)")
    print(f"Failure-free ceiling         : {s['failure_free_ceiling']:.1f}% (per run ${r['per_run_usd']['C_failure_free']:.5f})")
    print(f"Escalation tax               : +{r['escalation_tax_pct']:.1f}% over the failure-free pipeline")
    print(f"Escalation/retry events/run  : {r['escalations_per_run']:.2f}")
    print(f"Input share of baseline cost : {r['baseline_input_share_pct']:.1f}%")
    t = r["tokens_per_run"]
    print(f"Tokens/run (C): input {t['C_input']:,.0f} (of which cached {t['C_cached']:,.0f} = "
          f"{r['cache_share_of_input_pct']:.1f}%), output {t['C_output']:,.0f}")
    print(f"Source-script convention (failed attempt NOT billed): C = ${r['per_run_usd']['C_source_convention']:.5f}/run")

    print("\n[2] MONTE CARLO CROSS-CHECK (%s runs, seed 42)" % f"{r['monte_carlo']['runs']:,}")
    m = r["monte_carlo"]
    print(f"B mean ${m['B_mean']:.5f} (error {m['B_error_pct']:+.2f}%) | C mean ${m['C_mean']:.5f} "
          f"(error {m['C_error_pct']:+.2f}%) | C P99 ${m['C_p99']:.5f}")

    print("\n[3] TAIL RISK")
    tl = r["tail"]
    print(f"Exact P99 per run : A ${tl['A_p99']:.5f} | C ${tl['C_p99']:.5f} ({tl['C_p99_over_mean']:.2f}x C mean) | C worst case ${tl['C_max']:.5f}")
    print(f"A 30x-volatility run costs ${tl['A_30x_run']:.2f}; the same run on C costs ${tl['C_30x_run']:.2f}")

    print("\n[4] SENSITIVITY - prompt-cache hit rate (share of static prefix served from cache)")
    for x in r["sens_cache_hit"]:
        print(f"  hit {x['hit']*100:>5.0f}% -> ${x['per_run']:.5f}/run | saving vs A {x['saving_vs_A_pct']:.1f}%")
    print("\n[5] SENSITIVITY - Tier 1 first-attempt failure probability")
    for x in r["sens_tier1_failure"]:
        print(f"  p = {x['p']*100:>4.0f}% -> ${x['per_run']:.5f}/run | saving vs A {x['saving_vs_A_pct']:.1f}%")
    print(f"  Break-even (routing alone, no cache, vs premium-by-default): Tier 1 failure rate "
          f"{r['breakeven_tier1_failure_pct']:.1f}% (routed cost at 100% failure: "
          f"${r['routing_cost_at_100pct_tier1_failure']:.5f}/run vs A ${r['per_run_usd']['A']:.5f})")

    print("\n[6] LONG-RUNNING AGENT (premium model, +500 output tokens per step, compaction every 10 steps)")
    print(f"{'Steps':>5}{'Input tok':>12}{'Compacted':>12}{'Cut':>6}{'Plain $':>10}{'Cache $':>10}{'Compact $':>11}{'Both $':>9}")
    for x in r["long_run"]:
        print(f"{x['steps']:>5}{x['input_tokens_plain']:>12,}{x['input_tokens_compacted']:>12,}"
              f"{x['input_reduction_pct']:>5.0f}%{x['cost_plain']:>10.3f}{x['cost_cache']:>10.3f}"
              f"{x['cost_compact']:>11.3f}{x['cost_both']:>9.3f}")
    print("  Saving vs plain: " + " | ".join(
        f"{x['steps']} steps: cache {x['cache_saving_pct']:.0f}%, compact {x['compact_saving_pct']:.0f}%, "
        f"both {x['both_saving_pct']:.0f}%" for x in r["long_run"]))
    g = r["long_run_growth"]
    print(f"  10x the steps (4 -> 40) = {g['token_ratio_4_to_40']:.1f}x the input tokens")

    ag = r["agentic"]
    t = ag["tool_scenario"]
    print(f"\n[7] TOOL LAYER (agentic Step 2: {t['rounds']} sequential tool rounds, {t['result_tokens']}-token gateway results,"
          f" {t['schema_tokens']}-token tool schemas, ceiling {t['ceiling_rounds']} rounds)")
    pr, sv = ag["per_run_usd"], ag["saving_pct"]
    print(f"{'Design':<46}{'per run':>10}{'per 1,000':>12}")
    for lab, key in (("Workflow, premium (A, no tools)", None), ("Workflow, routed + cached (C, no tools)", None),
                     ("A1 Naive premium agent (raw results, no cache)", "A1_naive"),
                     ("   + gateway only", "A1_gateway_only"),
                     ("B1 Routed + gateway, no cache", "B1"),
                     ("C1 Routed + cached, NO gateway", "C1_naive"),
                     ("C1 Routed + cached + gateway (proposed)", "C1")):
        v = r["per_run_usd"]["A"] if lab.startswith("Workflow, premium") else \
            r["per_run_usd"]["C"] if lab.startswith("Workflow, routed") else pr[key]
        print(f"{lab:<46}${v:>9.5f}${v*1000:>10.2f}")
    print(f"C1 vs naive premium agent    : {sv['C1_vs_A1_naive']:.1f}% lower = ${ag['annual_saving_usd']:,.0f} per year "
          f"(${ag['annual_usd']['A1_naive']:,.0f} -> ${ag['annual_usd']['C1']:,.0f})")
    print(f"  decomposition (in order)   : gateway {sv['gateway_on_A1']:.1f}%, then routing {sv['routing_after_gateway']:.1f}%, "
          f"then caching {sv['caching_after_routing']:.1f}% of the remainder")
    print(f"Gateway alone on design C1   : {sv['gateway_on_C']:.1f}% lower than raw results")
    print(f"Tool tax vs workflow C       : +{ag['tool_tax_pct']:.1f}% (+${ag['tool_tax_usd_per_1000']:.2f} per 1,000)")
    print(f"C1 vs premium workflow A     : {sv['C1_vs_A0_workflow']:.1f}% lower; C1 stays below A up to {ag['rounds_C1_stays_below_A0']} tool rounds")
    print(f"Parallel tool calls (1 round x {t['rounds']} calls): ${pr['C1_parallel']:.5f}/run ({sv['parallel_vs_sequential']:.1f}% below sequential)")
    print(f"Tail: C1 P99 ${pr['C1_p99']:.5f} ({ag['tail']['C1_p99_over_mean']:.2f}x mean) | at the {t['ceiling_rounds']}-round ceiling "
          f"${pr['C1_at_ceiling']:.5f} ({ag['tail']['ceiling_over_mean']:.2f}x) | 30-round runaway: "
          f"${pr['C1_runaway_30_naive']:.2f} naive ({ag['tail']['runaway_naive_over_mean']:.1f}x), ${pr['C1_runaway_30_gateway']:.2f} with gateway")
    print(f"Tokens/run (C1): input {ag['tokens_per_run']['C1_input']:,.0f} (cached {ag['tokens_per_run']['C1_cached']:,.0f} = "
          f"{ag['cache_share_of_input_pct']:.1f}%), output {ag['tokens_per_run']['C1_output']:,.0f}")
    print(f"Monte Carlo check (C1)       : mean ${ag['monte_carlo']['C1_mean']:.5f} (error {ag['monte_carlo']['C1_error_pct']:+.2f}%)")
    print("  Tool rounds -> $/run: " + " | ".join(f"{x['rounds']}: A1 {x['A1_naive']:.3f} / C1n {x['C1_naive']:.3f} / C1 {x['C1']:.3f}"
                                                  for x in ag["rounds_sweep"] if x["rounds"] in (0, 1, 3, 5, 8, 10)))
    print("  Result size (tokens) -> C1 $/run: " + " | ".join(f"{x['result_tokens']}: {x['C1']:.4f}" for x in ag["result_size_sweep"]))
    print("  Cache share -> C1 $/run: " + " | ".join(f"{x['hit']*100:.0f}%: {x['C1']:.4f}" for x in ag["cache_hit_sweep"]))
    print(line)


# --------------------------------------------------------------------------
# 6. SELF-TEST AND VERIFICATION AGAINST THE PAPER
# --------------------------------------------------------------------------
PUBLISHED = {  # figures printed in the white paper (default scenario), with tolerances
    "A per run": (lambda r: r["per_run_usd"]["A"], 0.0, 0.00005),
    "B per run": (lambda r: r["per_run_usd"]["B"], 0.0, 0.00005),
    "C per run": (lambda r: r["per_run_usd"]["C"], 0.0, 0.00005),
    "Annual A": (lambda r: r["annual_usd"]["A"], 0.0, 1000),
    "Annual C": (lambda r: r["annual_usd"]["C"], 0.0, 1000),
    "Annual saving": (lambda r: r["annual_saving_usd"], 0.0, 1000),
    "Combined saving %": (lambda r: r["savings_pct"]["combined"], 0.0, 0.05),
    "Routing-alone %": (lambda r: r["savings_pct"]["routing_alone"], 0.0, 0.05),
    "Cache-on-routing %": (lambda r: r["savings_pct"]["cache_on_top_of_routing"], 0.0, 0.05),
    "Escalation tax %": (lambda r: r["escalation_tax_pct"], 0.0, 0.05),
    "Break-even Tier 1 failure %": (lambda r: r["breakeven_tier1_failure_pct"], 0.0, 0.05),
    "Agentic A1 per run": (lambda r: r["agentic"]["per_run_usd"]["A1_naive"], 0.0, 0.00005),
    "Agentic C1 per run": (lambda r: r["agentic"]["per_run_usd"]["C1"], 0.0, 0.00005),
    "Agentic annual A1": (lambda r: r["agentic"]["annual_usd"]["A1_naive"], 0.0, 1000),
    "Agentic annual C1": (lambda r: r["agentic"]["annual_usd"]["C1"], 0.0, 1000),
    "Agentic annual saving": (lambda r: r["agentic"]["annual_saving_usd"], 0.0, 1000),
    "Agentic saving C1 vs A1 %": (lambda r: r["agentic"]["saving_pct"]["C1_vs_A1_naive"], 0.0, 0.05),
    "Gateway saving on C1 %": (lambda r: r["agentic"]["saving_pct"]["gateway_on_C"], 0.0, 0.05),
    "Tool tax %": (lambda r: r["agentic"]["tool_tax_pct"], 0.0, 0.05),
    "Parallel-call saving %": (lambda r: r["agentic"]["saving_pct"]["parallel_vs_sequential"], 0.0, 0.05),
    "Rounds C1 stays below A": (lambda r: float(r["agentic"]["rounds_C1_stays_below_A0"]), 0.0, 0.0),
}


def selftest() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    sc = Scenario()
    # hand calculation: one premium call, 6,900 in / 300 out, no cache = $0.0252
    usd, _ = call_cost(PREMIUM, 6900, 300, 6000, False, 1.0)
    check("hand-calc premium call = $0.0252", abs(usd - 0.0252) < 1e-12)
    # with cache: 6000*0.30 + 900*3.00 + 300*15.00 = 9,000 -> $0.009
    usd, cached = call_cost(PREMIUM, 6900, 300, 6000, True, 1.0)
    check("hand-calc cached premium call = $0.009", abs(usd - 0.009) < 1e-12 and cached == 6000)
    for label, steps in (("routed", routed_pipeline()), ("premium", premium_pipeline(False))):
        d = exact_distribution(sc, steps)
        check(f"{label}: outcome probabilities sum to 1", abs(sum(p for p, _ in d.outcomes) - 1) < 1e-12)
    clean = exact_distribution(sc, premium_pipeline(True))
    check("failure-free baseline has a single outcome", len(clean.outcomes) == 1)
    check("cache never increases cost", exact_distribution(replace(sc, cache=True), routed_pipeline()).mean_cost
          < exact_distribution(sc, routed_pipeline()).mean_cost)
    exact = exact_distribution(replace(sc, cache=True), routed_pipeline()).mean_cost
    mc, _ = monte_carlo(replace(sc, cache=True), routed_pipeline(), 200_000)
    check(f"Monte Carlo within 1% of exact ({(mc/exact-1)*100:+.2f}%)", abs(mc / exact - 1) < 0.01)
    # --- tool layer ---
    for cache in (False, True):                       # zero tool rounds and no schema == the original workflow
        z = ToolScenario(rounds=0, schema_tokens=0)
        same = abs(tool_distribution(replace(sc, cache=cache), routed_pipeline(), z).mean_cost
                   - exact_distribution(replace(sc, cache=cache), routed_pipeline()).mean_cost) < 1e-12
        check(f"tool layer with 0 rounds reproduces the workflow (cache={cache})", same)
    one = run_pattern_tools(sc, premium_pipeline(True), [False] * 4, ToolScenario(rounds=1)).cost
    check(f"hand-calc premium agent, 1 tool round = $0.1509 (got {one:.4f})", abs(one - 0.1509) < 1e-9)
    td_ = [tool_distribution(replace(sc, cache=True), routed_pipeline(), ToolScenario(rounds=k)).mean_cost for k in range(6)]
    check("cost rises with every extra tool round", all(a < b for a, b in zip(td_, td_[1:])))
    check("gateway never costs more than raw results",
          td_[3] < tool_distribution(replace(sc, cache=True), routed_pipeline(), ToolScenario(rounds=3, gateway=False)).mean_cost)
    check("tool-layer outcome probabilities sum to 1",
          abs(sum(p for p, _ in tool_distribution(sc, routed_pipeline(), ToolScenario()).outcomes) - 1) < 1e-12)
    tsx = ToolScenario()
    mc_t, _ = monte_carlo(replace(sc, cache=True), routed_pipeline(), 200_000,
                          runner=lambda s_, st, pat: run_pattern_tools(s_, st, pat, tsx))
    ex_t = tool_distribution(replace(sc, cache=True), routed_pipeline(), tsx).mean_cost
    check(f"tool-layer Monte Carlo within 1% of exact ({(mc_t/ex_t-1)*100:+.2f}%)", abs(mc_t / ex_t - 1) < 0.01)
    t4, _ = long_run(4, sc, False)
    check("long-run 4 steps input tokens = 6,900*4 + 500*6 = 30,600", t4 == 6900 * 4 + 500 * (0 + 1 + 2 + 3))
    return ok


def verify(r: dict) -> bool:
    print("Checking computed values against the figures printed in the white paper:")
    ok = True
    for name, (fn, _, tol) in PUBLISHED.items():
        got = fn(r)
        want = PAPER_VALUES[name]
        good = abs(got - want) <= tol
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<30} paper {want:>14,.5f}  computed {got:>14,.5f}")
    return ok


# Values as printed in the paper for the default scenario (v1.1 adds the tool-layer figures).
PAPER_VALUES = {
    'A per run': 0.1182,
    'B per run': 0.07734,
    'C per run': 0.03756,
    'Annual A': 345144.0,
    'Annual C': 109678.704,
    'Annual saving': 235465.296,
    'Combined saving %': 68.22234,
    'Routing-alone %': 34.56751,
    'Cache-on-routing %': 51.43442,
    'Escalation tax %': 20.38846,
    'Break-even Tier 1 failure %': 79.86808,
    'Agentic A1 per run': 0.2595,
    'Agentic C1 per run': 0.07207619999999998,
    'Agentic annual A1': 757740.0,
    'Agentic annual C1': 210462.50399999993,
    'Agentic annual saving': 547277.496,
    'Agentic saving C1 vs A1 %': 72.22497109826591,
    'Gateway saving on C1 %': 32.890693958726736,
    'Tool tax %': 91.89003546212578,
    'Parallel-call saving %': 23.624996878303783,
    'Rounds C1 stays below A': 5.0,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--mc-runs", type=int, default=200_000)
    ap.add_argument("--runs-per-day", type=int, default=Scenario.runs_per_day)
    ap.add_argument("--static-tokens", type=int, default=Scenario.static_tokens)
    ap.add_argument("--goal-tokens", type=int, default=Scenario.goal_tokens)
    ap.add_argument("--cache-hit", type=float, default=Scenario.cache_hit)
    ap.add_argument("--tool-rounds", type=int, default=ToolScenario.rounds, help="tool-calling turns in Step 2")
    ap.add_argument("--result-tokens", type=int, default=ToolScenario.result_tokens, help="tokens per gateway-normalised tool result")
    ap.add_argument("--no-bill-failed", action="store_true",
                    help="do not bill the wasted first attempt of a failed step (simpler convention)")
    a = ap.parse_args(argv)

    if a.selftest:
        print("SELF-TEST"); return 0 if selftest() else 1

    sc = Scenario(static_tokens=a.static_tokens, goal_tokens=a.goal_tokens, runs_per_day=a.runs_per_day,
                  cache_hit=a.cache_hit, bill_failed_attempts=not a.no_bill_failed)
    ts = ToolScenario(rounds=a.tool_rounds, result_tokens=a.result_tokens)
    r = compute(sc, a.mc_runs, ts)
    if a.json:
        print(json.dumps(r, indent=2)); return 0
    print_report(r)
    if a.verify:
        print(); return 0 if verify(r) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
