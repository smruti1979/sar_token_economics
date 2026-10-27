# Token Economics for a Cospas-Sarsat Alert Intelligence Agent

Reproducible cost model behind the white paper
**"Token-Aware Agentic Architecture for Cospas-Sarsat Alert Intelligence: Predictable Cost, Protected Safety"** (Smruti Rout, October 2026).

Every cost figure in the paper is produced by one file, `sar_token_economics.py`. It needs **Python 3.9 or later and nothing else**: no third-party packages.

> **Independent analysis.** This repository is not an official Cospas-Sarsat publication. The architecture it models is a proposal, not a programme specification.

## What the script shows

For an AI agent that performs first-line triage of Cospas-Sarsat alerts (a four-step pipeline at 8,000 assessments per day), the script computes:

1. The cost per run, per day and per year of three architectures:
   - **A. Premium by default**: all steps on the premium model, no caching.
   - **B. Routed**: a Tier 1 / Tier 2 model split with automatic escalation, no caching.
   - **C. Routed and cached**: B plus prompt caching of the static prefix.
2. Exact expected costs, found by enumerating all 16 failure outcomes (no sampling noise), cross-checked by a 200,000-run Monte Carlo simulation.
3. Tail risk (99th-percentile run), the escalation tax, sensitivity to the cache hit rate and the Tier 1 failure rate, and the break-even failure rate.
4. Long-running agents: cumulative input tokens and cost with and without caching and context compaction.

### Headline results (base case)

| Result | Value |
| --- | --- |
| Cost per assessment, A to C | $0.1182 to $0.0376 |
| Annual cost, A to C | $345,144 to $109,679 |
| Annual saving | $235,465 (68.2%) |
| Routing alone | 34.6% reduction |
| Caching on top of routing | a further 51.4% reduction |
| Break-even Tier 1 failure rate (routing alone) | 79.9% |

## Quick start

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python3 sar_token_economics.py --selftest   # sanity checks on the arithmetic
python3 sar_token_economics.py              # full report
python3 sar_token_economics.py --verify     # assert the paper's headline numbers
python3 sar_token_economics.py --json       # machine-readable output
```

Try a different scenario:

```bash
python3 sar_token_economics.py --runs-per-day 20000 --static-tokens 8000 --cache-hit 0.75
```

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--selftest` | off | Run sanity checks (hand-calculated call costs, probabilities sum to 1, Monte Carlo within 1% of exact) |
| `--verify` | off | Run the report, then assert the 11 headline numbers printed in the paper |
| `--json` | off | Print machine-readable output |
| `--mc-runs N` | 200,000 | Number of Monte Carlo runs for the cross-check |
| `--runs-per-day N` | 8,000 | Assessments per day |
| `--static-tokens N` | 6,000 | Size of the static (cacheable) prompt prefix |
| `--goal-tokens N` | 900 | Size of the dynamic goal and context |
| `--cache-hit X` | 1.0 | Share of the static prefix served from cache (0 to 1) |
| `--no-bill-failed` | off | Do not bill the wasted first attempt of a failed step (the simpler convention; the paper bills it) |

`--verify` checks the paper's default scenario, so it will fail by design if you change an assumption. Use the plain report to explore other scenarios.

## Files

| File | Purpose |
| --- | --- |
| `sar_token_economics.py` | The complete cost model, self-tests and verification |
| `expected_output.txt` | Output captured when the paper was written (self-test followed by `--verify`) |
| `LICENSE` | MIT licence |

To regenerate `expected_output.txt` and confirm that your run matches it:

```bash
{ python3 sar_token_economics.py --selftest; echo; python3 sar_token_economics.py --verify; } > my_output.txt
diff my_output.txt expected_output.txt && echo "Identical"
```

## Method in brief

1. At each step, input equals the whole context window: the static prompt, the goal and all earlier step outputs.
2. Step cost is `(uncached input × input price + cached input × cache-read price + output × output price) / 1,000,000`.
3. With caching, the static prefix is billed at the cache-read price and the remainder at the standard input price.
4. If a step fails validation, the wasted attempt is billed, then the step re-runs (on the premium model if it began on Tier 1) with 40% more output.
5. The expected cost is the probability-weighted sum over all 16 failure outcomes of the four steps.

## Assumptions and limits

- **Prices are 2024-era list prices** for GPT-4o Mini and Claude 3.5 Sonnet, taken from each provider's published pricing. Prices and model generations change often. Update the `PRICES` table and re-run before using any result for budgeting.
- **Scenario inputs are illustrative**, not measured Cospas-Sarsat figures: prompt sizes, output sizes, volume and failure rates should be replaced with values measured in a pilot.
- **Not modelled:** cache-write fees, cache expiry between calls, Tier 3 reasoning-model costs, latency, and integration, hosting and engineering effort.
- Results are modelled outcomes, not guarantees.

## Citation

If you use this code or the paper, please cite it:

```
S. Rout, "Token-Aware Agentic Architecture for Cospas-Sarsat Alert Intelligence:
Predictable Cost, Protected Safety," white paper, Oct. 2026.
Code: https://github.com/<your-username>/<your-repo> (release v1.0)
```

## Licence

Released under the [MIT License](LICENSE).
