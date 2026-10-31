# BAAR-Algo Research & Security Audit

This document maps the **BAAR-Algo** implementation to published research on LLM cost management and resource security, and records where our own benchmarks agree or disagree with it.

## 1. Resilience to "Unbounded Consumption" (OWASP LLM10:2025)

The **OWASP Top 10 for LLM Applications** identifies **Unbounded Consumption** as a critical vulnerability.
- **Threat**: Attackers submit resource-intensive queries (e.g., maximum context) to maximize inference time and cost, depleting the provider's API quota or the user's budget (**Denial of Wallet**).
- **BAAR Solution**: Our **BCD (Budget-Constrained Decoding)** mechanism performs a **Pre-Flight Inspection**. Before any API call is made, the library estimates the token count and compares it against the hard budget.
- **Proof**: `tests/test_resilience.py` confirms that zero network calls are initiated if the estimated cost exceeds the remaining budget.

## 2. Adversarial Scaling in Multi-Model Routing

Research in **Lagrangian Router Optimization** (e.g., RouteLLM) often focuses on "global" average cost. However, BAAR addresses **Local Deterministic Constraints**.
- **The "Gadget" Problem**: Adversaries can prepend neutral tokens to "inflate" query complexity.
- **BAAR Solution**: Our **Semantic Scorer** uses a 5-shot LLM-as-a-judge approach (gpt-4o-mini) which is specifically instructed to ignore token count and focus on logical intent.
- **Verification**: `benchmarks/stress_test.py` validates that "Gibberish Padding" (5000+ characters) does not trick the router into using a BIG model for a simple task.

## 3. Economic Stability & Budget Kill-Switches

Unlike "Platform Routers" (OmniRouter, etc.) that manage aggregate spend, BAAR is a **Local Agent Governance** tool.
- **Budget Integrity**: The `BudgetTracker` is the source of truth. Every successful completion is recorded *atomically* against the local budget store.
- **Transparency**: Every decision is recorded in the `RoutingLog`, providing a full audit trail for financial forensic analysis.

## 4. The output side: reasoning inflation (OverThink)

A pre-flight check that only prices the prompt isn't a hard cap: the response — and a reasoning model's hidden thinking, which is billed as output — can cost far more than any estimate.
- **Threat**: *OverThink* (arXiv 2502.02542) hides benign decoy puzzles in content a reasoning model reads, raising its reasoning tokens up to 46× while the visible answer looks normal. The authors found prompt-injection filters and paraphrasing largely ineffective.
- **BAAR Solution**: `on_output_overrun` bounds the output side. `"preflight"` reserves the worst-case output (capped by the context window, with a 20% prompt-estimate margin) so the budget is never overshot and the response never cut off; `"truncate"` (opt-in) derives `max_tokens` from the remaining budget. Streams record provider-reported usage, so hidden reasoning tokens are counted.
- **Verification**: `benchmarks/overthink_eval.py` (150 tasks, 30% poisoned ×10–46, simulated Claude-class model). Over seeds 1–30, post-hoc alerts overshoot in 30/30 runs and the default `estimate` mode in 27/30 (up to $0.66); `preflight`, `preflight` + `thinking_budget` and `truncate` never overshoot, and `preflight` never cuts an answer off. `tests/test_overthink_eval.py` pins these claims.

## 5. Capping reasoning without cutting the answer (Elastic Reasoning)

- **Research**: *Scalable Chain of Thoughts via Elastic Reasoning* (arXiv 2505.05315, ICLR 2026) splits the budget into a thinking part and a solution part: if thinking runs out, the reasoning block is closed and the answer keeps its own reserved budget.
- **BAAR Solution**: `thinking_budget`. On token-budget providers (Claude, Gemini, DeepSeek) Baar sends a thinking budget and sets `max_tokens` to answer limit + thinking, so `max_output_tokens` is the answer's own limit. On effort-level providers (OpenAI o-series, GPT-5) it maps to `reasoning_effort`, which is advisory.
- **Related**: *Token-Budget-Aware LLM Reasoning* (TALE, arXiv 2412.18547) cuts tokens ~67% by stating a budget in the prompt, but reports "token elasticity": models often ignore small budgets. Baar therefore never relies on a prompt instruction as an enforcement mechanism.

## 6. Ending generation cleanly (s1 budget forcing)

- **Research**: *s1: Simple test-time scaling* (arXiv 2501.19393, EMNLP 2025) ends thinking by inserting an end-of-thinking marker plus "Final Answer:", instead of cutting the model off mid-thought. That requires control over decoding.
- **BAAR Solution**: on hosted APIs the closest equivalent is `on_output_overrun="continue"`: the answer is generated in bounded segments, each reserving only its own cost, and a segment that stops at its limit is continued with the text so far. The answer is cut only when the budget can't pay for another useful segment.

## 7. Predicting output length (conformal reservations)

- **Research**: output-length prediction for scheduling — S3 (Jin et al., 2023), *Efficient LLM Scheduling by Learning to Rank* (arXiv 2408.15792), TRAIL (*Don't Stop Me Now*, arXiv 2410.01035) — and a 2026 uncertainty-aware follow-up (arXiv 2604.00499) arguing that length is set by when the end token is sampled, so it is heavy-tailed and a point estimate is a poor fit.
- **BAAR Solution**: `overrun_risk=r` in `"preflight"` reserves the split-conformal (1 − r) quantile of each model's observed output lengths — index ⌈(n+1)(1−r)⌉−1 of the sorted history — instead of its full max output. Under exchangeability a new response exceeds it with probability ≤ r. `max_tokens` is unchanged, so nothing is cut off. In the OverThink benchmark, `overrun_risk=0.05` overshoots in 2/30 runs (up to $0.16): the accepted risk, in exchange for using more of the budget.

## 8. Budget awareness (BATS)

- **Research**: *Budget-Aware Tool-Use Enables Effective Agent Scaling* (BATS, arXiv 2511.17006) adds a short budget status block to each agent step; agents then keep improving as the budget grows, where plain ReAct plateaus, with no training.
- **BAAR Solution**: `budget_awareness=True` adds a status line (spent, remaining, calls left at the current rate) to the end of each call's last user message — not the system prompt, so a cached prefix stays intact. `WindDown` turns the call past a threshold into a final, wrap-up call; later calls raise `BudgetExhausted` locally with that answer in `.partial_result`.

## 9. Prompt caching — and a negative result

- **Research**: *Don't Break the Cache* (arXiv 2601.06007) reports 41–80% savings from prompt caching on long agent tasks; work on prefix stability (NeurIPS 2026) reports that switching prompts or models mid-loop breaks accumulated prefix reuse and can raise cost.
- **BAAR Solution**: Baar records cached prompt tokens per step, prices streams at the cache-read rate and reports the hit rate. Reservations stay uncached, so caching never weakens the guarantee.
- **Our finding**: `benchmarks/cache_eval.py` (24 conversations × 12 turns, 2k–80k contexts, real gpt-5 / gpt-5-mini pricing, OpenAI-style caching) shows routing still costs 31% less than always using gpt-5 ($2.20 vs $3.20) despite cache losses. A "sticky" rule — keep an easy turn on the big model when its cached prefix makes that turn cheaper — cost *more* overall ($2.34). Each model keeps its own cache and the small model is ~5× cheaper on every line, so rebuilding its cache once pays for itself within a turn or two, which a per-turn rule cannot see. Baar does not ship sticky routing.

## 10. Loop detection

- **Research**: *DeLoop* (arXiv 2610.19376) defines an agent loop as repeating a call that already returned the same thing, and finds that summarizing history roughly doubles looping. Default thresholds start from OpenHands' StuckDetector.
- **BAAR Solution**: `LoopDetector` hashes (tool, normalized arguments) → result, refuses the 3rd identical call before it runs, warns the model on the first repeat (a note appended to the tool's string result), and doesn't count short "running / pending / queued" results, so polling isn't a loop. In noburn's live benchmark (gpt-5-mini, gpt-4.1-mini), warning the model changed its course in 19 of 20 runs, while stopping without the warning did worse than no detection.
- **Verification**: `benchmarks/loop_eval.py` (180 traces, seeds 1–50): with a run budget, 100% of loops caught, 0% of healthy runs stopped, median 2 wasted calls. `scripts/check_loop_parity.py` checks that baar-core and noburn give the same verdicts on a shared scenario set.

## Key References
1. [OWASP Top 10 for LLM Applications (L10: Unbounded Consumption)](https://owasp.org/www-project-top-10-for-llm-applications-2/)
2. [RouteLLM: A Principled Framework for Serving and Evaluating LLM Routers](https://arxiv.org/abs/2406.18665)
3. [Unbounded Resource Consumption in Large Language Models](https://arxiv.org/html/2507.18053v1)
4. [OverThink: Slowdown Attacks on Reasoning LLMs (arXiv 2502.02542)](https://arxiv.org/abs/2502.02542)
5. [Scalable Chain of Thoughts via Elastic Reasoning (arXiv 2505.05315)](https://arxiv.org/abs/2505.05315)
6. [Token-Budget-Aware LLM Reasoning / TALE (arXiv 2412.18547)](https://arxiv.org/abs/2412.18547)
7. [s1: Simple test-time scaling (arXiv 2501.19393)](https://arxiv.org/abs/2501.19393)
8. [Efficient LLM Scheduling by Learning to Rank (arXiv 2408.15792)](https://arxiv.org/abs/2408.15792)
9. [Don't Stop Me Now: Embedding Based Scheduling for LLMs / TRAIL (arXiv 2410.01035)](https://arxiv.org/abs/2410.01035)
10. [Uncertainty-aware output length prediction (arXiv 2604.00499)](https://arxiv.org/abs/2604.00499)
11. [Budget-Aware Tool-Use Enables Effective Agent Scaling / BATS (arXiv 2511.17006)](https://arxiv.org/abs/2511.17006)
12. [Don't Break the Cache: Prompt Caching for Long-Horizon Agentic Tasks (arXiv 2601.06007)](https://arxiv.org/abs/2601.06007)
13. [Prefix Stability for Prompt Caching in LLM Agents (NeurIPS 2026)](https://neurips.cc/virtual/2026/162685)
14. DeLoop (arXiv 2610.19376), as cited in `baar/core/loop.py`
