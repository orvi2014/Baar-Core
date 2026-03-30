#!/usr/bin/python3
"""
benchmarks/standard_eval.py — The Scientific Validation Suite for BAAR-Algo.
Evaluates accuracy vs cost on MMLU, GSM8K, and HumanEval.
"""
import os
import sys
import argparse
import time
import json
import zlib
import random
import statistics
from unittest.mock import patch, MagicMock

# Core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baar import BAARRouter, ModelTier
from baar.router import token_counter
from benchmarks.data_loader import get_mmlu_subset, get_gsm8k_subset, get_humaneval_subset, StandardTask
from benchmarks.evaluators import evaluate_mmlu, evaluate_gsm8k, evaluate_code


def _is_routing_prompt(task_content: str) -> bool:
    lowered = task_content.lower()
    return (
        "task complexity classifier for an ai routing system" in lowered
        and "task to classify:" in lowered
    )


def _extract_routing_task(task_content: str) -> str:
    marker = "Task to classify:"
    if marker in task_content:
        return task_content.split(marker, 1)[1].strip()
    return task_content


def _mock_router_json(task_text: str) -> str:
    """
    Mock routing-model response: always return valid JSON.
    This keeps BAAR routing behavior meaningful in --mock mode.
    """
    text = task_text.lower()
    complexity = 0.22
    reason = "simple request"

    if any(k in text for k in ["def ", "class ", "import ", "```", "function", "debug", "error", "traceback"]):
        complexity, reason = 0.86, "code content"
    elif any(k in text for k in ["analyze", "compare", "trade-off", "architecture", "evaluate", "reasoning"]):
        complexity, reason = 0.78, "reasoning required"
    elif any(k in text for k in ["####", "show your work"]):
        complexity, reason = 0.74, "structured math"
    elif any(k in text for k in ["choices:", "options:", "choose the correct"]):
        complexity, reason = 0.62, "multi-choice"
    elif len(task_text) > 700:
        complexity, reason = 0.58, "long prompt"

    return json.dumps({"complexity": complexity, "reason": reason})


def _find_target_task(tasks: list[StandardTask], task_content: str) -> StandardTask | None:
    """Best-effort task lookup for benchmark mock responses."""
    exact = next((t for t in tasks if t.task == task_content), None)
    if exact is not None:
        return exact
    contains = next((t for t in tasks if t.task in task_content or task_content in t.task), None)
    return contains


def build_value_fn(policy: str):
    """
    Returns a lightweight value estimator in USD units for BAAR value gating.
    Higher return value = higher task value.
    """
    if policy == "none":
        return None

    def value_fn(task: str) -> float:
        text = task.lower()
        # Keep value in USD-like range to compare against estimate_cost().
        # Baseline targets: SMALL≈0.0005, BIG≈0.01 in mock mode.
        value = 0.0035
        if any(k in text for k in ["fix", "debug", "traceback", "error handling"]):
            value = 0.0120
        elif any(k in text for k in ["write code", "implement", "def ", "class ", "function"]):
            value = 0.0100
        elif any(k in text for k in ["explain", "analyze", "compare", "trade-off", "architecture"]):
            value = 0.0070
        elif len(task) < 20:
            value = 0.0020

        if policy == "strict":
            value *= 0.85

        return max(0.0, value)

    return value_fn


def _percentile(values: list[float], p: float) -> float:
    """Simple percentile helper without numpy dependency."""
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def derive_alpha_from_data(
    tasks: list[StandardTask],
    budget: float,
    value_fn,
    reject_rate_target: float,
    source: str,
    sample_size: int,
    small_exploration_rate: float,
) -> tuple[float, list[dict], list[float]]:
    """
    Calibration pass: estimate value/cost ratio distribution without executing completions.
    Returns derived alpha plus detailed records and ratios.
    """
    calib_router = BAARRouter(
        budget=budget,
        value_fn=value_fn,
        small_exploration_rate=small_exploration_rate,
    )

    sample = tasks[: max(1, min(sample_size, len(tasks)))]
    records: list[dict] = []
    ratios: list[float] = []
    for t in sample:
        decision = calib_router._router.decide(
            task=t.task,
            remaining_budget=calib_router.remaining,
            budget_utilization=calib_router._tracker.utilization,
        )
        model = decision.model
        prompt_tokens = token_counter(t.task, model=model)
        est_cost = calib_router._tracker.estimate_cost(model, prompt_tokens)
        value = float(value_fn(t.task))
        ratio = value / max(est_cost, 1e-12)
        records.append(
            {
                "id": t.id,
                "value": value,
                "cost": est_cost,
                "ratio": ratio,
                "tier": decision.tier.value,
            }
        )
        ratios.append(ratio)

    if source == "median":
        alpha = statistics.median(ratios) * 0.5
    else:
        alpha = _percentile(ratios, reject_rate_target * 100.0)
    return max(0.0, alpha), records, ratios

def main():
    parser = argparse.ArgumentParser(description="BAAR-Algo Scientific Evaluation")
    parser.add_argument("--dataset", type=str, choices=["mmlu", "gsm8k", "humaneval", "all"], default="all")
    parser.add_argument("--limit", type=int, default=20, help="Tasks per dataset (max 100 for MMLU/GSM8K)")
    parser.add_argument("--mock", action="store_true", help="Run with mocks (free)")
    parser.add_argument("--budget", type=float, default=1.0, help="Initial budget for the run")
    parser.add_argument(
        "--complexity-threshold",
        type=float,
        default=0.80,
        help="Default BAAR complexity threshold for non-coding datasets.",
    )
    parser.add_argument(
        "--coding-threshold",
        type=float,
        default=0.75,
        help="BAAR complexity threshold for coding-heavy datasets (HumanEval).",
    )
    parser.add_argument(
        "--value-policy",
        type=str,
        choices=["none", "simple", "strict"],
        default="none",
        help="Enable BAAR value gate with a built-in value_fn policy.",
    )
    parser.add_argument(
        "--value-reject-alpha",
        type=float,
        default=0.3,
        help="Reject only when estimated_value < estimated_cost * alpha.",
    )
    parser.add_argument(
        "--max-reject-rate",
        type=float,
        default=0.5,
        help="Safety clamp for value-gate rejects in BAAR runs.",
    )
    parser.add_argument(
        "--auto-calibrate-alpha",
        action="store_true",
        help="Derive value_reject_alpha from observed value/cost ratios before BAAR run.",
    )
    parser.add_argument(
        "--target-reject-rate",
        type=float,
        default=0.2,
        help="Target reject fraction used when auto-calibrating alpha percentile.",
    )
    parser.add_argument(
        "--alpha-source",
        type=str,
        choices=["percentile", "median"],
        default="percentile",
        help="Alpha derivation rule: percentile(target_reject_rate) or median(ratio)*0.5.",
    )
    parser.add_argument(
        "--calibration-sample",
        type=int,
        default=120,
        help="Number of tasks per dataset used for alpha calibration pass.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible exploration behavior.",
    )
    parser.add_argument(
        "--small-exploration-rate",
        type=float,
        default=0.0,
        help="Probability of downshifting BIG->SMALL during routing exploration (0.0-1.0).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Per-task progress (live) and BAAR routing mix (small vs big) after each baar run",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    print(
        f"⚙️ Config: mock={args.mock}, value_policy={args.value_policy}, "
        f"value_reject_alpha={args.value_reject_alpha}, max_reject_rate={args.max_reject_rate}, "
        f"auto_calibrate_alpha={args.auto_calibrate_alpha}, alpha_source={args.alpha_source}, "
        f"target_reject_rate={args.target_reject_rate}, calibration_sample={args.calibration_sample}, "
        f"small_exploration_rate={args.small_exploration_rate}, "
        f"complexity_threshold={args.complexity_threshold}, "
        f"coding_threshold={args.coding_threshold}, seed={args.seed}"
    )

    datasets = ["mmlu", "gsm8k", "humaneval"] if args.dataset == "all" else [args.dataset]
    
    total_results = {}

    for ds_name in datasets:
        print(f"\n🧪 Testing {ds_name.upper()} dataset (limit={args.limit})...")
        
        # Load tasks
        if ds_name == "mmlu":
            tasks = get_mmlu_subset(limit=args.limit)
            eval_fn = evaluate_mmlu
        elif ds_name == "gsm8k":
            tasks = get_gsm8k_subset(limit=args.limit)
            eval_fn = evaluate_gsm8k
        else:
            tasks = get_humaneval_subset(limit=args.limit)
            eval_fn = evaluate_code
        
        if not tasks:
            print(f"⚠️ No tasks loaded for {ds_name}. Check dependencies.")
            continue

        # Strategies to compare
        strategies = ["always-big", "always-small", "baar"]
        ds_results = {}

        for strat in strategies:
            print(f"  → Running strategy: {strat.upper()}...")
            threshold_for_dataset = (
                args.coding_threshold if ds_name == "humaneval" else args.complexity_threshold
            )
            value_fn = None
            alpha_for_run = args.value_reject_alpha
            if strat == "baar":
                value_fn = build_value_fn(args.value_policy)
                if args.auto_calibrate_alpha and value_fn is not None:
                    alpha_for_run, calib_records, ratios = derive_alpha_from_data(
                        tasks=tasks,
                        budget=args.budget,
                        value_fn=value_fn,
                        reject_rate_target=max(0.0, min(1.0, args.target_reject_rate)),
                        source=args.alpha_source,
                        sample_size=args.calibration_sample,
                        small_exploration_rate=args.small_exploration_rate,
                    )
                    p10 = _percentile(ratios, 10)
                    p50 = _percentile(ratios, 50)
                    p90 = _percentile(ratios, 90)
                    estimated_reject = (
                        sum(1 for r in ratios if r < alpha_for_run) / max(1, len(ratios))
                    )
                    print(
                        "      Alpha calibration:"
                        f" n={len(calib_records)} ratio_p10={p10:.3f}"
                        f" ratio_p50={p50:.3f} ratio_p90={p90:.3f}"
                        f" -> alpha={alpha_for_run:.3f}"
                        f" (est_reject={estimated_reject*100:.1f}%)",
                        flush=True,
                    )
            router = BAARRouter(
                budget=args.budget,
                complexity_threshold=threshold_for_dataset,
                value_fn=value_fn,
                value_reject_alpha=alpha_for_run,
                max_reject_rate=args.max_reject_rate,
                small_exploration_rate=args.small_exploration_rate if strat == "baar" else 0.0,
            )
            
            # Setup strategy forcing (same as in engine.py)
            if strat != "baar":
                original_decide = router._router.decide
                if strat == "always-big":
                    router._router.decide = lambda *args, **kwargs: MagicMock(
                        tier=ModelTier.BIG,
                        model=router.big_model,
                        confidence=1.0,
                        complexity_score=1.0,
                        reason="forced-big",
                        forced_by_budget=False,
                        routing_cache_hit=False,
                    )
                else:
                    router._router.decide = lambda *args, **kwargs: MagicMock(
                        tier=ModelTier.SMALL,
                        model=router.small_model,
                        confidence=1.0,
                        complexity_score=0.1,
                        reason="forced-small",
                        forced_by_budget=False,
                        routing_cache_hit=False,
                    )

            correct = 0
            if args.mock:
                with patch("litellm.completion") as mock_comp, \
                     patch("baar.core.budget.completion_cost") as mock_cost:
                    def mock_side_effect(model, messages, **kwargs):
                        task_content = str(messages[-1]["content"])
                        mock_resp = MagicMock()
                        mock_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=200)
                        mock_resp.choices = [MagicMock()]

                        # Route-scoring calls must return classifier JSON, not final answers.
                        if _is_routing_prompt(task_content):
                            routing_task = _extract_routing_task(task_content)
                            mock_resp.choices[0].message.content = _mock_router_json(routing_task)
                            return mock_resp

                        # For execution calls, map message content to benchmark task.
                        target_task = _find_target_task(tasks, task_content)
                        if target_task is None:
                            # Keep benchmark resilient if content matching misses.
                            mock_resp.choices[0].message.content = "I'm not sure."
                            return mock_resp

                        # Accuracy logic
                        if "gpt-4o" in model and "mini" not in model:
                            # BIG is always right
                            if target_task.dataset == "humaneval":
                                # For coding, we want just the code
                                mock_resp.choices[0].message.content = target_task.ground_truth
                            elif target_task.dataset == "gsm8k":
                                # For math, we need the #### prefix
                                mock_resp.choices[0].message.content = f"The answer is #### {target_task.ground_truth.split('####')[-1].strip()}"
                            else:
                                mock_resp.choices[0].message.content = f"The correct choice is {target_task.ground_truth}"
                        else:
                            # SMALL is only right some of the time (simulated).
                            # Pessimistic stub — not calibrated to real gpt-4o-mini accuracy.
                            # Deterministic ~30% "wrong" per task id so tables are reproducible.
                            if (zlib.adler32(target_task.id.encode()) % 100) < 30:
                                mock_resp.choices[0].message.content = "I'm not sure."
                            else:
                                if target_task.dataset == "humaneval":
                                    mock_resp.choices[0].message.content = target_task.ground_truth
                                elif target_task.dataset == "gsm8k":
                                    mock_resp.choices[0].message.content = f"#### {target_task.ground_truth.split('####')[-1].strip()}"
                                else:
                                    mock_resp.choices[0].message.content = f"Choice {target_task.ground_truth}"
                        return mock_resp

                    mock_comp.side_effect = mock_side_effect

                    def mock_cost_side_effect(completion_response):
                        # Use actual decision tier from last log entry
                        if router.log.steps and router.log.steps[-1].decision.tier == ModelTier.BIG:
                            return 0.01
                        return 0.0005

                    mock_cost.side_effect = mock_cost_side_effect

                    for i, t in enumerate(tasks):
                        try:
                            response = router.chat(t.task)
                            if ds_name == "humaneval":
                                if evaluate_code(response, t.task, t.ground_truth):
                                    correct += 1
                            elif eval_fn(response, t.ground_truth):
                                correct += 1
                        except Exception:
                            # Budget exceeded is expected if budget is very low
                            pass
                        if args.verbose and (i + 1) % max(1, len(tasks) // 10 or 1) == 0:
                            print(f"      … {i + 1}/{len(tasks)} tasks", flush=True)
            else:
                for i, t in enumerate(tasks):
                    try:
                        response = router.chat(t.task)
                        if ds_name == "humaneval":
                            if evaluate_code(response, t.task, t.ground_truth):
                                correct += 1
                        elif eval_fn(response, t.ground_truth):
                            correct += 1
                    except Exception:
                        # API/key/provider errors are counted as incorrect in live mode
                        pass
                    if args.verbose and (i + 1) % max(1, len(tasks) // 10 or 1) == 0:
                        print(f"      … {i + 1}/{len(tasks)} tasks", flush=True)

            accuracy = (correct / len(tasks)) * 100
            ds_results[strat] = {
                "accuracy": round(accuracy, 1),
                "cost": round(router.spent, 6),
                "savings": round(router.log.savings_vs_always_big()["savings_pct"], 1) if strat == "baar" else 0,
                "rejects": router.log.reject_steps if strat == "baar" else 0,
                "reject_rate": round((router.log.reject_steps / max(1, len(tasks))) * 100, 1) if strat == "baar" else 0,
            }

            if strat == "baar" and router.log.steps:
                n = len(router.log.steps)
                small_n = router.log.small_calls
                big_n = router.log.big_calls
                rej_n = router.log.reject_steps
                cache_hits = router.log.routing_cache_hits
                print(
                    f"      BAAR routing: small={small_n}, big={big_n}, reject={rej_n}, "
                    f"route-cache_hits={cache_hits} (over {n} logged steps)",
                    flush=True,
                )
                if args.verbose and n <= 50:
                    for s in router.log.steps:
                        d = s.decision
                        print(
                            f"        step {s.step_num}: tier={d.tier.value} "
                            f"complexity={d.complexity_score:.3f} "
                            f"route_cache_hit={getattr(d, 'routing_cache_hit', False)}",
                            flush=True,
                        )

        total_results[ds_name] = ds_results

    # Print Report
    print("\n" + "="*80)
    print("🔬 BAAR-ALGO SCIENTIFIC VALIDATION REPORT (Subsets)")
    print("="*80)
    print(f"{'Dataset':<15} | {'Strategy':<12} | {'Accuracy %':<10} | {'Total Cost USD':<15} | {'Savings':<8}")
    print("-" * 80)
    
    for ds, res in total_results.items():
        for strat in ["always-big", "always-small", "baar"]:
            r = res[strat]
            if strat == "baar":
                savings_str = f"{r['savings']}% (rej {r['reject_rate']}%)"
            else:
                savings_str = "-"
            print(f"{ds.upper():<15} | {strat.upper():<12} | {r['accuracy']:<10} | ${r['cost']:<14.6f} | {savings_str:<8}")
        print("-" * 80)

if __name__ == "__main__":
    main()
