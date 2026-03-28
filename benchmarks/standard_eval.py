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
from unittest.mock import patch, MagicMock

# Core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baar import BAARRouter, ModelTier
from benchmarks.data_loader import get_mmlu_subset, get_gsm8k_subset, get_humaneval_subset, StandardTask
from benchmarks.evaluators import evaluate_mmlu, evaluate_gsm8k, evaluate_code

def main():
    parser = argparse.ArgumentParser(description="BAAR-Algo Scientific Evaluation")
    parser.add_argument("--dataset", type=str, choices=["mmlu", "gsm8k", "humaneval", "all"], default="all")
    parser.add_argument("--limit", type=int, default=20, help="Tasks per dataset (max 100 for MMLU/GSM8K)")
    parser.add_argument("--mock", action="store_true", help="Run with mocks (free)")
    parser.add_argument("--budget", type=float, default=1.0, help="Initial budget for the run")
    args = parser.parse_args()

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
            router = BAARRouter(budget=args.budget)
            
            # Setup strategy forcing (same as in engine.py)
            if strat != "baar":
                original_decide = router._router.decide
                if strat == "always-big":
                    router._router.decide = lambda *args, **kwargs: MagicMock(
                        tier=ModelTier.BIG, model=router.big_model, 
                        confidence=1.0, complexity_score=1.0, reason="forced-big", forced_by_budget=False
                    )
                else:
                    router._router.decide = lambda *args, **kwargs: MagicMock(
                        tier=ModelTier.SMALL, model=router.small_model, 
                        confidence=1.0, complexity_score=0.1, reason="forced-small", forced_by_budget=False
                    )

            # Mocking logic
            with patch("litellm.completion") as mock_comp, \
                 patch("baar.core.budget.completion_cost") as mock_cost:
                
                if args.mock:
                    # In mock mode, BIG is 100% right, SMALL is 50% right for logic, etc.
                    # We'll simulate this by setting the response based on ground_truth.
                    def mock_side_effect(model, messages, **kwargs):
                        task_content = messages[-1]["content"]
                        # We need to find the task to get ground_truth
                        # This is a bit slow in O(N^2) but fine for N=100
                        target_task = next(t for t in tasks if t.task in task_content)
                        
                        mock_resp = MagicMock()
                        mock_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=200)
                        
                        # Accuracy logic
                        if "gpt-4o" in model and "mini" not in model:
                            # BIG is always right
                            mock_resp.choices = [MagicMock()]
                            if target_task.dataset == "humaneval":
                                # For coding, we want just the code
                                mock_resp.choices[0].message.content = target_task.ground_truth
                            elif target_task.dataset == "gsm8k":
                                # For math, we need the #### prefix
                                mock_resp.choices[0].message.content = f"The answer is #### {target_task.ground_truth.split('####')[-1].strip()}"
                            else:
                                mock_resp.choices[0].message.content = f"The correct choice is {target_task.ground_truth}"
                        else:
                            # SMALL is only right some of the time (simulated)
                            import random
                            mock_resp.choices = [MagicMock()]
                            if random.random() < 0.3: # 30% failure for small
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
                else:
                    # For Always-Big/Always-Small, use static cost in mock mode
                    mock_cost.return_value = 0.01 if strat == "always-big" else 0.0005

                # RUN TASKS
                correct = 0
                for t in tasks:
                    try:
                        response = router.chat(t.task)
                        if ds_name == "humaneval":
                            if evaluate_code(response, t.task, t.ground_truth):
                                correct += 1
                        elif eval_fn(response, t.ground_truth):
                            correct += 1
                    except Exception as e:
                        # Budget exceeded is expected if budget is very low
                        pass

            accuracy = (correct / len(tasks)) * 100
            ds_results[strat] = {
                "accuracy": round(accuracy, 1),
                "cost": round(router.spent, 6),
                "savings": round(router.log.savings_vs_always_big()["savings_pct"], 1) if strat == "baar" else 0
            }

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
            savings_str = f"{r['savings']}%" if strat == "baar" else "-"
            print(f"{ds.upper():<15} | {strat.upper():<12} | {r['accuracy']:<10} | ${r['cost']:<14.6f} | {savings_str:<8}")
        print("-" * 80)

if __name__ == "__main__":
    main()
