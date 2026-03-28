"""
benchmarks/data_loader.py — Utilities for loading standard datasets from HuggingFace.
Supports MMLU, GSM8K, and HumanEval.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

@dataclass
class StandardTask:
    id: str
    dataset: str
    task: str
    ground_truth: str
    metadata: Dict = field(default_factory=dict)

def get_mmlu_subset(limit: int = 50) -> List[StandardTask]:
    """Load a subset of MMLU (Massive Multitask Language Understanding)."""
    if not load_dataset:
        return []
    
    # Using 'clinical_knowledge' or 'all'
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    tasks = []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        choices = item["choices"]
        question = item["question"]
        options = "\n".join([f"{chr(65+j)}. {c}" for j, c in enumerate(choices)])
        target = chr(65 + item["answer"]) # 'A', 'B', etc.
        
        tasks.append(StandardTask(
            id=f"mmlu_{i}",
            dataset="mmlu",
            task=f"{question}\n\nChoices:\n{options}\n\nRespond with ONLY the letter of the correct choice.",
            ground_truth=target
        ))
    return tasks

def get_gsm8k_subset(limit: int = 50) -> List[StandardTask]:
    """Load a subset of GSM8K (Grade School Math 8K)."""
    if not load_dataset:
        return []
    
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    tasks = []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        tasks.append(StandardTask(
            id=f"gsm8k_{i}",
            dataset="gsm8k",
            task=item["question"] + "\n\nShow your work and end with 'The answer is #### <number>'.",
            ground_truth=item["answer"]
        ))
    return tasks

def get_humaneval_subset(limit: int = 20) -> List[StandardTask]:
    """Load a subset of HumanEval (Coding)."""
    if not load_dataset:
        return []
    
    ds = load_dataset("openai_humaneval", split="test", streaming=True)
    tasks = []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        tasks.append(StandardTask(
            id=f"humaneval_{i}",
            dataset="humaneval",
            task=item["prompt"],
            ground_truth=item["canonical_solution"],
            metadata={"test": item["test"], "entry_point": item["entry_point"]}
        ))
    return tasks
