"""
baar/__main__.py — CLI entry point for 'python -m baar'.
"""
import sys
from baar import __version__

def main():
    print(f"BAAR-Algo (baar-core) v{__version__}")
    print("\nBudget-Aware Agentic Routing — Intelligent LLM Model Selection.")
    print("\nUsage:")
    print("  baar-bench   : Run the scientific validation suite")
    print("  baar-stress  : Run the adversarial resilience suite")
    print("  baar-telemetry <file.jsonl> : Summarize telemetry (reject/failover/spend)")
    print("\nSee README.md for more information.")

if __name__ == "__main__":
    main()
