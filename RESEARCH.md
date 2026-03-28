# BAAR-Algo Research & Security Audit

This document maps the **BAAR-Algo** implementation to established industry research on LLM cost management and resource security.

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

## Key References
1. [OWASP Top 10 for LLM Applications (L10: Unbounded Consumption)](https://owasp.org/www-project-top-10-for-llm-applications-2/)
2. [RouteLLM: A Principled Framework for Serving and Evaluating LLM Routers](https://arxiv.org/abs/2406.18665)
3. [Unbounded Resource Consumption in Large Language Models](https://arxiv.org/html/2507.18053v1)
