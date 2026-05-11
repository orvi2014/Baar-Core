# Baar-Core Launch Brief

## What is this?
Baar-Core is a Python library that enforces hard financial budgets on LLM agents.
It blocks the API call **before** it happens — no DNS lookup, no TCP connection, $0 spent once budget is gone.
Also does semantic routing: cheap model for simple tasks, capable model for complex ones.

GitHub: https://github.com/orvi2014/Baar-Core
Landing: https://baar.robatdasorvi.com
PyPI: baar-core (v0.5.0)

## Current goal
Get 20 upvotes on Product Hunt this week as the first public launch milestone.

## Product Hunt submission (ready to submit)

**Name:** Baar-Core  
**Tagline:** Budget kill-switch for LLM agents — blocks calls, not bills  
**Topics:** Developer Tools · Artificial Intelligence · Open Source · Python · LLM  

**Short description:**
Hard budget enforcement for LLM agents. Estimates token cost locally and raises an exception *before* any API call fires — no DNS, no TCP, $0 spent once the budget is gone. Includes semantic routing (cheap vs. capable model), SQLite-backed per-user quotas, BaarMiddleware for LangGraph, @baar_guard for tool governance, and a 3-line integration.

**Thumbnail** (square logo — upload first):
- /Users/orvi/Baar-Core/ph_assets/00_thumbnail.png

**Screenshots** (upload in order):
- /Users/orvi/Baar-Core/ph_assets/01_hero.png
- /Users/orvi/Baar-Core/ph_assets/02_code.png
- /Users/orvi/Baar-Core/ph_assets/03_architecture.png
- /Users/orvi/Baar-Core/ph_assets/04_saas_budgets.png
- /Users/orvi/Baar-Core/ph_assets/05_middleware_guard.png

**Maker comment (post as first comment after launch):**
Hey PH — builder here.

I shipped an agent to production and spent two weeks quietly terrified. Not of bugs. Of the bill.

There's no seatbelt for LLM agents. Cost tools show you what you spent. Alerts tell you after the threshold. But nothing *stops* the call before it happens.

So I built the circuit breaker I wanted.

Baar-Core trips before the HTTP request leaves your machine. Budget gone — exception raised, call blocked, $0 charged. No DNS lookup. No TCP handshake. Nothing.

It also solves a second problem I had: routing. Why pay for GPT-4o when the user just asked "what's 2+2"? A tiny local model scores complexity 0–1 and picks cheap vs. capable automatically.

What you get:
→ Pre-flight budget enforcement (not tracking — blocking)
→ Semantic routing that picks the right model per task
→ Per-user spending limits, SQLite-backed, survives restarts
→ BaarMiddleware — wraps any LangGraph agent with step cap + budget gate
→ @baar_guard — governs individual tools with call limits and budget thresholds
→ 3 lines of code to wire in

If you've ever shipped an agent and felt that background anxiety about costs — this is for you.

👉 https://baar.robatdasorvi.com
👉 https://github.com/orvi2014/Baar-Core

## When to publish
Best days: Tuesday or Wednesday (highest dev traffic on PH).  
Next window: **Tuesday May 12, 2026** or **Wednesday May 13, 2026**.  
Post at 00:01 PST (08:01 UTC) — PH resets daily at midnight PST, earliest posts get the most exposure.

## Post-launch promotion (same day)
1. HN Show HN post (9–10am PST) — see LAUNCH_HN.md
2. Post in Discord: LangChain #showcase, HuggingFace #projects, LiteLLM community
3. Reply to every PH comment within the first 2 hours

## 1-month star goal
Target: 100 GitHub stars by end of May 2026.
Current: 10 stars as of v0.5.0 launch day.
