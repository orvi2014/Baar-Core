"""
examples/fastapi_per_user_budget.py

SaaS pattern: each user gets a $0.10 monthly quota enforced at the library
level — no proxy, no middleware, no external service required.

Budget persists across restarts via SQLiteBudgetStore. When a user exhausts
their quota, Baar-Core raises BudgetExhausted *before* any API call is made.

Run:
    pip install baar-core fastapi uvicorn
    export OPENAI_API_KEY=sk-...
    uvicorn examples.fastapi_per_user_budget:app --reload

Try it:
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"user_id": "alice", "message": "What is the capital of France?"}'
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from baar import BAARRouter, BudgetExhausted
from baar.core.stores import SQLiteBudgetStore

DB_PATH = Path(os.getenv("BAAR_DB_PATH", "user_budgets.db"))
MONTHLY_BUDGET_USD = float(os.getenv("BAAR_BUDGET_USD", "0.10"))

app = FastAPI(title="Budget-safe LLM API")


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    spent_usd: float
    remaining_usd: float


def _router_for(user_id: str) -> BAARRouter:
    store = SQLiteBudgetStore(str(DB_PATH), namespace=user_id)
    return BAARRouter(
        budget=MONTHLY_BUDGET_USD,
        store=store,
        small_model="gpt-4o-mini",
        big_model="gpt-4o",
        use_llm_router=True,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    router = _router_for(req.user_id)
    try:
        reply = await router.achat(req.message)
    except BudgetExhausted as e:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "monthly_budget_exhausted",
                "remaining_usd": round(e.remaining, 6),
                "message": f"Your ${MONTHLY_BUDGET_USD:.2f} monthly quota is exhausted. "
                           "Resets at the start of next month.",
            },
        )
    return ChatResponse(
        reply=reply,
        spent_usd=round(router.spent, 6),
        remaining_usd=round(router.remaining, 6),
    )


@app.get("/quota/{user_id}")
async def get_quota(user_id: str):
    router = _router_for(user_id)
    return {
        "user_id": user_id,
        "budget_usd": MONTHLY_BUDGET_USD,
        "spent_usd": round(router.spent, 6),
        "remaining_usd": round(router.remaining, 6),
        "utilization_pct": round(router._tracker.utilization * 100, 1),
    }


@app.post("/quota/{user_id}/reset")
async def reset_quota(user_id: str):
    store = SQLiteBudgetStore(str(DB_PATH), namespace=user_id)
    store.reset()
    return {"user_id": user_id, "status": "reset", "budget_usd": MONTHLY_BUDGET_USD}
