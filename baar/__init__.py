"""
baar — Budget-Aware Agentic Routing.
Public API for the BAAR-Algo project.
"""

from baar.router import BAARRouter, token_counter
from baar.core.budget import BudgetExceeded, BudgetTracker
from baar.core.router import Router, ModelTier, RoutingDecision
from baar.core.models import StepResult, RoutingLog

__version__ = "0.1.2"
__all__ = [
    "BAARRouter",
    "token_counter",
    "BudgetExceeded",
    "BudgetTracker",
    "Router",
    "ModelTier",
    "RoutingDecision",
    "StepResult",
    "RoutingLog",
]
