"""
baar — Budget-Aware Agentic Routing.
Public API for the BAAR-Algo project.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("baar-core")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.3.0"  # fallback when running from source without install

from baar.router import BAARRouter, BAARConfig, token_counter
from baar.core.budget import BudgetExceeded, BudgetTracker
from baar.core.exceptions import TaskRejected, BudgetExhausted
from baar.core.router import Router, ModelTier, RoutingDecision
from baar.core.models import StepResult, RoutingLog
from baar.core.stores import BudgetStore, MemoryBudgetStore, FileBudgetStore, SQLiteBudgetStore

__all__ = [
    "BAARRouter",
    "BAARConfig",
    "token_counter",
    "BudgetExceeded",
    "BudgetTracker",
    "TaskRejected",
    "BudgetExhausted",
    "Router",
    "ModelTier",
    "RoutingDecision",
    "StepResult",
    "RoutingLog",
    "BudgetStore",
    "MemoryBudgetStore",
    "FileBudgetStore",
    "SQLiteBudgetStore",
]
