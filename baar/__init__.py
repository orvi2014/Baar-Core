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
from baar.core.budget import BudgetExceeded, BudgetTracker, BudgetWindow, Alert, WindowedBudgetStore
from baar.core.exceptions import TaskRejected, BudgetExhausted, PolicyViolation
from baar.core.router import Router, ModelTier, RoutingDecision
from baar.core.models import StepResult, RoutingLog
from baar.core.policy import Policy, PolicyAction, Rule
from baar.core.stores import BudgetStore, MemoryBudgetStore, FileBudgetStore, SQLiteBudgetStore

__all__ = [
    "BAARRouter",
    "BAARConfig",
    "token_counter",
    "BudgetExceeded",
    "BudgetTracker",
    "BudgetWindow",
    "Alert",
    "WindowedBudgetStore",
    "TaskRejected",
    "BudgetExhausted",
    "PolicyViolation",
    "Router",
    "ModelTier",
    "RoutingDecision",
    "StepResult",
    "RoutingLog",
    "Policy",
    "PolicyAction",
    "Rule",
    "BudgetStore",
    "MemoryBudgetStore",
    "FileBudgetStore",
    "SQLiteBudgetStore",
]
