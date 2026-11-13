"""
baar — Budget-Aware Agentic Routing.
Public API for the BAAR-Algo project.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("baar-core")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.14.0"  # fallback when running from source without install

from baar.router import BAARRouter, BAARConfig, token_counter
from baar.core.budget import BudgetExceeded, BudgetTracker, BudgetWindow, Alert, Reservation, WindowedBudgetStore
from baar.core.exceptions import TaskRejected, BudgetExhausted, PolicyViolation
from baar.core.caps import OutputLimitUnavailable
from baar.core.lengths import OutputLengths
from baar.core.awareness import BudgetStatus, WindDown
from baar.calibration import CalibrationResult, QuantileTable, calibrate_from_failures, ltt_select
from baar.early_stop import EarlyStopPolicy, RunState, RunStopped, fit_early_stop, calibrate_early_stop
from baar.certify import Certificate, NotEnoughRuns, certify_calibrated, certify_early_stop
from baar.monitor import GuaranteeAlarm, GuaranteeMonitor
from baar.runlog import load_runs
from baar.core.router import Router, ModelTier, RoutingDecision
from baar.core.models import StepResult, RoutingLog
from baar.core.policy import Policy, PolicyAction, Rule
from baar.core.loop import LoopDetector, LoopDetected, LoopEvent, loop_nudge
from baar.core.stores import BudgetStore, MemoryBudgetStore, FileBudgetStore, SQLiteBudgetStore

__all__ = [
    "BAARRouter",
    "BAARConfig",
    "token_counter",
    "BudgetExceeded",
    "OutputLimitUnavailable",
    "OutputLengths",
    "BudgetStatus",
    "WindDown",
    "QuantileTable",
    "CalibrationResult",
    "ltt_select",
    "calibrate_from_failures",
    "EarlyStopPolicy",
    "RunState",
    "RunStopped",
    "fit_early_stop",
    "calibrate_early_stop",
    "Certificate",
    "NotEnoughRuns",
    "certify_calibrated",
    "certify_early_stop",
    "GuaranteeAlarm",
    "GuaranteeMonitor",
    "load_runs",
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
    "LoopDetector",
    "LoopDetected",
    "LoopEvent",
    "loop_nudge",
    "BudgetStore",
    "MemoryBudgetStore",
    "FileBudgetStore",
    "SQLiteBudgetStore",
    "RedisBudgetStore",
    "PostgresBudgetStore",
    "Reservation",
]


def __getattr__(name):
    # Stores with optional dependencies are imported on first use.
    if name in ("RedisBudgetStore", "PostgresBudgetStore"):
        from baar.core import stores
        return getattr(stores, name)
    raise AttributeError(f"module 'baar' has no attribute {name!r}")
