"""
tests/test_loop_parity.py — baar-core gives the shared loop-detection verdicts.

The scenarios in tests/fixtures/loop_parity.json are the contract between
baar-core and noburn's SDK; scripts/check_loop_parity.py --other <loop.py>
compares the two. Here CI checks baar-core's half, and that the checker catches drift.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import check_loop_parity  # noqa: E402
from baar.core import loop as baar_loop  # noqa: E402

SCENARIOS = check_loop_parity.load_scenarios()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_baar_core_matches_the_shared_verdict(scenario):
    assert check_loop_parity.verdict(baar_loop, scenario) == scenario["expect"]


def test_checker_flags_a_drifted_implementation(tmp_path):
    source = open(baar_loop.__file__, encoding="utf-8").read()
    drifted = source.replace("max_repeats: int = 2,", "max_repeats: int = 4,")
    assert drifted != source
    path = tmp_path / "drifted_loop.py"
    path.write_text(drifted, encoding="utf-8")
    module = check_loop_parity.load_module(str(path), "_drifted_loop")
    problems = check_loop_parity.check({"drifted": module}, SCENARIOS)
    assert any("third identical call is refused" in p for p in problems)


def test_checker_reports_unsupported_options_as_mismatches(tmp_path):
    path = tmp_path / "old_loop.py"
    path.write_text(
        "class LoopDetector:\n"
        "    def __init__(self, **kw):\n"
        "        if 'in_progress' in kw: raise TypeError('unexpected keyword argument in_progress')\n"
        "    def before(self, *a): pass\n"
        "    def after(self, *a, **k): return None\n"
        "    def peek(self, *a): return None\n",
        encoding="utf-8",
    )
    module = check_loop_parity.load_module(str(path), "_old_loop")
    problems = check_loop_parity.check({"old": module}, SCENARIOS)
    assert any("TypeError" in p for p in problems)


def test_cli(capsys):
    assert check_loop_parity.main([]) == 0
    assert "all match" in capsys.readouterr().out
    assert check_loop_parity.main(["--other", baar_loop.__file__]) == 0
