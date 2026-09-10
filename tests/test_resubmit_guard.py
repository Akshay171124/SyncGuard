"""Tests for the SLURM resubmit crash-loop guard."""

import subprocess
import tempfile
from pathlib import Path

GUARD = Path(__file__).resolve().parents[1] / "scripts" / "lib" / "resubmit_guard.sh"


def _run(script: str, cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f'source "{GUARD}"\n{script}'],
        cwd=cwd, capture_output=True, text=True,
    )


def test_starts_at_zero():
    with tempfile.TemporaryDirectory() as d:
        r = _run("guard_count job", d)
        assert r.stdout.strip() == "0"


def test_failures_accumulate():
    with tempfile.TemporaryDirectory() as d:
        r = _run("guard_record_failure job; guard_record_failure job; guard_count job", d)
        assert r.stdout.strip() == "2"


def test_reset_clears_count():
    with tempfile.TemporaryDirectory() as d:
        r = _run("guard_record_failure job; guard_reset job; guard_count job", d)
        assert r.stdout.strip() == "0"


def test_may_resubmit_below_limit():
    with tempfile.TemporaryDirectory() as d:
        r = _run("guard_record_failure job; guard_may_resubmit job 3", d)
        assert r.returncode == 0


def test_may_not_resubmit_at_limit():
    with tempfile.TemporaryDirectory() as d:
        r = _run(
            "guard_record_failure job; guard_record_failure job; "
            "guard_record_failure job; guard_may_resubmit job 3", d)
        assert r.returncode == 1


def test_counts_are_per_job_name():
    with tempfile.TemporaryDirectory() as d:
        r = _run("guard_record_failure a; guard_record_failure a; guard_count b", d)
        assert r.stdout.strip() == "0"
