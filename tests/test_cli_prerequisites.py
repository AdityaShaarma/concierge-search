"""Every `make` target must fail with a sentence, not a stack trace.

The chain of stages in this project is long, and the owner walks it from empty:
no dataset, no metro chosen, no gold queries, no artifacts. Almost every target
they run early will stop. What it prints when it stops is therefore a real part
of the interface, not an edge case — and the failure mode these tests exist to
prevent is subtle, because nothing is *broken* when it regresses. Each stage
already raises with a message naming the remedy; the regression is that the
message arrives wrapped in a traceback, which reads as a crash and sends the
owner debugging their install instead of running the step the sentence names.

Two levels, deliberately:

* unit tests on `run_cli`, which pin the contract (prerequisite → message and
  exit 2; anything else → propagate); and
* one subprocess test per entrypoint, which pins the thing the owner actually
  sees. This second kind is what catches a *new* stage added without the
  wrapper — the exact way this regressed in the first place.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from concierge.cli import EXIT_PREREQUISITE, PrerequisiteError, run_cli
from concierge.config import REPO_ROOT, MetroNotSetError, load_config
from concierge.data.load import MissingDatasetError
from concierge.synth.generate import GoldSetMissingError, SynthArtifactMissingError
from concierge.synth.splits import (
    SplitManifestCorruptError,
    SplitManifestMissingError,
)

# ─────────────────────────────────────────────────────────────────────────────
# run_cli's contract
# ─────────────────────────────────────────────────────────────────────────────


def test_prerequisite_is_reported_not_raised(capsys: pytest.CaptureFixture[str]) -> None:
    def main() -> int:
        raise PrerequisiteError("run `make synth-generate` first")

    assert run_cli(main) == EXIT_PREREQUISITE
    assert "run `make synth-generate` first" in capsys.readouterr().err


def test_message_is_printed_without_a_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    """The whole point: no "Traceback" and no exception class name in the output."""

    def main() -> int:
        raise MissingDatasetError("data/raw is empty. Download the dataset.")

    run_cli(main)
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "MissingDatasetError" not in err
    assert "Download the dataset." in err


def test_other_exceptions_keep_their_traceback() -> None:
    """A genuine defect must not be flattened into a polite sentence."""

    def main() -> int:
        raise KeyError("business_id")

    with pytest.raises(KeyError):
        run_cli(main)


def test_a_corrupt_manifest_is_not_a_prerequisite() -> None:
    """Damaged artifact ≠ unrun step.

    The owner can fix "no manifest yet" by running a step. They cannot fix a
    manifest that fails its own checksum that way, so it keeps the traceback
    that tells them which file to look at.
    """

    def main() -> int:
        raise SplitManifestCorruptError("checksum mismatch")

    with pytest.raises(SplitManifestCorruptError):
        run_cli(main)


def test_exit_code_is_distinct_from_one() -> None:
    """So a wrapper can tell "a step is left" apart from "this crashed"."""
    assert EXIT_PREREQUISITE == 2


def test_successful_main_returns_its_own_code() -> None:
    assert run_cli(lambda: 0) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Which error types are the owner's turn to act
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exc_type",
    [
        MissingDatasetError,
        MetroNotSetError,
        GoldSetMissingError,
        SynthArtifactMissingError,
        SplitManifestMissingError,
    ],
)
def test_owner_actionable_errors_are_prerequisites(exc_type: type[Exception]) -> None:
    assert issubclass(exc_type, PrerequisiteError)


@pytest.mark.parametrize("exc_type", [SplitManifestCorruptError])
def test_integrity_errors_are_not_prerequisites(exc_type: type[Exception]) -> None:
    assert not issubclass(exc_type, PrerequisiteError)


def test_leakage_errors_keep_their_tracebacks() -> None:
    """Imported inside the test: these modules pull in torch and lightgbm.

    A leak means a number this project would report is invalid. It must never be
    presentable as a routine "run the previous step" notice, so it is checked
    explicitly rather than left to the absence of a mixin.
    """
    from concierge.ltr.train import LeakageError
    from concierge.rerank.train import GoldLeakageError
    from concierge.synth.judge import QrelsMutatedError
    from concierge.synth.splits import SplitLeakageError

    for exc_type in (SplitLeakageError, LeakageError, GoldLeakageError, QrelsMutatedError):
        assert not issubclass(exc_type, PrerequisiteError), exc_type.__name__


def test_metro_error_is_still_a_value_error() -> None:
    """`concierge.data.build` catches `ValueError`; adding the mixin kept that."""
    assert issubclass(MetroNotSetError, ValueError)
    with pytest.raises(ValueError):
        load_config("configs/base.yaml").require_metro()


# ─────────────────────────────────────────────────────────────────────────────
# What the owner actually sees
# ─────────────────────────────────────────────────────────────────────────────

# Every module reachable from a `make` target that stops on a prerequisite when
# the repo is fresh. Kept as an explicit list so adding a stage without wiring
# `run_cli` fails here rather than in the owner's terminal.
PREREQUISITE_ENTRYPOINTS = [
    "concierge.data.explore",
    "concierge.data.build",
    "concierge.retrieval.build_index",
    "concierge.synth.generate",
    "concierge.synth.negatives",
    "concierge.synth.judge",
    "concierge.synth.splits",
    "concierge.rerank.train",
    "concierge.ltr.build_examples",
    "concierge.ltr.train",
    "eval.build_runs",
    "eval.label_cli",
    "eval.run_eval",
]


@pytest.mark.slow
@pytest.mark.parametrize("module", PREREQUISITE_ENTRYPOINTS)
def test_entrypoint_stops_cleanly_on_a_fresh_repo(module: str) -> None:
    """Run the module as the owner would and assert on its real output.

    Marked slow because each case is a fresh interpreter, and several import
    torch or lightgbm. It earns the seconds: this is the only check that
    exercises the `if __name__ == "__main__"` guard, which is where the wrapper
    lives and which no in-process test can reach.

    This test assumes a repo with no dataset and no built artifacts — true in CI
    and on a fresh clone. If the owner has built the pipeline it will not apply,
    hence the tolerance for exit 0 below rather than a hard requirement of 2.
    """
    proc = subprocess.run(
        [sys.executable, "-m", module, "--config", "configs/base.yaml"],
        capture_output=True,
        text=True,
        timeout=600,
        # Closed, not inherited: `eval.label_cli` is interactive, and a stage
        # that reached a prompt instead of its prerequisite check would hang the
        # suite rather than fail it. With no stdin it errors immediately, which
        # is a result this test can report.
        stdin=subprocess.DEVNULL,
    )
    combined = proc.stdout + proc.stderr

    if proc.returncode == 0:
        pytest.skip(f"{module} succeeded — artifacts already built in this checkout")

    assert proc.returncode == EXIT_PREREQUISITE, (
        f"{module} exited {proc.returncode}, expected {EXIT_PREREQUISITE}.\n{combined}"
    )
    assert "Traceback" not in combined, (
        f"{module} printed a traceback instead of a message:\n{combined}"
    )
    # A message that only reports the absence leaves the owner stuck; it has to
    # name what to do next. The accepted markers are the four shapes a remedy
    # actually takes in this project: run a make target, edit the config, write
    # the gold file, or download the dataset. Checked loosely on purpose —
    # pinning exact wording would make every message edit a test failure, and
    # the property worth defending is "it says what to do", not the phrasing.
    #
    # Matching is done on whitespace-collapsed text: these messages are printed
    # through `rich`, which hard-wraps to the terminal width, so a path or a
    # phrase can arrive split across two lines.
    flat = " ".join(combined.split())
    remedies = (
        "make ",
        "configs/base.yaml",
        "Download the Yelp Open Dataset",
        "gold_queries.csv",
        "eval/README.md",
        # `rich` hard-wraps to the terminal width and will break a path
        # mid-token, so "gold_queries.csv" can arrive as "gold_queries.cs v"
        # even after whitespace is collapsed. The surrounding instruction is
        # what the owner reads anyway.
        "written by you, by hand",
    )
    assert any(hint in flat for hint in remedies), (
        f"{module} stopped without naming a remedy:\n{combined}"
    )


@pytest.mark.slow
def test_the_load_tester_stops_cleanly_on_a_missing_workload() -> None:
    """`serve.load_test` has its own case because it takes no `--config`.

    That is exactly why it was missed when the rest of the entry points were
    converted: the sweep above walks the `make` targets, every one of which is
    invoked as `-m <module> --config configs/base.yaml`, and the load tester
    takes `--queries` instead. It is still a program the owner runs by hand —
    the measured p50/p95 in the README comes out of it — and a missing workload
    file is an input they supply, not a defect, so it must read as a sentence.

    No server is needed: the workload is read before the first request, which
    this test also pins by pointing `--url` at a port nothing is listening on.
    A traceback here would mean the file is being opened after the run starts.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "serve.load_test",
            "--url",
            "http://127.0.0.1:9",
            "--queries",
            "eval/this-file-does-not-exist.csv",
            "--requests",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=300,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == EXIT_PREREQUISITE, (
        f"expected exit 2, got {proc.returncode}:\n{combined}"
    )
    assert "Traceback" not in combined, combined
    flat = " ".join(combined.split())
    assert "--queries" in flat, f"the message must name the flag to fix:\n{combined}"
