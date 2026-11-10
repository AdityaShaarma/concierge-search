"""The boundary between "you still have a step to do" and "something is wrong".

Every `make` target in this project is a stage in a chain, and most of the chain
cannot run until the owner has done something only they can do: accept the Yelp
licence and download the dataset, pick a metro, write the gold queries, grade a
pool. The stages already detect those conditions and raise with a message that
names the remedy — but a message raised as an uncaught exception reaches the
owner as a 20-line traceback, and a traceback reads as *this software is broken*
rather than *you have not run the previous step yet*.

That distinction is worth keeping sharp, because the two cases want opposite
things from the reader:

`PrerequisiteError`
    An expected state. The pipeline is fine; an input does not exist yet. The
    owner should read one sentence, run the step it names, and move on. Printed
    as a plain message with exit code 2 — no traceback, because there is no
    defect to inspect and a stack would only bury the instruction.

everything else
    Unexpected. A bug, a corrupt artifact, or an integrity violation such as a
    train/test leak. Here the traceback is the *point*: the owner needs the file
    and line, and the alarming presentation is proportionate. These propagate
    untouched.

Deliberately exempt: leakage and parity failures (`SplitLeakageError`,
`GoldLeakageError`, `ParityError`, `QrelsMutatedError`). Those mean a number
this project would otherwise report is invalid, and they keep their tracebacks
so they can never be mistaken for a routine "run the previous step" notice.

This module imports nothing from the rest of the package on purpose. The data,
synth, retrieval, rerank and ltr layers all raise prerequisite errors, so
anything it imported would become a cycle.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

__all__ = ["EXIT_PREREQUISITE", "PrerequisiteError", "run_cli"]

# Distinct from 1 so a wrapper script can tell "the owner has a step left" apart
# from "this crashed". `make` surfaces either as a non-zero failure, so nothing
# is lost by being specific.
EXIT_PREREQUISITE = 2


class PrerequisiteError(Exception):
    """An input the owner must supply, or an earlier stage, is missing.

    Subclasses mix this in alongside whatever builtin they already specialise
    (`FileNotFoundError`, `ValueError`, `RuntimeError`), so existing `except`
    clauses that catch those keep working. The mixin only adds the ability to
    ask "is this the owner's turn to act?" without naming every stage's error
    type at every call site.

    The message is the entire user interface for these: it must name the remedy,
    in the form the owner would type it (`make synth-generate`), not merely
    report the absence.
    """


def run_cli(
    main: Callable[[], int],
    *,
    stream: TextIO | None = None,
) -> int:
    """Run a module's `main`, turning a prerequisite into a message.

    Wrapped at the `if __name__ == "__main__"` boundary rather than inside
    `main` itself, so `main` stays a plain callable that raises — tests assert on
    the exception type and its text, which is far more precise than scraping
    stdout for a string.

    Non-prerequisite exceptions are re-raised, not caught: see the module
    docstring for why a traceback is the right output for those.
    """
    try:
        return main()
    except PrerequisiteError as exc:
        print(f"{exc}", file=stream if stream is not None else sys.stderr)
        return EXIT_PREREQUISITE
