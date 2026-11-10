"""Tests for the shared contracts themselves.

These are deliberately boring. They exist because every other module depends on
these types behaving exactly as documented, so a silent change here would
corrupt results everywhere downstream.
"""

from __future__ import annotations

import pytest

from concierge.config import load_config
from concierge.types import (
    SCORE_BM25,
    Candidate,
    ConstraintMatch,
    ConstraintStatus,
    HoursInterval,
    ParsedQuery,
)


class TestHoursInterval:
    def test_normal_interval_covers_inside_only(self):
        # 7:00 - 20:00
        interval = HoursInterval(start_min=7 * 60, end_min=20 * 60)
        assert not interval.crosses_midnight
        assert interval.covers(7 * 60)  # inclusive start
        assert interval.covers(19 * 60 + 59)
        assert not interval.covers(20 * 60)  # exclusive end
        assert not interval.covers(6 * 60 + 59)

    def test_overnight_interval_wraps(self):
        # 17:00 - 02:00 the next day. The whole point of the three-valued logic.
        interval = HoursInterval(start_min=17 * 60, end_min=2 * 60)
        assert interval.crosses_midnight
        assert interval.covers(23 * 60)  # 11pm, open
        assert interval.covers(1 * 60)  # 1am, still open
        assert not interval.covers(3 * 60)  # 3am, closed
        assert not interval.covers(12 * 60)  # noon, closed

    def test_interval_ending_at_midnight(self):
        # "17:0-0:0" — end_min 0 is <= start_min, so it is treated as wrapping
        # to midnight exactly. 11:59pm open, 00:00 closed.
        interval = HoursInterval(start_min=17 * 60, end_min=0)
        assert interval.covers(23 * 60 + 59)
        assert not interval.covers(0)


class TestCandidate:
    def test_with_score_does_not_mutate_original(self):
        original = Candidate(business_id="b001")
        updated = original.with_score(SCORE_BM25, 12.5)
        assert original.scores == {}, "stages must not mutate each other's candidates"
        assert updated.scores == {SCORE_BM25: 12.5}
        assert updated.business_id == "b001"


class TestConstraintMatch:
    def test_unknown_never_reads_as_pass_or_fail(self):
        match = ConstraintMatch("NoiseLevel", "quiet", None, ConstraintStatus.UNKNOWN)
        assert "no data" in match.explanation
        assert "wanted" not in match.explanation


class TestParsedQuery:
    def test_counts_every_hard_constraint_kind(self):
        from concierge.types import OpenAt

        q = ParsedQuery(
            raw="quiet cafe with wifi open past 10pm, cheap",
            semantic_text="quiet cafe",
            open_at=OpenAt(day="Friday", minute_of_day=22 * 60),
            attributes={"WiFi": "free", "NoiseLevel": "quiet"},
            price_max=2,
        )
        assert q.n_hard_constraints == 4  # open_at + 2 attributes + price

    def test_unconstrained_query_counts_zero(self):
        assert ParsedQuery(raw="good pho", semantic_text="good pho").n_hard_constraints == 0


class TestConfig:
    def test_base_config_loads_and_validates(self):
        cfg = load_config("configs/base.yaml")
        assert cfg.seed == 42
        assert cfg.retrieval.fusion.k == 60

    def test_metro_is_unset_and_fails_loudly(self):
        cfg = load_config("configs/base.yaml")
        assert cfg.data.metro is None, "committed config must not pre-decide the metro"
        with pytest.raises(ValueError, match="make explore"):
            cfg.require_metro()

    def test_unknown_key_is_rejected(self):
        """A typo'd key must fail, not be silently ignored — a config that
        accepts `retreive: 500` would run the whole cascade at the default and
        the ablation row would be mislabelled.

        Asserting the specific pydantic error rather than bare `Exception`: a
        bare assertion also passes when `Config` fails to import at all.
        """
        from pydantic import ValidationError

        from concierge.config import Config

        with pytest.raises(ValidationError, match="extra_forbidden"):
            Config.model_validate({"seed": 1, "nonsense_key": True})

    def test_cascade_must_narrow(self):
        from concierge.config import CandidatesConfig

        with pytest.raises(ValueError, match="shrink"):
            CandidatesConfig(retrieve=10, rerank=50, final=10)

    def test_fingerprint_is_stable_and_sensitive(self):
        cfg = load_config("configs/base.yaml")
        assert cfg.fingerprint() == cfg.fingerprint()
        other = cfg.model_copy(update={"seed": 43})
        assert cfg.fingerprint() != other.fingerprint()


# ─────────────────────────────────────────────────────────────────────────────
# Package layering
# ─────────────────────────────────────────────────────────────────────────────


class TestPackageLayering:
    """`concierge.synth` may depend on `concierge.rerank`, but not the reverse
    at module scope.

    This is the seam that produced a real near-miss. `concierge.synth.negatives`
    has to mine hard negatives against the same query text the cross-encoder is
    trained on, so it imports that helper; the helper was defined in
    `concierge.rerank.train`, which imports `concierge.synth.splits`. The two
    packages therefore referenced each other, and it resolved only because
    `concierge/synth/__init__.py` is empty and `splits` imports `negatives`
    inside functions rather than at module scope. Any future module-scope
    import on that last edge would have turned a working tree into an
    ImportError, and the helper's two definitions-worth of query-text logic
    would have been the thing tempting someone to re-derive it locally instead.

    The helper now lives in `concierge.rerank.model`, beside `query_text_for`.
    These tests pin the two properties that keeps: the module that owns query
    text depends on neither package, and every module on the former cycle still
    imports standalone in a fresh interpreter.
    """

    def test_the_module_owning_query_text_does_not_import_synth_or_the_trainer(self):
        """`concierge.rerank.model` is imported by the serving reranker, the
        trainer, the ONNX export and the mining stage. It must stay at the
        bottom of the layering, or the cycle comes back by another route."""
        import ast
        from pathlib import Path

        source = Path("concierge/rerank/model.py").read_text()
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            # Module scope only: a deferred import inside a function cannot
            # create an import-time cycle, which is the thing under test.
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)

        offenders = {
            name
            for name in imported
            if name.startswith("concierge.synth") or name.startswith("concierge.rerank.train")
        }
        assert not offenders, (
            f"concierge.rerank.model imports {sorted(offenders)} at module scope, "
            "which re-creates the synth <-> rerank cycle it was moved here to break"
        )

    def test_training_query_text_has_exactly_one_definition(self):
        """Re-exported through the trainer for readability, but the same object.

        Two *definitions* is the failure this guards against: mining and
        training would then agree only by coincidence, which is the skew the
        helper exists to remove.
        """
        from concierge.rerank import model as rerank_model
        from concierge.rerank import train as rerank_train
        from concierge.synth import negatives

        assert rerank_train.training_query_text is rerank_model.training_query_text
        assert negatives.training_query_text is rerank_model.training_query_text

    @pytest.mark.parametrize(
        "module",
        [
            "concierge.rerank.model",
            "concierge.rerank.train",
            "concierge.synth.negatives",
            "concierge.synth.splits",
        ],
    )
    def test_each_module_on_the_former_cycle_imports_standalone(self, module: str):
        """In a *fresh* interpreter, one at a time.

        Importing them inside this process proves little: by the time this test
        runs, the suite has already imported most of the package, so a genuine
        cycle would be masked by whatever happened to be in `sys.modules`
        first. A subprocess is the only way to assert the property that matters
        — "this module is importable on its own".
        """
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"import {module} failed:\n{result.stderr}"


# ─────────────────────────────────────────────────────────────────────────────
# DECISIONS.md is a document the code cites by id, so the ids are a contract
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionIds:
    """DECISIONS.md is the owner's interview study document, and source files tell
    a reader to look things up in it by id (`See DECISIONS.md D228`). That makes
    the ids an interface, and an interface with duplicates is broken: two
    different decisions were both numbered D225 and two both D226, so half the
    citations to those ids resolved to the wrong entry — a reader following
    `DECISIONS.md D225` from `ltr/train.py` landed on an argument about category
    policy exemptions instead of the constraint-feature ablation it was about.

    The file is ~3,000 lines and appended to by hand, so this is checked rather
    than remembered.
    """

    @staticmethod
    def _headings() -> list[str]:
        import re
        from pathlib import Path

        from concierge.config import REPO_ROOT

        text = Path(REPO_ROOT, "DECISIONS.md").read_text(encoding="utf-8")
        return re.findall(r"^#{2,3} (D\d{3})", text, flags=re.MULTILINE)

    def test_every_decision_id_is_used_once(self):
        ids = self._headings()
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        assert not duplicates, (
            f"duplicate decision id(s) in DECISIONS.md: {duplicates}. Every citation "
            "to a duplicated id is ambiguous; renumber the later entry and update the "
            "citing sites."
        )

    def test_there_are_decisions_to_check(self):
        # A regex that stopped matching would make the test above pass forever.
        assert len(self._headings()) > 100

    def test_no_title_carries_an_unreplaced_placeholder(self):
        import re
        from pathlib import Path

        from concierge.config import REPO_ROOT

        text = Path(REPO_ROOT, "DECISIONS.md").read_text(encoding="utf-8")
        titles = re.findall(r"^#{2,3} D\d{3}.*$", text, flags=re.MULTILINE)
        # "D0xx" was a template placeholder; "D-<module>-N" was an abandoned
        # per-module numbering scheme. Ten titles carried the first and twelve
        # the second, in the Stage 4 and dense-retrieval sections — the two an
        # interviewer is most likely to be reading. Visible scaffolding in the
        # study document undercuts the care the rest of the file shows.
        bad = [t for t in titles if "D0xx" in t or re.search(r"— D-[a-z]+-\d+ —", t)]
        assert not bad, f"unreplaced id placeholder(s) in DECISIONS.md titles: {bad[:5]}"

    #: Files that may cite a decision id. `.audit/` is excluded on purpose: those
    #: are point-in-time review records, and a reviewer proposing an id that was
    #: never written up should not fail the suite.
    _CITING_SUFFIXES = (".py", ".yaml", ".yml", ".md", ".ts", ".tsx")
    _CITING_ROOTS = ("concierge", "eval", "serve", "scripts", "configs", "docs", "web", "tests")
    _CITING_FILES = ("Makefile", "README.md")

    def test_every_cited_decision_id_resolves_to_an_entry(self):
        """A citation to an id with no entry sends the reader nowhere.

        The uniqueness test above catches the *other* way the id interface
        breaks. This catches the one that actually happened next: a batch of
        work added `Config.fingerprint_for` and left eleven sites across
        `concierge/` and `tests/` saying "see DECISIONS.md D237" while the
        highest entry in the file was D234, because writing the entry was a
        separate step that did not happen. Uniqueness passed the whole time —
        nothing was duplicated, the target simply did not exist.

        That matters more here than a broken link usually would: DECISIONS.md is
        what the owner studies before an interview, and the citation is how the
        code says "the reasoning for this line is written down". A dangling one
        means the reasoning is *not* written down, on precisely the lines
        someone thought worth defending.

        Only the forward direction is checked. An entry nothing cites is fine —
        plenty record decisions about process or data that no single line owns.
        """
        import re
        from pathlib import Path

        from concierge.config import REPO_ROOT

        entries = set(self._headings())
        dangling: dict[str, list[str]] = {}
        candidates = [
            path
            for root in self._CITING_ROOTS
            for path in Path(REPO_ROOT, root).rglob("*")
            if path.suffix in self._CITING_SUFFIXES
            and not {"__pycache__", "node_modules"} & set(path.parts)
        ]
        candidates += [Path(REPO_ROOT, name) for name in self._CITING_FILES]
        for path in candidates:
            body = path.read_text(encoding="utf-8", errors="replace")
            for cited in sorted(set(re.findall(r"\bD\d{3}\b", body))):
                if cited not in entries:
                    dangling.setdefault(cited, []).append(str(path.relative_to(REPO_ROOT)))

        assert not dangling, (
            "decision id(s) cited by the code with no entry in DECISIONS.md: "
            + "; ".join(
                f"{i} (cited in {', '.join(sorted(f))})" for i, f in sorted(dangling.items())
            )
            + ". Write the entry, or correct the citation — a reader following it "
            "finds nothing."
        )

    def test_no_citation_uses_a_malformed_decision_id(self):
        """A four-digit "id" is a citation the dangling-citation test cannot see.

        That test scans for `\\bD\\d{3}\\b`, which does not match inside a longer
        run of digits. So an id of four digits or more is checked against
        nothing and sends the reader nowhere — exactly like a dangling citation,
        but silently.

        It is not a hypothetical. Writing D242, this file's line numbers 1135
        and 1477 were pasted in as though they were decision ids; the entries
        meant were D096 and D129. Every id in this project is three digits, so
        anything longer is a mistake by construction.

        (The bad ids are described rather than written out: writing one here
        would fail this test, which is the check working.)
        """
        import re
        from pathlib import Path

        from concierge.config import REPO_ROOT

        malformed: dict[str, set[str]] = {}
        for root in self._CITING_ROOTS:
            for path in Path(REPO_ROOT, root).rglob("*"):
                if path.suffix not in self._CITING_SUFFIXES:
                    continue
                if {"__pycache__", "node_modules"} & set(path.parts):
                    continue
                for bad in re.findall(
                    r"\bD\d{4,}\b", path.read_text(encoding="utf-8", errors="replace")
                ):
                    malformed.setdefault(bad, set()).add(str(path.relative_to(REPO_ROOT)))
        for file in self._CITING_FILES:
            for bad in re.findall(
                r"\bD\d{4,}\b", Path(REPO_ROOT, file).read_text(encoding="utf-8")
            ):
                malformed.setdefault(bad, set()).add(file)

        assert not malformed, (
            "decision ids are three digits; these are not ids: "
            + ", ".join(f"{i} (in {', '.join(sorted(f))})" for i, f in sorted(malformed.items()))
            + ". A line number pasted as an id reads as a citation and resolves to nothing."
        )

    def test_the_citation_scan_actually_finds_citations(self):
        """Guards the test above against silently scanning nothing.

        If the roots or suffixes were narrowed, or `rglob` returned an empty
        list, the dangling-citation test would pass by examining no files at
        all. Pinned low (the code cites far more than five) so it asserts the
        scan works without becoming a test of how many citations exist.
        """
        import re
        from pathlib import Path

        from concierge.config import REPO_ROOT

        found: set[str] = set()
        for root in self._CITING_ROOTS:
            for path in Path(REPO_ROOT, root).rglob("*"):
                if path.suffix not in self._CITING_SUFFIXES:
                    continue
                if {"__pycache__", "node_modules"} & set(path.parts):
                    continue
                found.update(
                    re.findall(r"\bD\d{3}\b", path.read_text(encoding="utf-8", errors="replace"))
                )
        assert len(found) > 5


class TestMakefileTargets:
    """The Makefile is the project's documented interface, and it is not type-checked.

    Both properties here are ones that broke in practice, silently, in a way
    that looked like something else:

    1. **Phony.** Every target produces no file of its own name, so each one has
       to be declared `.PHONY`. `label` was once left out, and `pool` was left
       out the day it was added — one line below a comment describing exactly
       that mistake. The symptom is not an error: Make sees a file or directory
       in the repo root with the target's name, decides the work is up to date,
       and prints "nothing to be done", so the owner believes a step ran.

    2. **Flag forwarding.** `make pool --systems bm25` is Make rejecting its own
       unrecognized option, and it answers with a dump of Make's options that
       names neither the target nor the flag. The flags were not unreachable —
       `eval/README.md` writes several of them out as bare
       `uv run python -m eval.label_cli --stats` lines — but reaching them meant
       knowing which module backs which target, which is a second interface to
       the same programs. `--ablate-non-text`, the Stage 4 ablation CLAUDE.md §4
       asks for, is named in the README without any runnable command at all.

    So every documented target must either forward `$(ARGS)` or refuse it out
    loud. Silently dropping it is the one option ruled out: a dropped flag is a
    run the owner believes took an option it never saw.
    """

    @staticmethod
    def _makefile() -> str:
        from pathlib import Path

        from concierge.config import REPO_ROOT

        return Path(REPO_ROOT, "Makefile").read_text(encoding="utf-8")

    @classmethod
    def _documented_targets(cls) -> dict[str, str]:
        """``{target: recipe text}`` for every target carrying a ``##`` help line."""
        import re

        text = cls._makefile()
        out: dict[str, str] = {}
        for match in re.finditer(r"^([a-zA-Z_-]+):.*?##[^\n]*\n", text, flags=re.MULTILINE):
            rest = text[match.end() :]
            # The recipe is the run of lines that are tab-indented or blank,
            # stopping at the next line that starts in column zero.
            lines: list[str] = []
            for line in rest.splitlines():
                if line.startswith("\t"):
                    lines.append(line)
                elif line.strip() == "":
                    continue
                else:
                    break
            out[match.group(1)] = "\n".join(lines)
        return out

    def test_the_scan_finds_the_targets(self):
        # Guards both tests below against passing by examining nothing.
        targets = self._documented_targets()
        assert {"pool", "label", "eval", "train", "synth", "test"} <= set(targets), targets.keys()
        assert targets["pool"].strip().startswith("$(RUN) python -m eval.build_runs")

    def test_every_documented_target_is_phony(self):
        import re

        declared: set[str] = set()
        for line in re.findall(r"^\.PHONY:(.*)$", self._makefile(), flags=re.MULTILINE):
            declared.update(line.split())
        missing = sorted(set(self._documented_targets()) - declared)
        assert not missing, (
            f"documented target(s) not declared .PHONY: {missing}. Add them to a "
            ".PHONY line. Left out, each one stops working the moment a file or "
            "directory of that name appears in the repo root, and the failure "
            "reads as Make deciding the work is already done."
        )

    def test_every_documented_target_either_forwards_args_or_refuses_it(self):
        # `help` is the only exemption: it prints this file's own help lines and
        # runs none of the project's programs, so there is nothing to forward to.
        offenders = sorted(
            target
            for target, recipe in self._documented_targets().items()
            if target != "help" and "$(ARGS)" not in recipe and "refuse_args" not in recipe
        )
        assert not offenders, (
            f"target(s) that neither forward nor refuse ARGS: {offenders}. Append "
            "$(ARGS) to the program's line, or — if the target runs several "
            "programs, so one flag string cannot mean one thing — add "
            "$(call refuse_args,<target>,<where the flags do go>). Dropping ARGS "
            "silently is the one thing ruled out."
        )

    #: Docs whose `make <target> ARGS="..."` commands must actually work.
    _DOC_FILES = ("README.md", "eval/README.md", "docs/architecture.md")

    def test_every_documented_args_flag_exists_in_that_targets_program(self):
        """A flag shown on the wrong target is a command that cannot be run.

        The docs are where the owner (and a reader of the repo) gets these
        commands, and `ARGS` makes them one string away from a program that will
        reject them. This resolves each documented target to the module its
        recipe runs and checks the flags against that module's own parser, so
        `make train-ltr ARGS="--ablate-non-text"` cannot drift onto a target
        whose program has no such option — the mistake that was one line away
        while this was being written, since the flag had never appeared attached
        to any command at all.

        Static on purpose: reading `add_argument` / `typer.Option` out of the
        source costs nothing and needs no dataset, API key or model download.
        """
        import re
        from pathlib import Path

        from concierge.config import REPO_ROOT

        modules = {
            target: match.group(1)
            for target, recipe in self._documented_targets().items()
            if (match := re.search(r"python -m ([\w.]+)", recipe))
        }

        def declared_flags(module: str) -> set[str]:
            src = Path(REPO_ROOT, *module.split(".")).with_suffix(".py").read_text(encoding="utf-8")
            return set(re.findall(r'(?:add_argument|typer\.Option)\(\s*"(--[a-z0-9-]+)"', src))

        problems: list[str] = []
        checked = 0
        for doc in self._DOC_FILES:
            text = Path(REPO_ROOT, doc).read_text(encoding="utf-8")
            for target, argstr in re.findall(r'make ([a-z-]+) ARGS="([^"]*)"', text):
                if target not in modules:
                    problems.append(f"{doc}: `make {target} ARGS=` runs no python module")
                    continue
                available = declared_flags(modules[target])
                for flag in re.findall(r"(--[a-z0-9-]+)", argstr):
                    checked += 1
                    if flag not in available:
                        problems.append(
                            f'{doc}: `make {target} ARGS="{argstr}"` — '
                            f"{modules[target]} has no {flag}"
                        )
        assert not problems, "\n".join(problems)
        # Guards the loop against passing by matching nothing.
        assert checked >= 10, f"only {checked} documented ARGS flags found; the scan is broken"

    @pytest.mark.slow
    def test_make_can_parse_every_documented_target(self):
        """A dry run of each target: catches Makefile syntax errors, executes nothing.

        `make -n` expands the recipe and prints it without running it. This
        exists because the first draft of `refuse_args` put the recipe tab inside
        the `define` body instead of on the `$(call ...)` line, and *every*
        guarded target then died with "missing separator" — a parse error, so no
        text-only test of this file could have seen it.
        """
        import subprocess

        from concierge.config import REPO_ROOT

        failures: dict[str, str] = {}
        for target in sorted(self._documented_targets()):
            proc = subprocess.run(
                ["make", "-n", target],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
            )
            if proc.returncode != 0:
                failures[target] = (proc.stderr or proc.stdout).strip().splitlines()[0]
        assert not failures, f"`make -n <target>` failed: {failures}"

    @pytest.mark.slow
    @pytest.mark.parametrize("target", ["setup", "synth", "train", "fmt", "clean"])
    def test_a_target_that_cannot_forward_args_stops_instead_of_ignoring_them(self, target):
        import subprocess

        from concierge.config import REPO_ROOT

        # -n as well as the guard: if the guard ever stopped firing, this must
        # not fall through to really running `clean`.
        proc = subprocess.run(
            ["make", "-n", target, "ARGS=--assume-yes"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        assert proc.returncode != 0, f"make {target} ARGS=... silently ignored the flags"
        assert "does not forward ARGS" in proc.stderr, proc.stderr
        # The message has to say where the flags *do* go, or it only says no.
        assert "make " in proc.stderr or "takes no flags" in proc.stderr, proc.stderr
