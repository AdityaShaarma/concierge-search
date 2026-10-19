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
