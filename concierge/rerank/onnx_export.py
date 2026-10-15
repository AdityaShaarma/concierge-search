"""Export the Stage 3 cross-encoder to ONNX Runtime, and prove the scores match.

    python -m concierge.rerank.onnx_export --config configs/base.yaml

WHY
───
The reranker is the slowest stage in the cascade and it runs on a laptop CPU.
ONNX Runtime executes the same graph with a CPU-tuned engine (operator fusion,
no Python per-layer dispatch), which is often faster for a small encoder — and
sometimes is not, because PyTorch has fused attention kernels of its own. Which
way it falls is a property of the machine, the batch shape and the document
lengths, so this script *measures* both backends and prints what it measured.
Nothing here assumes ONNX is the faster option; ``serving.use_onnx`` should only
be turned on when a measurement on the serving host says so.

WHY THE PARITY CHECK IS THE POINT
─────────────────────────────────
An ONNX export can succeed and still be wrong: a traced control-flow branch, a
dropped attention mask, a padding assumption. Nothing crashes — the service just
starts returning slightly different scores, and slightly different scores mean a
different *ordering*, which means every number in the README was measured on a
system that is no longer the one being served. So we score identical pairs with
both backends and check two things:

1. ``max |torch_score - onnx_score| <= TOLERANCE`` (default 1e-4). This is a
   *stated* tolerance, not a measured one: two float32 implementations of the
   same graph differ at rounding scale, while these logits span several units, so
   1e-4 sits above numerical noise and far below any gap that could reorder two
   candidates. The actual difference is printed and saved on every run — read
   that number rather than trusting this paragraph.
2. **Identical ranking** of the probe pairs. Absolute agreement is the real
   requirement, but ordering is what the user sees, so it is asserted separately
   rather than inferred from the tolerance.

Both failures raise. An unverified export must never be allowed to serve, which
is also why ``serving.use_onnx`` ships as ``false`` in ``configs/base.yaml``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from concierge.config import Config, load_config
from concierge.rerank.model import rerank_dir, resolve_weights

ONNX_FILENAME = "model.onnx"
PARITY_FILENAME = "onnx_parity.json"

# See module docstring, point 1. Stated, justified, and asserted — not tuned until
# the test passed.
DEFAULT_TOLERANCE = 1e-4

# opset 17 covers everything a BERT-style encoder needs (including LayerNorm as a
# single op) and is supported by every onnxruntime release this project pins.
DEFAULT_OPSET = 17

# Probe inputs used when no real documents are on disk. These are INPUTS, not
# results: they exist to exercise short, long and awkward text through both
# backends. Any number this script prints is measured on whatever pairs it was
# actually given.
_PROBE_QUERIES = (
    "quiet cafe with wifi",
    "cheap ramen with outdoor seating",
    "somewhere good for a first date that isn't loud",
)
_PROBE_DOCUMENTS = (
    "Corner Coffee. Coffee shop, cafe. free WiFi, quiet, outdoor seating, $.",
    "Loud Sports Bar. Bar, American. loud, sports on TV, no outdoor seating, $$.",
    # Long enough to be truncated at max_length, which is exactly the path most
    # likely to differ between backends if the mask were exported wrong.
    "Noodle House. Ramen, Japanese. " + ("Rich broth and springy noodles. " * 200),
    "",  # empty document: a degenerate but reachable case for a sparse corpus
)


class ParityError(RuntimeError):
    """Raised when the ONNX export does not reproduce the PyTorch scores."""


@dataclass(frozen=True)
class ParityResult:
    """Measured comparison between the two backends.

    Every field is something this script observed. ``speedup`` is machine- and
    thread-count-specific and must be quoted with the machine, never as a
    property of ONNX.
    """

    n_pairs: int
    tolerance: float
    max_abs_diff: float
    mean_abs_diff: float
    # Same pairs, re-scored one at a time (batch 1, no padding). See check_parity:
    # this is what catches a traced graph that froze its batch or sequence axis.
    max_abs_diff_single_pair: float
    ordering_identical: bool
    torch_ms_per_pair: float
    onnx_ms_per_pair: float
    speedup: float
    max_length: int
    weights_source: str
    onnx_path: str

    @property
    def within_tolerance(self) -> bool:
        return (
            self.max_abs_diff <= self.tolerance and self.max_abs_diff_single_pair <= self.tolerance
        )

    def summary(self) -> str:
        return (
            f"parity over {self.n_pairs} pairs: max|Δ|={self.max_abs_diff:.3e} "
            f"(tolerance {self.tolerance:.1e}), mean|Δ|={self.mean_abs_diff:.3e}, "
            f"max|Δ| unbatched={self.max_abs_diff_single_pair:.3e}, "
            f"ordering {'identical' if self.ordering_identical else 'CHANGED'}\n"
            f"measured latency: torch {self.torch_ms_per_pair:.2f} ms/pair vs "
            f"onnx {self.onnx_ms_per_pair:.2f} ms/pair -> {self.speedup:.2f}x "
            "(this machine, CPU, default thread settings)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────


def _require_onnx() -> None:
    """Fail with instructions rather than a bare ImportError.

    ONNX is an optional extra (``pyproject.toml`` ``[project.optional-dependencies]``)
    because it is only needed in Phase 5 and the wheels are large.
    """
    missing = []
    for module in ("onnx", "onnxruntime"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise ImportError(
            f"missing {', '.join(missing)}. Install the optional extra: `uv sync --extra onnx`."
        )


def _wrap_for_export(model: Any, use_token_type_ids: bool) -> Any:
    """Wrap the HF model so its forward takes plain tensors and returns logits.

    Exporting the raw model means exporting something whose forward accepts
    keyword arguments and returns a dataclass. Wrapping it keeps the ONNX graph's
    inputs and single output explicit, which is what makes the ORT call site
    below readable (and the parity check trustworthy — there is no ambiguity
    about which output we compared).
    """
    import torch

    class Wrapped(torch.nn.Module):
        """One forward signature, with token_type_ids optional.

        BERT-family tokenizers emit token_type_ids; some other encoders do not.
        Keeping it a defaulted argument (rather than two conditional forwards)
        means one traced signature to reason about, and the export simply passes
        two tensors instead of three when the tokenizer has no segment ids.
        """

        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, input_ids: Any, attention_mask: Any, token_type_ids: Any = None) -> Any:
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                inputs["token_type_ids"] = token_type_ids
            return self.model(**inputs).logits

    wrapped = Wrapped()
    wrapped.eval()
    return wrapped


def export(
    cfg: Config,
    *,
    model_path: str | Path | None = None,
    out_path: Path | None = None,
    opset: int = DEFAULT_OPSET,
) -> tuple[Path, str]:
    """Write an ONNX graph for the cross-encoder. Returns ``(path, weights_source)``.

    Exports the fine-tuned checkpoint when one exists for this config, otherwise
    the pretrained base model — and returns which, because serving an ONNX graph
    built from the wrong weights is indistinguishable from serving the right ones
    until the eval numbers move.
    """
    _require_onnx()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name_or_path, weights_source = resolve_weights(cfg, model_path)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(name_or_path)
    model.eval()

    use_token_type_ids = "token_type_ids" in getattr(tokenizer, "model_input_names", [])
    wrapped = _wrap_for_export(model, use_token_type_ids)

    # Dummy input shapes do not constrain the exported graph as long as the axes
    # below are marked dynamic — but they must be >1 in both dimensions, or a
    # batch/sequence axis can be traced away as a constant.
    dummy = tokenizer(
        ["quiet cafe with wifi", "cheap ramen"],
        ["Corner Coffee. free WiFi, quiet.", "Noodle House. Ramen."],
        padding="max_length",
        truncation="only_second",
        max_length=min(32, cfg.rerank.max_length),
        return_tensors="pt",
    )
    input_names = ["input_ids", "attention_mask"] + (
        ["token_type_ids"] if use_token_type_ids else []
    )
    args = tuple(dummy[name] for name in input_names)

    target = out_path if out_path is not None else rerank_dir(cfg) / ONNX_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)

    # Both batch and sequence length vary at serving time (a shortlist of 50
    # pairs, padded to the longest document in the batch). Freezing either axis
    # would make the graph work for the export shape and crash — or worse,
    # silently pad — for everything else.
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}

    # WHICH EXPORTER. torch >= 2.9 defaults `torch.onnx.export` to the dynamo
    # exporter. We ask for the legacy TorchScript one (`dynamo=False`) for three
    # reasons observed on this project's pinned torch:
    #   * the dynamo path additionally requires `onnxscript`, which is not in the
    #     `onnx` extra in pyproject.toml;
    #   * it writes the weights to a sidecar `model.onnx.data` file, so the
    #     served artifact becomes two files that must travel together, while the
    #     legacy exporter writes one self-contained graph;
    #   * it silently raised the opset to 18 and converted back down, i.e. it
    #     rewrote the graph we asked for.
    # The legacy exporter is deprecated and will eventually be removed. When it
    # is, switch to `dynamo=True` with `dynamic_shapes` (not `dynamic_axes`), add
    # `onnxscript` to the extra, and re-run this script — the parity check below
    # is exactly the test that tells you whether the new graph is still faithful.
    export_kwargs: dict[str, Any] = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(
        wrapped,
        args,
        str(target),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        **export_kwargs,
    )
    return target, weights_source


# ─────────────────────────────────────────────────────────────────────────────
# Parity check
# ─────────────────────────────────────────────────────────────────────────────


def probe_pairs(cfg: Config, n_documents: int = 24) -> list[tuple[str, str]]:
    """Pairs to compare the backends on.

    Real business documents when they are on disk — the parity check should see
    the text the service will actually score, including the long documents most
    likely to expose a truncation or masking bug. Falls back to the built-in
    probe strings so the export is verifiable before the dataset is downloaded.
    """
    documents: list[str]
    try:
        from concierge.retrieval.build_index import find_documents_file, load_documents

        loaded = load_documents(find_documents_file(cfg))
        # Longest documents first: they exercise the max_length boundary.
        documents = [doc.text for doc in sorted(loaded, key=lambda d: -len(d.text))[:n_documents]]
    except (FileNotFoundError, ImportError):
        documents = list(_PROBE_DOCUMENTS)

    return [(query, document) for query in _PROBE_QUERIES for document in documents]


def _torch_scores(
    name_or_path: str, pairs: Sequence[tuple[str, str]], cfg: Config, batch_size: int
) -> tuple[list[float], float]:
    """Score with PyTorch on CPU. Returns ``(scores, seconds)``."""
    from concierge.rerank.model import TransformersCrossEncoderScorer

    scorer = TransformersCrossEncoderScorer(
        name_or_path, max_length=cfg.rerank.max_length, batch_size=batch_size, device="cpu"
    )
    # One warmup batch first: the first forward pass pays lazy kernel setup, and
    # including that in the timing would overstate the ONNX speedup.
    scorer.score_pairs(pairs[: min(len(pairs), batch_size)])
    started = time.perf_counter()
    scores = scorer.score_pairs(pairs)
    return scores, time.perf_counter() - started


def _onnx_scores(
    onnx_path: Path,
    name_or_path: str,
    pairs: Sequence[tuple[str, str]],
    cfg: Config,
    batch_size: int,
) -> tuple[list[float], float]:
    """Score with ONNX Runtime. Returns ``(scores, seconds)``."""
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    expected = {i.name for i in session.get_inputs()}

    def run(batch: Sequence[tuple[str, str]]) -> list[float]:
        encoded = tokenizer(
            [q for q, _ in batch],
            [d for _, d in batch],
            padding=True,
            truncation="only_second",
            max_length=cfg.rerank.max_length,
            return_tensors="np",
        )
        # Feed only what the graph declares, as int64: ORT is strict about both,
        # and a dtype mismatch here is a confusing runtime error rather than a
        # silent wrong answer.
        feed = {
            name: np.asarray(value, dtype=np.int64)
            for name, value in encoded.items()
            if name in expected
        }
        logits = session.run(["logits"], feed)[0]
        column = 0 if logits.shape[-1] == 1 else 1
        return [float(x) for x in logits[:, column]]

    run(list(pairs[: min(len(pairs), batch_size)]))  # warmup, as above
    started = time.perf_counter()
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        scores.extend(run(list(pairs[start : start + batch_size])))
    return scores, time.perf_counter() - started


def check_parity(
    cfg: Config,
    onnx_path: Path,
    *,
    model_path: str | Path | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    batch_size: int = 16,
) -> ParityResult:
    """Score the same pairs with both backends and compare. Raises on mismatch."""
    _require_onnx()
    name_or_path, weights_source = resolve_weights(cfg, model_path)
    test_pairs = list(pairs) if pairs is not None else probe_pairs(cfg)
    if not test_pairs:
        raise ValueError("parity check needs at least one pair")

    torch_scores, torch_seconds = _torch_scores(name_or_path, test_pairs, cfg, batch_size)
    onnx_scores, onnx_seconds = _onnx_scores(onnx_path, name_or_path, test_pairs, cfg, batch_size)
    # Score the SAME pairs again one at a time. Within a single batch every pair
    # is padded to the same length, so one batched comparison cannot tell whether
    # the exported graph is genuinely shape-dynamic — and the TorchScript tracer
    # does bake Python-level branches on tensor shapes into constants (it warns
    # about exactly that during export). Batch 1 changes both the batch size and
    # the sequence length, so a frozen axis or a padding-dependent graph shows up
    # here as a disagreement even when the batched numbers looked clean.
    onnx_single, _ = _onnx_scores(onnx_path, name_or_path, test_pairs, cfg, 1)

    diffs = [abs(a - b) for a, b in zip(torch_scores, onnx_scores, strict=True)]
    single_diffs = [abs(a - b) for a, b in zip(torch_scores, onnx_single, strict=True)]
    # Ranking comparison on the whole probe set: `sorted` is stable, so identical
    # scores produce identical orders and only a real reordering shows up.
    order_torch = sorted(range(len(torch_scores)), key=lambda i: -torch_scores[i])
    order_onnx = sorted(range(len(onnx_scores)), key=lambda i: -onnx_scores[i])

    result = ParityResult(
        n_pairs=len(test_pairs),
        tolerance=tolerance,
        max_abs_diff=max(diffs),
        mean_abs_diff=sum(diffs) / len(diffs),
        max_abs_diff_single_pair=max(single_diffs),
        ordering_identical=order_torch == order_onnx,
        torch_ms_per_pair=torch_seconds * 1000 / len(test_pairs),
        onnx_ms_per_pair=onnx_seconds * 1000 / len(test_pairs),
        speedup=torch_seconds / onnx_seconds if onnx_seconds > 0 else float("inf"),
        max_length=cfg.rerank.max_length,
        weights_source=weights_source,
        onnx_path=str(onnx_path),
    )

    if not result.within_tolerance:
        raise ParityError(
            f"ONNX scores differ from PyTorch by up to {result.max_abs_diff:.3e} batched and "
            f"{result.max_abs_diff_single_pair:.3e} unbatched, above the stated tolerance "
            f"{tolerance:.1e}. Do not serve this graph: every ranking it produces would "
            "differ from the evaluated system. If only the unbatched number is out, the "
            "export froze a batch or sequence axis.\n" + result.summary()
        )
    if not result.ordering_identical:
        raise ParityError(
            "ONNX scores are within tolerance but reorder the probe pairs, so the served "
            "ranking would not match the evaluated one.\n" + result.summary()
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the cross-encoder to ONNX and verify score parity."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model-path", default=None, help="Override which weights to export.")
    parser.add_argument("--out", default=None, help="Output .onnx path.")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_path = Path(args.out) if args.out else None
    path, weights_source = export(
        cfg, model_path=args.model_path, out_path=out_path, opset=args.opset
    )
    print(f"exported {weights_source} -> {path}", file=sys.stderr)

    result = check_parity(
        cfg,
        path,
        model_path=args.model_path,
        tolerance=args.tolerance,
        batch_size=args.batch_size,
    )
    print(result.summary())
    if result.speedup < 1.0:
        # Not a failure — the graph is faithful, it is just not faster here. ONNX
        # Runtime is not unconditionally quicker than PyTorch: torch uses fused
        # SDPA attention, the exported graph may not, and both default to using
        # every core. Serving a slower backend for no reason is still a
        # regression, so say so rather than letting `use_onnx: true` look like a
        # free win.
        print(
            "MEASURED SLOWER than PyTorch on this machine. Keep serving.use_onnx=false "
            "unless a measurement on the serving host says otherwise.",
            file=sys.stderr,
        )

    # Written next to the graph so the served artifact carries its own evidence.
    report_path = path.parent / PARITY_FILENAME
    report_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"✓ parity report {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
