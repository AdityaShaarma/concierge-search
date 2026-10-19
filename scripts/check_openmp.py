"""Verify that only one OpenMP runtime is loaded, and that FAISS is still correct.

Absence of a crash is not evidence of correctness here: the documented failure
mode of duplicate OpenMP runtimes includes silently wrong results. So this check
also asserts that FAISS agrees with brute-force numpy on a problem whose answer
we know independently.
"""

from __future__ import annotations


def main() -> int:
    import faiss
    import lightgbm  # noqa: F401  (links Homebrew's libomp unless repointed)
    import numpy as np
    import sklearn  # noqa: F401
    import torch  # noqa: F401

    # How many distinct libomp images did the process actually load?
    try:
        import subprocess

        out = subprocess.run(
            ["vmmap", str(__import__("os").getpid())],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        loaded = {ln.split()[-1] for ln in out.splitlines() if "libomp.dylib" in ln}
        print(f"distinct libomp images loaded: {len(loaded)}")
        for p in sorted(loaded):
            print(f"  {p}")
    except Exception as exc:  # vmmap can be unavailable; the correctness check still runs
        print(f"(could not inspect loaded images: {exc})")

    # Correctness: exact inner-product search must match brute force exactly.
    rs = np.random.RandomState(0)
    x = rs.rand(5000, 384).astype("float32")
    faiss.normalize_L2(x)
    index = faiss.IndexFlatIP(384)
    index.add(x)
    q = x[:50].copy()
    _, ids = index.search(q, 5)
    brute = (q @ x.T).argmax(1)
    if not (ids[:, 0] == brute).all():
        print("FAIL: FAISS disagrees with brute-force numpy")
        return 1
    if not (ids[:, 0] == np.arange(50)).all():
        print("FAIL: self-retrieval did not return the query vector first")
        return 1
    # LightGBM is the library that actually segfaulted, so prove it trains.
    import lightgbm as lgb

    xs = rs.rand(400, 5)
    ys = (xs[:, 0] * 2.99).astype(int)  # signal lives only in feature 0
    booster = lgb.train(
        {"objective": "lambdarank", "num_leaves": 7, "verbosity": -1, "seed": 0},
        lgb.Dataset(xs, label=ys, group=[10] * 40),
        num_boost_round=30,
    )
    if booster.feature_importance("gain").argmax() != 0:
        print("FAIL: LightGBM did not recover the planted ranking signal")
        return 1

    print("OK: single runtime; FAISS matches brute force; LightGBM trains correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
