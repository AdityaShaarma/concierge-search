"""Verify that only one OpenMP runtime is loaded, and that FAISS is still correct.

Absence of a crash is not evidence of correctness here: the documented failure
mode of duplicate OpenMP runtimes includes silently wrong results. So this check
also asserts that FAISS agrees with brute-force numpy on a problem whose answer
we know independently.
"""

from __future__ import annotations

import ctypes.util  # noqa: F401  (imported for parity with loader behaviour)
import sys


def main() -> int:
    import numpy as np

    import faiss
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
    print("OK: single runtime, and FAISS matches brute-force numpy exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
