"""Consolidate the duplicate OpenMP runtimes in the virtualenv.

WHY THIS EXISTS
---------------
On macOS, three of our dependencies each ship their own copy of LLVM's OpenMP
runtime: scikit-learn, faiss-cpu, and torch. Importing more than one of them
loads several copies of `libomp.dylib` into a single address space. The runtimes
then fight over the same thread-barrier state, and the process either crashes in
`__kmp_join_barrier` or — far worse — keeps running and silently returns wrong
numbers. LLVM's own warning says exactly that:

    "it can degrade performance or cause incorrect results ... As an unsafe,
    unsupported, undocumented workaround you can set KMP_DUPLICATE_LIB_OK=TRUE"

We deliberately do NOT use that workaround. This project's entire value is that
its measurements are trustworthy; an environment variable whose documented
failure mode is "silently produce incorrect results" is not an acceptable
foundation for a benchmark.

Instead we make the premise false: point every bundled copy at one real runtime,
so only a single OpenMP library is ever loaded. Originals are preserved as
`.orig` next to each symlink, so this is fully reversible.

Run after any `uv sync`, which restores the vendored copies. `make setup` does
this for you.
"""

from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path

# The copy every other library is pointed at. torch's is chosen because torch is
# the most tightly coupled to its own build of the runtime.
CANONICAL = Path("torch/lib/libomp.dylib")
DUPLICATES = [
    Path("sklearn/.dylibs/libomp.dylib"),
    Path("faiss/.dylibs/libomp.dylib"),
]


def main() -> int:
    if sys.platform != "darwin":
        print("not macOS — nothing to do (the duplicate-libomp issue is macOS-specific)")
        return 0

    site = Path(sysconfig.get_paths()["purelib"])
    canonical = site / CANONICAL
    if not canonical.exists():
        print(f"canonical runtime missing: {canonical}\nrun `uv sync` first.")
        return 1

    changed = 0
    for rel in DUPLICATES:
        target = site / rel
        if not target.exists():
            print(f"  skip    {rel} (not installed)")
            continue
        if target.is_symlink():
            print(f"  ok      {rel} (already consolidated)")
            continue
        backup = target.with_suffix(".dylib.orig")
        if not backup.exists():
            shutil.copy2(target, backup)
        target.unlink()
        target.symlink_to(canonical.resolve())
        print(f"  linked  {rel} -> {CANONICAL}")
        changed += 1

    print(f"\n{changed} runtime(s) consolidated. Verify with: make check-openmp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
