"""Consolidate the duplicate OpenMP runtimes in the virtualenv.

WHY THIS EXISTS
---------------
On macOS, four of our dependencies pull in their own OpenMP runtime:

  scikit-learn, faiss-cpu, torch   vendor a private copy under site-packages
  lightgbm                         links `@rpath/libomp.dylib`, whose rpath lists
                                   only /opt/homebrew/opt/libomp/lib, so it loads
                                   Homebrew's system-wide copy instead

Importing more than one of them loads several copies of `libomp.dylib` into a
single address space. The runtimes
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
import subprocess
import sys
import sysconfig
from pathlib import Path

# The copy every other library is pointed at. torch's is chosen because torch is
# the most tightly coupled to its own build of the runtime.
CANONICAL = Path("torch/lib/libomp.dylib")
# Vendored copies, replaced with a symlink to CANONICAL.
DUPLICATES = [
    Path("sklearn/.dylibs/libomp.dylib"),
    Path("faiss/.dylibs/libomp.dylib"),
]

# LightGBM needs different treatment: it does not vendor a copy, it resolves
# `@rpath/libomp.dylib` against an rpath containing only Homebrew's prefix, and
# has no @loader_path entry — so dropping a symlink beside it would not be found.
# We rewrite the dependency to an absolute path to CANONICAL instead, which
# bypasses rpath resolution entirely. Modifying a Mach-O invalidates its
# signature, so it is re-signed ad-hoc afterwards (required on Apple silicon).
RPATH_LINKED = [Path("lightgbm/lib/lib_lightgbm.dylib")]


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

    for rel in RPATH_LINKED:
        target = site / rel
        if not target.exists():
            print(f"  skip    {rel} (not installed)")
            continue
        current = subprocess.run(
            ["otool", "-L", str(target)], capture_output=True, text=True
        ).stdout
        if str(canonical.resolve()) in current:
            print(f"  ok      {rel} (already repointed)")
            continue
        backup = target.with_suffix(".dylib.orig")
        if not backup.exists():
            shutil.copy2(target, backup)
        subprocess.run(
            [
                "install_name_tool",
                "-change",
                "@rpath/libomp.dylib",
                str(canonical.resolve()),
                str(target),
            ],
            check=True,
            capture_output=True,
        )
        # Ad-hoc re-sign: Apple silicon refuses to load a modified, stale-signed image.
        subprocess.run(
            ["codesign", "--force", "-s", "-", str(target)],
            check=True,
            capture_output=True,
        )
        print(f"  repoint {rel} -> {CANONICAL}")
        changed += 1

    print(f"\n{changed} runtime(s) consolidated. Verify with: make check-openmp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
