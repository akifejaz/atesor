#!/usr/bin/env python3
"""Compute the list of packages from a list-JSON that have no zip yet.

Used by the retry workflow to figure out which packages failed in the
main run and need to be re-tried. A package is considered "missing" when
no zip in ``--packages-dir`` matches the canonical filename pattern
produced by ``src.packager.package_build``:

    <repo>-<YYYYMMDD>-<HHMMSS>-<platform>.zip

The script emits the missing names, one per line, to stdout. With
``--format space`` they're space-joined on a single line, ready to pass
straight to ``batch_test.py`` as positional name filters.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Iterable


def _load_package_names(list_path: str) -> list[str]:
    """Return the ordered package names declared in a list JSON file."""
    with open(list_path) as fh:
        data = json.load(fh)
    pkgs = data.get("packages", [])
    out: list[str] = []
    for p in pkgs:
        name = p.get("name")
        if not name:
            continue
        out.append(name)
    return out


def _built_names(packages_dir: str, platform: str) -> set[str]:
    """Return the set of repo names for which a zip exists on disk."""
    if not os.path.isdir(packages_dir):
        return set()
    pat = re.compile(
        rf"^(?P<name>.+?)-\d{{8}}-\d{{6}}-{re.escape(platform)}\.zip$"
    )
    out: set[str] = set()
    for fname in os.listdir(packages_dir):
        m = pat.match(fname)
        if m:
            out.add(m.group("name"))
    return out


def _emit(names: Iterable[str], fmt: str) -> None:
    if fmt == "space":
        print(" ".join(names))
    else:  # default: lines
        for n in names:
            print(n)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--list",
        required=True,
        help="Path to a package-list JSON (e.g. .github/packages/smoke.json).",
    )
    p.add_argument(
        "--packages-dir",
        required=True,
        help="Directory containing previously-built .zip artifacts.",
    )
    p.add_argument(
        "--platform",
        required=True,
        choices=["alpine", "debian"],
        help="Platform slug used in the zip filename.",
    )
    p.add_argument(
        "--format",
        choices=["lines", "space"],
        default="lines",
        help="Output format. 'space' is convenient for shell substitution.",
    )
    args = p.parse_args(argv)

    declared = _load_package_names(args.list)
    built = _built_names(args.packages_dir, args.platform)
    missing = [n for n in declared if n not in built]

    _emit(missing, args.format)

    # Also surface a summary on stderr so CI logs are self-explanatory.
    print(
        f"[missing-pkgs] list={os.path.basename(args.list)} "
        f"platform={args.platform} declared={len(declared)} "
        f"built={len(built)} missing={len(missing)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
