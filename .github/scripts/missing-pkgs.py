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
    """Return the ordered package names declared in a list JSON file.

    Accepts both schemas in use across the workflow:

    * ``.github/packages/*.json`` — ``{"packages": [{"name": ...}, ...]}``
    * ``remaining-<platform>.json`` (from ``plan-remaining.py``) —
      ``{"packages": ["name1", "name2", ...]}``

    Mixed lists are tolerated; entries with neither a ``name`` key nor
    a string value are silently skipped.
    """
    with open(list_path) as fh:
        data = json.load(fh)
    pkgs = data.get("packages", [])
    out: list[str] = []
    for p in pkgs:
        if isinstance(p, str):
            if p:
                out.append(p)
            continue
        if isinstance(p, dict):
            name = p.get("name")
            if name:
                out.append(name)
    return out


def _apply_shard(
    names: list[str], shard_index: int, shard_total: int,
) -> list[str]:
    """Return the contiguous slice of ``names`` for one shard.

    Must mirror ``batch_test._apply_shard``: ceil-division chunks so
    every shard except possibly the last is full-sized, and every name
    appears in exactly one shard.
    """
    if shard_total <= 1:
        return list(names)
    n = len(names)
    chunk = -(-n // shard_total)  # ceil(n / shard_total)
    start = shard_index * chunk
    end = start + chunk
    return list(names[start:end])


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
    p.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help=(
            "Zero-based index of the shard to filter the declared list "
            "to before computing missing packages (requires "
            "--shard-total). Mirrors batch_test.py's contiguous "
            "ceil(N/total) chunking so the retry workflow can mirror "
            "the main run's shard layout."
        ),
    )
    p.add_argument(
        "--shard-total",
        type=int,
        default=None,
        help=(
            "Total shard count the declared list is split into "
            "(requires --shard-index). Pass 1 to disable sharding."
        ),
    )
    args = p.parse_args(argv)

    shard_index = args.shard_index
    shard_total = args.shard_total
    if (shard_index is None) != (shard_total is None):
        print(
            "[ERROR] --shard-index and --shard-total must be provided "
            "together.",
            file=sys.stderr,
        )
        return 2
    if shard_total is not None:
        if shard_total < 1:
            print(
                f"[ERROR] --shard-total must be >= 1 (got {shard_total}).",
                file=sys.stderr,
            )
            return 2
        if shard_index < 0 or shard_index >= shard_total:
            print(
                f"[ERROR] --shard-index={shard_index} out of range "
                f"[0, {shard_total}).",
                file=sys.stderr,
            )
            return 2

    declared = _load_package_names(args.list)
    if shard_total is not None:
        declared = _apply_shard(declared, shard_index, shard_total)
    built = _built_names(args.packages_dir, args.platform)
    missing = [n for n in declared if n not in built]

    _emit(missing, args.format)

    # Also surface a summary on stderr so CI logs are self-explanatory.
    shard_tag = (
        f" shard={shard_index + 1}/{shard_total}"
        if shard_total is not None
        else ""
    )
    print(
        f"[missing-pkgs] list={os.path.basename(args.list)}"
        f"{shard_tag} platform={args.platform} "
        f"declared={len(declared)} built={len(built)} "
        f"missing={len(missing)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
