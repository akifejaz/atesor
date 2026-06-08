#!/usr/bin/env python3
"""Compute the per-platform "remaining" package plan for batch-port.

Given:

* a package list JSON (e.g. ``.github/packages/full.json``),
* the set of asset filenames already published under the current
  monthly release tag (one name per line in ``--released-assets``),
* a target ``--platform`` slug (``alpine`` or ``debian``), and
* a shard ``--group-size``,

this script writes the ordered list of packages that still need
building for that platform to ``--remaining-out`` (as
``{"packages": [...]}``), and emits two ``KEY=VALUE`` lines suitable
for ``$GITHUB_OUTPUT``::

    <prefix>_total=<N>
    <prefix>_groups=[0, 1, ..., N-1]

where ``<prefix>`` defaults to the platform name. ``N`` is the number
of shards needed to cover the remaining packages at ``--group-size``
each (``ceil(remaining / group_size)``).

"Already released" means an asset under the current monthly tag whose
filename matches the canonical pattern produced by
``src.packager.package_build``::

    <package>-<YYYYMMDD>-<HHMMSS>-<platform>.zip

The platform is matched exactly, so ``foo-...-alpine.zip`` does not
mark ``foo`` as released for the ``debian`` platform.

When the remaining list is empty the script emits ``<prefix>_total=0``
and ``<prefix>_groups=[]`` — downstream matrix jobs are then skipped
naturally because their matrix has no entries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Iterable

_PACKAGES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "packages")
)


def _resolve_list_path(value: str) -> str:
    """Bare name resolves under ``.github/packages/<name>.json``.

    Mirrors ``plan-shards.py::_resolve_list_path`` so the workflow can
    pass either ``--list full`` (production) or an explicit file path
    (tests, ad-hoc invocations).
    """
    if os.sep in value or value.endswith(".json"):
        return value
    return os.path.join(_PACKAGES_DIR, f"{value}.json")


def _load_package_names(list_path: str) -> list[str]:
    """Return the ordered package names declared in a list JSON file."""
    with open(list_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    pkgs = data.get("packages", [])
    out: list[str] = []
    for p in pkgs:
        name = p.get("name")
        if not name:
            continue
        out.append(name)
    return out


def _released_names(
    asset_names: Iterable[str], platform: str,
) -> set[str]:
    """Return the set of package names already released for ``platform``.

    Filenames that don't match the canonical pattern are silently
    ignored — they can't be confidently attributed to a package, so we
    prefer to err on the side of rebuilding.
    """
    pat = re.compile(
        rf"^(?P<name>.+?)-\d{{8}}-\d{{6}}-{re.escape(platform)}\.zip$"
    )
    out: set[str] = set()
    for raw in asset_names:
        name = raw.strip()
        if not name:
            continue
        m = pat.match(name)
        if m:
            out.add(m.group("name"))
    return out


def _load_released_assets(path: str | None) -> list[str]:
    """Read released asset filenames from ``path`` (one per line).

    Missing or empty file is treated as "no assets" — that's the
    expected state on the first run of a new month.
    """
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def compute_plan(
    declared: list[str],
    released: set[str],
    group_size: int,
) -> tuple[list[str], int, list[int]]:
    """Return ``(remaining, total_shards, group_indices)``.

    ``remaining`` preserves the declared order; ``total_shards`` is
    ``ceil(len(remaining) / group_size)`` (0 when empty).
    """
    if group_size < 1:
        raise ValueError(f"group-size must be >= 1 (got {group_size})")
    remaining = [n for n in declared if n not in released]
    if not remaining:
        return remaining, 0, []
    total = math.ceil(len(remaining) / group_size)
    return remaining, total, list(range(total))


def _write_remaining(path: str, remaining: list[str]) -> None:
    """Persist the remaining list as ``{"packages": [...]}`` JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"packages": remaining}, fh, indent=2)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--list", required=True,
        help="Path to a package-list JSON (e.g. .github/packages/full.json).",
    )
    p.add_argument(
        "--released-assets", required=False, default=None,
        help=(
            "Path to a file containing released asset filenames, one "
            "per line. Missing or empty file = no skips."
        ),
    )
    p.add_argument(
        "--platform", required=True, choices=["alpine", "debian"],
        help="Platform slug used in zip filenames.",
    )
    p.add_argument(
        "--group-size", required=True, type=int,
        help="Max packages per shard (matches main workflow's input).",
    )
    p.add_argument(
        "--remaining-out", required=True,
        help="Where to write the ordered remaining list as JSON.",
    )
    p.add_argument(
        "--output-key-prefix", default=None,
        help=(
            "Prefix for the emitted KEY=VALUE lines. Defaults to "
            "the platform slug, yielding e.g. ``debian_total`` and "
            "``debian_groups``."
        ),
    )
    args = p.parse_args(argv)

    prefix = args.output_key_prefix or args.platform

    declared = _load_package_names(_resolve_list_path(args.list))
    released_all = _load_released_assets(args.released_assets)
    released = _released_names(released_all, args.platform)

    remaining, total, groups = compute_plan(
        declared, released, args.group_size,
    )

    _write_remaining(args.remaining_out, remaining)

    print(f"{prefix}_total={total}")
    print(f"{prefix}_groups={json.dumps(groups)}")

    print(
        f"[plan-remaining] list={os.path.basename(args.list)} "
        f"platform={args.platform} declared={len(declared)} "
        f"released_assets={len(released)} "
        f"skipped={len(declared) - len(remaining)} "
        f"remaining={len(remaining)} shards={total} "
        f"(group_size={args.group_size}) -> {args.remaining_out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
