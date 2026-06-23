#!/usr/bin/env python3
"""Merge data/ updates from shard artifacts into one canonical data/ tree."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Accept "2026-06-18T12:45:04.618860" and "...Z" variants.
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _prefer_recipe(
    current: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    c_dt = _parse_time(current.get("last_built"))
    i_dt = _parse_time(incoming.get("last_built"))
    if c_dt and i_dt:
        return incoming if i_dt >= c_dt else current
    if i_dt and not c_dt:
        return incoming
    if not i_dt and c_dt:
        return current
    # Fall back to incoming on tie/unknown; this lets later shards win.
    return incoming


def _merge_recipe_cache(
    base: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    out = dict(base)
    out.setdefault("version", incoming.get("version", base.get("version", "2.0")))
    out.setdefault("packages", {})

    base_pkgs = out["packages"]
    inc_pkgs = incoming.get("packages", {})

    for pkg_name, sandboxes in inc_pkgs.items():
        base_pkgs.setdefault(pkg_name, {})
        for sandbox_name, recipe in sandboxes.items():
            existing = base_pkgs[pkg_name].get(sandbox_name)
            if existing is None:
                base_pkgs[pkg_name][sandbox_name] = recipe
            else:
                base_pkgs[pkg_name][sandbox_name] = _prefer_recipe(
                    existing, recipe
                )
    return out


def _prefer_example(
    current: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    c_source = (current.get("source") or "").lower()
    i_source = (incoming.get("source") or "").lower()
    # Preserve curated/manual entries over auto-learned entries.
    if c_source == "manual" and i_source != "manual":
        return current
    if i_source == "manual" and c_source != "manual":
        return incoming

    c_dt = _parse_time(current.get("timestamp"))
    i_dt = _parse_time(incoming.get("timestamp"))
    if c_dt and i_dt:
        return incoming if i_dt >= c_dt else current
    if i_dt and not c_dt:
        return incoming
    if not i_dt and c_dt:
        return current
    return incoming


def _example_id(ex: dict[str, Any]) -> str:
    ex_id = ex.get("id")
    if ex_id:
        return str(ex_id)
    # Defensive fallback if malformed entry lacks id.
    return json.dumps(ex, sort_keys=True)


def _merge_examples_file(
    base: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    out = dict(base)
    merged: dict[str, dict[str, Any]] = {}

    for ex in base.get("examples", []):
        merged[_example_id(ex)] = ex
    for ex in incoming.get("examples", []):
        ex_id = _example_id(ex)
        existing = merged.get(ex_id)
        if existing is None:
            merged[ex_id] = ex
        else:
            merged[ex_id] = _prefer_example(existing, ex)

    out["examples"] = [merged[k] for k in sorted(merged.keys())]
    out.setdefault("version", incoming.get("version", base.get("version", "2.0")))
    return out


def _merge_recipe_cache_files(
    repo_data_dir: Path, artifact_root: Path
) -> bool:
    target = repo_data_dir / "recipe_cache.json"
    if not target.exists():
        return False

    merged = _load_json(target)
    changed = False

    for path in artifact_root.rglob("data/recipe_cache.json"):
        incoming = _load_json(path)
        before = json.dumps(merged, sort_keys=True)
        merged = _merge_recipe_cache(merged, incoming)
        after = json.dumps(merged, sort_keys=True)
        if after != before:
            changed = True

    if changed:
        _save_json(target, merged)
    return changed


def _merge_examples_files(repo_data_dir: Path, artifact_root: Path) -> bool:
    examples_dir = repo_data_dir / "examples"
    if not examples_dir.exists():
        return False

    changed = False
    for target in examples_dir.glob("*_examples.json"):
        merged = _load_json(target)
        target_changed = False
        suffix = f"data/examples/{target.name}"
        for path in artifact_root.rglob(target.name):
            if not str(path).endswith(suffix):
                continue
            incoming = _load_json(path)
            before = json.dumps(merged, sort_keys=True)
            merged = _merge_examples_file(merged, incoming)
            after = json.dumps(merged, sort_keys=True)
            if after != before:
                target_changed = True
        if target_changed:
            _save_json(target, merged)
            changed = True
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        required=True,
        help="Root directory where shard artifacts were downloaded.",
    )
    parser.add_argument(
        "--repo-data-dir",
        required=True,
        help="Repository data directory (typically ./data).",
    )
    args = parser.parse_args()

    artifact_root = Path(args.artifacts_root)
    repo_data_dir = Path(args.repo_data_dir)

    if not artifact_root.exists():
        print("No artifact root found; nothing to merge.")
        return 0

    rc_changed = _merge_recipe_cache_files(repo_data_dir, artifact_root)
    ex_changed = _merge_examples_files(repo_data_dir, artifact_root)
    print(
        "Merged data artifacts: "
        f"recipe_cache_changed={rc_changed}, examples_changed={ex_changed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
