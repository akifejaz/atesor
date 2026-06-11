#############################################################################
# Copyright (c) 2026 10xEngineers
#
# Author: Akif Ejaz <akif.ejaz@10xengineers.ai>
# This program and the accompanying materials are made available under the
# terms of the MIT License which is available at
# https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT
#############################################################################

"""Package a successfully ported repository as a downloadable zip.

Layout produced::

    <repo>-<YYYYMMDD-HHMMSS>-<platform>.zip
    ├── build_recipe.md          # the porting recipe (root of zip)
    ├── manifest.json            # machine-readable metadata
    ├── <repo>.log               # batch_test per-package log (optional)
    ├── agent_<repo>.log         # main.py per-package debug log (optional)
    └── <repo>/                  # source tree (excluding .git/, symlinks)
        └── ...

Used by ``main.py --package`` and the CI batch workflow to produce
artifacts that downstream consumers can download directly.
"""

import json
import logging
import os
import zipfile
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Top-level entries inside the repo subtree that are excluded from packages.
# Kept conservative on purpose — users asked for "repo code", not a clone.
_EXCLUDED_DIRS = frozenset({".git"})


def _safe_zip_path(packages_dir: str, base_name: str) -> str:
    """Return an unused path under ``packages_dir`` for ``base_name``.

    On collision, append ``.1``, ``.2``, ... before the ``.zip`` extension.
    Avoids silently overwriting an existing artifact when two runs share a
    seconds-precision timestamp (rare but observed in tight CI loops).
    """
    candidate = os.path.join(packages_dir, base_name)
    if not os.path.exists(candidate):
        return candidate
    stem, ext = os.path.splitext(base_name)
    n = 1
    while True:
        candidate = os.path.join(packages_dir, f"{stem}.{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _add_repo_tree(
    zf: zipfile.ZipFile,
    repo_path: str,
    arc_root: str,
) -> tuple[int, int]:
    """Add ``repo_path`` to ``zf`` under ``arc_root/``.

    Excludes:
      * directories named in ``_EXCLUDED_DIRS`` (e.g. ``.git``)
      * any symlink (file or dir) — security: avoids packaging files
        outside ``repo_path`` if a malicious tree links to them.

    Returns ``(files_added, symlinks_skipped)``.
    """
    files_added = 0
    symlinks_skipped = 0

    # ``followlinks=False`` keeps os.walk from descending into symlinked dirs.
    for root, dirs, files in os.walk(repo_path, followlinks=False):
        # Skip symlinked directories (os.walk lists them in ``dirs`` but
        # would only recurse if followlinks=True; we still want to log).
        kept_dirs = []
        for d in dirs:
            full = os.path.join(root, d)
            if d in _EXCLUDED_DIRS:
                continue
            if os.path.islink(full):
                symlinks_skipped += 1
                logger.warning(
                    "Skipping symlinked directory in package: %s", full
                )
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs

        for fname in files:
            abs_p = os.path.join(root, fname)
            if os.path.islink(abs_p):
                symlinks_skipped += 1
                logger.warning("Skipping symlink in package: %s", abs_p)
                continue
            try:
                rel_p = os.path.relpath(abs_p, start=repo_path)
                zf.write(abs_p, arcname=os.path.join(arc_root, rel_p))
                files_added += 1
            except (OSError, ValueError) as exc:
                logger.warning("Skipping unreadable file %s: %s", abs_p, exc)

    return files_added, symlinks_skipped


def package_build(
    repo_name: str,
    repo_path: str,
    recipe_path: str,
    platform_name: str,
    packages_dir: str,
    repo_url: Optional[str] = None,
    agent_log_path: Optional[str] = None,
    batch_log_path: Optional[str] = None,
) -> str:
    """Produce a zip artifact for a successful build.

    Args:
        repo_name: Slug used in the filename (e.g. ``"amass"``).
        repo_path: Host path to the cloned repository directory.
        recipe_path: Host path to the porting recipe markdown file.
        platform_name: ``"alpine"`` / ``"debian"`` / etc — used in the
            filename and manifest.
        packages_dir: Host directory in which to write the zip.
        repo_url: Original git URL, recorded in the manifest if provided.
        agent_log_path: Optional host path to the per-package agent log
            (``workspace/logs/agent_<repo>.log``). Included at the zip
            root as ``agent_<repo>.log`` when the file exists.
        batch_log_path: Optional host path to the per-package batch log
            (``output/batch_logs/<repo>.log``). Included at the zip
            root as ``<repo>.log`` when the file exists.

    Returns:
        Absolute path to the created zip file.

    Raises:
        FileNotFoundError: if ``repo_path`` is not a directory or
            ``recipe_path`` is not a regular file.
        OSError: on disk / zip write failures.
    """
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"Repository directory not found for packaging: {repo_path}"
        )
    if not os.path.isfile(recipe_path):
        raise FileNotFoundError(
            f"Build recipe not found for packaging: {recipe_path}"
        )

    os.makedirs(packages_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{repo_name}-{timestamp}-{platform_name}.zip"
    zip_path = _safe_zip_path(packages_dir, base_name)

    manifest = {
        "schema_version": 1,
        "repo_name": repo_name,
        "repo_url": repo_url,
        "platform": platform_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "recipe_filename_in_zip": "build_recipe.md",
        "source_root_in_zip": repo_name,
        "logs_in_zip": [],
    }

    # Resolve which optional logs are actually present on disk. We log
    # (don't fail) when a caller hands us a missing path — the most
    # common case is the batch log being absent for single-package runs
    # invoked outside batch_test.
    logs_to_add: list[tuple[str, str]] = []  # (host_path, arcname)
    if batch_log_path:
        if os.path.isfile(batch_log_path):
            logs_to_add.append((batch_log_path, f"{repo_name}.log"))
        else:
            logger.warning(
                "Batch log not found, omitting from package: %s",
                batch_log_path,
            )
    if agent_log_path:
        if os.path.isfile(agent_log_path):
            logs_to_add.append((agent_log_path, f"agent_{repo_name}.log"))
        else:
            logger.warning(
                "Agent log not found, omitting from package: %s",
                agent_log_path,
            )
    manifest["logs_in_zip"] = [arc for _, arc in logs_to_add]

    logger.info("Creating package: %s", zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(recipe_path, arcname="build_recipe.md")
        zf.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        for host_path, arcname in logs_to_add:
            zf.write(host_path, arcname=arcname)
        files_added, symlinks_skipped = _add_repo_tree(
            zf, repo_path, arc_root=repo_name
        )

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    logger.info(
        "Package built: %s (%.1f MB, %d files, %d symlinks skipped)",
        zip_path,
        size_mb,
        files_added,
        symlinks_skipped,
    )
    return zip_path


__all__ = ["package_build"]
