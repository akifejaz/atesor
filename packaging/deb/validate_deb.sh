#!/usr/bin/env bash
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
#
# Smoke-tests the built .deb in a clean ubuntu:22.04 container: installs
# python3 + the package and runs the CLI. Heavy runtime deps
# (docker.io / qemu-user-static) are declared but skipped here via
# --force-depends so the smoke test stays fast.
#
# Usage: packaging/deb/validate_deb.sh

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "${here}/../.." && pwd)
out_dir="${repo_root}/dist"

version=$(cd "${repo_root}" && PYTHONPATH=. python3 -c \
    "from src import __version__; print(__version__)" 2>/dev/null || echo 0.1.0)
deb="atesor-ai_${version}_amd64.deb"

if [ ! -f "${out_dir}/${deb}" ]; then
    echo "ERROR: ${out_dir}/${deb} not found. Run build_deb.sh first." >&2
    exit 1
fi

echo ">> Validating ${deb} in a clean ubuntu:22.04 container"
docker run --rm -e DEB="${deb}" -v "${out_dir}":/out:ro ubuntu:22.04 bash -c '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends python3 >/dev/null
    # --force-depends lets the intentionally-absent heavy runtime deps
    # (docker.io / qemu-user-static) downgrade to warnings so this smoke
    # test stays fast. Do NOT mask the exit code: a genuine unpack or
    # postinst failure must still fail the validation.
    dpkg -i --force-depends "/out/${DEB}"
    dpkg -s atesor-ai | grep -q "^Status: install ok installed" \
        || { echo "FAIL: package not properly installed"; exit 1; }
    echo "--- which atesor-ai ---"; command -v atesor-ai
    echo "--- atesor-ai --version ---"; atesor-ai --version
    echo "--- atesor-ai --help (head) ---"; atesor-ai --help | head -6
    echo "--- import smoke (bundled deps) ---"
    /opt/atesor-ai/venv/bin/python3 -c "import langchain, langgraph, docker, pydantic, google.generativeai; print(\"bundled deps import OK\")"
'
echo ">> Validation OK"
