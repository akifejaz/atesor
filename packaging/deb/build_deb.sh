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
# Builds the Atesor AI .deb inside a clean ubuntu:22.04 container so the
# bundled virtualenv matches the target (python3.10). Requires Docker on
# the build host. Output: dist/atesor-ai_<version>_amd64.deb
#
# Usage: packaging/deb/build_deb.sh [ubuntu-image]

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "${here}/../.." && pwd)
out_dir="${repo_root}/dist"
image="${1:-ubuntu:22.04}"

mkdir -p "${out_dir}"

version=$(cd "${repo_root}" && PYTHONPATH=. python3 -c \
    "from src import __version__; print(__version__)" 2>/dev/null || echo 0.1.0)

echo ">> Building atesor-ai ${version} (.deb) inside ${image}"
docker run --rm \
    -e VERSION="${version}" \
    -v "${repo_root}":/src:ro \
    -v "${out_dir}":/out \
    "${image}" bash /src/packaging/deb/build_in_container.sh

echo ">> Built: ${out_dir}/atesor-ai_${version}_amd64.deb"
echo ">> Install with: sudo apt-get install ./dist/atesor-ai_${version}_amd64.deb"
