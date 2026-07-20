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
# the build host. Output: dist/atesor-ai_<version>_<arch>.deb plus a
# full build log next to it.
#
# Usage: packaging/deb/build_deb.sh [ubuntu-image]

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "${here}/../.." && pwd)
out_dir="${repo_root}/dist"
image="${1:-ubuntu:22.04}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is required to build the .deb (the package is" >&2
    echo "assembled inside a clean ${image} container)." >&2
    exit 1
fi

mkdir -p "${out_dir}"

version=$(cd "${repo_root}" && PYTHONPATH=. python3 -c \
    "from src import __version__; print(__version__)" 2>/dev/null || echo 0.1.0)

build_log="${out_dir}/atesor-ai_${version}_build.log"
echo ">> Building atesor-ai ${version} (.deb) inside ${image}"
echo ">> Build log: ${build_log}"
docker run --rm \
    -e VERSION="${version}" \
    -v "${repo_root}":/src:ro \
    -v "${out_dir}":/out \
    "${image}" bash /src/packaging/deb/build_in_container.sh \
    2>&1 | tee "${build_log}"

# The architecture is derived inside the container, so resolve the
# artifact name from disk instead of assuming amd64.
deb_path=$(ls -t "${out_dir}/atesor-ai_${version}"_*.deb | head -1)
echo ">> Built: ${deb_path}"
echo ">> Install with: sudo apt-get install ./${deb_path#"${repo_root}"/}"
