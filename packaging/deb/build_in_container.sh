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
# Runs INSIDE a clean ubuntu:22.04 container (see build_deb.sh).
# Assembles /opt/atesor-ai (app tree + bundled venv) plus the Debian
# control metadata and produces /out/atesor-ai_<version>_<arch>.deb.

set -euo pipefail

VERSION="${VERSION:-0.1.0}"
PREFIX=/opt/atesor-ai
PKGROOT=/tmp/pkgroot
SRC=/src
PKG="${SRC}/packaging/deb"

# Refuse to run outside the build container: this script apt-installs
# packages and assembles /opt/atesor-ai on WHATEVER system executes it.
# build_deb.sh launches it inside a throwaway ubuntu container with the
# repo mounted at /src and dist/ at /out — those mounts are the marker.
if [ ! -f "${PKG}/build_in_container.sh" ] || [ ! -d /out ]; then
    echo "ERROR: build_in_container.sh must run inside the build" >&2
    echo "container (missing /src and/or /out mounts)." >&2
    echo "On the host, use:  bash packaging/deb/build_deb.sh" >&2
    exit 1
fi

echo ">> [1/7] Installing build prerequisites"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates >/dev/null

# Derive the architecture instead of hard-coding amd64: building on an
# arm64 host must not produce a mislabeled amd64 package.
ARCH=$(dpkg --print-architecture)
echo ">> Target architecture: ${ARCH}"

echo ">> [2/7] Staging application tree at ${PREFIX}/app"
mkdir -p "${PREFIX}/app"
cp -a \
    "${SRC}/main.py" \
    "${SRC}/src" \
    "${SRC}/data" \
    "${SRC}/Dockerfile" \
    "${SRC}/Dockerfile.debian" \
    "${SRC}/.dockerignore" \
    "${SRC}/.env-example" \
    "${SRC}/requirements.txt" \
    "${SRC}/LICENSE" \
    "${SRC}/README.md" \
    "${PREFIX}/app/"
# Strip caches and filelock droppings copied from the source tree.
find "${PREFIX}/app" -name '__pycache__' -type d -prune -exec rm -rf {} + || true
find "${PREFIX}/app" -name '*.pyc' -delete || true
find "${PREFIX}/app" -name '*.lock' -delete || true

# Security gate: the developer tree's .env holds real API keys and must
# never ship. Fail the build loudly if any .env is staged (a future
# cp -a addition could slip one in); .env-example is the only allowed
# environment file.
if find "${PREFIX}/app" -name '.env' -print | grep -q .; then
    echo "FATAL: a .env file was staged into the package — refusing to" \
         "ship secrets" >&2
    exit 1
fi

echo ">> [3/7] Building dependency virtualenv at ${PREFIX}/venv"
python3 -m venv --copies "${PREFIX}/venv"
"${PREFIX}/venv/bin/pip" install --no-cache-dir --upgrade pip wheel >/dev/null
"${PREFIX}/venv/bin/pip" install --no-cache-dir \
    -r "${PKG}/requirements-runtime.txt"
# Trim the venv to keep the package smaller.
find "${PREFIX}/venv" -name '__pycache__' -type d -prune -exec rm -rf {} + || true
find "${PREFIX}/venv" -name '*.pyc' -delete || true
find "${PREFIX}/venv" -name 'tests' -type d -prune -exec rm -rf {} + || true

# Disable bytecode writes for ANY invocation of the bundled
# interpreter (not just launcher runs): a root user calling the venv
# python directly would otherwise litter /opt with unowned .pyc files
# that dpkg cannot remove. sitecustomize is imported automatically by
# the site module.
site_pkgs=$("${PREFIX}/venv/bin/python3" -c \
    "import site; print(site.getsitepackages()[0])")
cat > "${site_pkgs}/sitecustomize.py" <<'EOF'
"""Keep /opt/atesor-ai byte-pristine: never write .pyc caches."""

import sys

sys.dont_write_bytecode = True
EOF

echo ">> [4/7] Writing BUILD_INFO (provenance for installed systems)"
{
    echo "atesor-ai ${VERSION} (${ARCH})"
    echo "built_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "base_image: $(. /etc/os-release && echo "${PRETTY_NAME}")"
    echo "python: $("${PREFIX}/venv/bin/python3" --version 2>&1)"
    echo
    echo "# Bundled Python dependencies (pip freeze)"
    "${PREFIX}/venv/bin/pip" freeze --no-color
} > "${PREFIX}/app/BUILD_INFO"

echo ">> [5/7] Assembling package root"
rm -rf "${PKGROOT}"
mkdir -p "${PKGROOT}/opt" "${PKGROOT}/usr/bin" "${PKGROOT}/DEBIAN"
cp -a "${PREFIX}" "${PKGROOT}/opt/"
install -m 0755 "${PKG}/atesor-ai.launcher" "${PKGROOT}/usr/bin/atesor-ai"
install -m 0755 "${PKG}/postinst" "${PKGROOT}/DEBIAN/postinst"
install -m 0755 "${PKG}/postrm" "${PKGROOT}/DEBIAN/postrm"

installed_kb=$(du -sk "${PKGROOT}/opt" "${PKGROOT}/usr" | awk '{s+=$1} END {print s}')
sed -e "s/@VERSION@/${VERSION}/" -e "s/@SIZE@/${installed_kb}/" \
    -e "s/@ARCH@/${ARCH}/" \
    "${PKG}/control.in" > "${PKGROOT}/DEBIAN/control"

echo ">> [6/7] Generating DEBIAN/md5sums (integrity data for dpkg -V)"
( cd "${PKGROOT}" && find opt usr -type f -exec md5sum {} + \
    | sort -k2 > DEBIAN/md5sums )

echo ">> [7/7] Building .deb"
deb="/out/atesor-ai_${VERSION}_${ARCH}.deb"
# Write to a temp name, then rename atomically: a concurrent reader
# (validate container, apt install, a second build) must never observe
# a half-written archive at the final path.
dpkg-deb --root-owner-group --build "${PKGROOT}" "${deb}.tmp"
mv -f "${deb}.tmp" "${deb}"
echo
dpkg-deb --info "${deb}"
echo ">> Wrote ${deb}"
