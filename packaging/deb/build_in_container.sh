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
# control metadata and produces /out/atesor-ai_<version>_amd64.deb.

set -euo pipefail

VERSION="${VERSION:-0.1.0}"
PREFIX=/opt/atesor-ai
PKGROOT=/tmp/pkgroot
SRC=/src
PKG="${SRC}/packaging/deb"

echo ">> [1/5] Installing build prerequisites"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates >/dev/null

echo ">> [2/5] Staging application tree at ${PREFIX}/app"
mkdir -p "${PREFIX}/app"
cp -a \
    "${SRC}/main.py" \
    "${SRC}/src" \
    "${SRC}/data" \
    "${SRC}/Dockerfile" \
    "${SRC}/Dockerfile.debian" \
    "${SRC}/requirements.txt" \
    "${SRC}/LICENSE" \
    "${SRC}/README.md" \
    "${PREFIX}/app/"
# Strip any caches copied from the source tree.
find "${PREFIX}/app" -name '__pycache__' -type d -prune -exec rm -rf {} + || true
find "${PREFIX}/app" -name '*.pyc' -delete || true

echo ">> [3/5] Building dependency virtualenv at ${PREFIX}/venv"
python3 -m venv --copies "${PREFIX}/venv"
"${PREFIX}/venv/bin/pip" install --no-cache-dir --upgrade pip wheel >/dev/null
"${PREFIX}/venv/bin/pip" install --no-cache-dir \
    -r "${PKG}/requirements-runtime.txt"
# Trim the venv to keep the package smaller.
find "${PREFIX}/venv" -name '__pycache__' -type d -prune -exec rm -rf {} + || true
find "${PREFIX}/venv" -name '*.pyc' -delete || true
find "${PREFIX}/venv" -name 'tests' -type d -prune -exec rm -rf {} + || true

echo ">> [4/5] Assembling package root"
rm -rf "${PKGROOT}"
mkdir -p "${PKGROOT}/opt" "${PKGROOT}/usr/bin" "${PKGROOT}/DEBIAN"
cp -a "${PREFIX}" "${PKGROOT}/opt/"
install -m 0755 "${PKG}/atesor-ai.launcher" "${PKGROOT}/usr/bin/atesor-ai"
install -m 0755 "${PKG}/postinst" "${PKGROOT}/DEBIAN/postinst"

installed_kb=$(du -sk "${PKGROOT}/opt" "${PKGROOT}/usr" | awk '{s+=$1} END {print s}')
sed -e "s/@VERSION@/${VERSION}/" -e "s/@SIZE@/${installed_kb}/" \
    "${PKG}/control.in" > "${PKGROOT}/DEBIAN/control"

echo ">> [5/5] Building .deb"
deb="/out/atesor-ai_${VERSION}_amd64.deb"
dpkg-deb --root-owner-group --build "${PKGROOT}" "${deb}"
echo
dpkg-deb --info "${deb}"
echo ">> Wrote ${deb}"
