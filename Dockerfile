# Atesor AI - RISC-V Porting Sandbox
# 
# This Dockerfile creates a native RISC-V 64-bit development environment using Alpine Linux.
# It uses QEMU user-mode emulation via Docker's multi-platform support (binfmt_misc).

# Base image: Alpine Linux for RISC-V 64-bit
FROM alpine:latest

LABEL maintainer="Atesor AI"
LABEL description="Minimal RISC-V 64-bit sandbox for automated software porting"

# Install essentials for building and analysis
RUN apk add --no-cache \
    build-base \
    cmake \
    git \
    curl \
    bash \
    python3 \
    ca-certificates \
    file \
    tar \
    xz \
    pkgconfig \
    meson \
    ninja \
    autoconf \
    automake \
    libtool \
    gettext-dev \
    nasm \
    perl \
    linux-headers \
    zlib-dev \
    samurai \
    py3-pip \
    py3-jinja2 \
    py3-jsonschema \
    gfortran \
    texinfo \
    patch \
    diffutils \
    protobuf-dev \
    protoc \
    abseil-cpp-dev \
    openssl-dev \
    libpng-dev \
    util-linux \
    coreutils

# Install Go from the official riscv64 tarball instead of `apk add go`.
# Alpine's `go` package lags upstream (currently 1.25.10); many modern
# repos (cheat, doggo, garble, amass, gost, gh, ftpgrab, ...) require
# `go >= 1.26` in go.mod and fail at `go mod tidy` with
# "go.mod requires go >= 1.26 (running go 1.25.x; GOTOOLCHAIN=local)".
# Pinning a recent toolchain in the image avoids slow GOTOOLCHAIN
# downloads during agent runs (which routinely OOM-kill under QEMU).
ARG GO_VERSION=1.26.3
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-riscv64.tar.gz" \
    | tar -xz -C /usr/local \
    && /usr/local/go/bin/go version

# Set up environment variables for native compilation inside the sandbox
ENV CC=gcc
ENV CXX=g++
# Go environment: use direct proxy to avoid network issues in isolated containers
ENV GOPATH=/root/go
ENV PATH="${GOPATH}/bin:/usr/local/go/bin:${PATH}"
ENV GOPROXY=https://proxy.golang.org,direct
ENV GONOSUMCHECK=*
ENV GOFLAGS=-buildvcs=false
# Pin the bundled toolchain hard: never let `go` auto-download a newer
# version mid-build, because such downloads regularly OOM-kill (exit 137)
# under QEMU emulation and produce empty stderr that masks the real error.
ENV GOTOOLCHAIN=local

# Set up workspace structure matching the agent's expectations
WORKDIR /workspace
RUN mkdir -p /workspace/repos /workspace/output /workspace/logs /workspace/patches

# Volume for persistent output
VOLUME /workspace/output

# Health check to ensure the environment is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD gcc --version || exit 1

# Keep container running for the agent to connect and execute commands
CMD ["tail", "-f", "/dev/null"]
