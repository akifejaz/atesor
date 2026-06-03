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
    go \
    util-linux \
    coreutils

# Set up environment variables for native compilation inside the sandbox
ENV CC=gcc
ENV CXX=g++
# Go environment: use direct proxy to avoid network issues in isolated containers
ENV GOPATH=/root/go
ENV PATH="${GOPATH}/bin:/usr/local/go/bin:${PATH}"
ENV GOPROXY=https://proxy.golang.org,direct
ENV GONOSUMCHECK=*
ENV GOFLAGS=-buildvcs=false

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
