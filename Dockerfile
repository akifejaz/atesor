# Atesor AI - RISC-V Porting Sandbox
# 
# This Dockerfile creates a native RISC-V 64-bit development environment using Alpine Linux.
# It uses QEMU user-mode emulation via Docker's multi-platform support (binfmt_misc).

# Base image: Alpine Linux for RISC-V 64-bit
FROM alpine:latest

LABEL maintainer="Atesor AI"
LABEL description="Minimal RISC-V 64-bit sandbox for automated software porting"

# Install absolute essentials for building and analysis
# build-base: provides gcc, g++, make, libc-dev, binutils
# cmake, git, curl: essential for fetching and building most repos
# bash, python3: common scripting requirements
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
    pkgconfig

# Set up environment variables for native compilation inside the sandbox
ENV CC=gcc
ENV CXX=g++

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
