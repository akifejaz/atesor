# RISC-V Porting Foundry - Development Sandbox
# 
# This Dockerfile creates a RISC-V development environment using QEMU user-mode emulation.
# It works on x86_64 hosts and provides a full RISC-V toolchain for cross-compilation.

FROM debian:bookworm-slim

LABEL maintainer="RISC-V Porting Foundry"
LABEL description="RISC-V development sandbox for automated software porting"

ENV DEBIAN_FRONTEND=noninteractive

# Install base development tools and RISC-V cross-compilation toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential build tools
    build-essential \
    cmake \
    ninja-build \
    meson \
    autoconf \
    automake \
    libtool \
    pkg-config \
    # Version control
    git \
    # Utilities
    wget \
    curl \
    ca-certificates \
    file \
    unzip \
    xz-utils \
    # Python
    python3 \
    python3-pip \
    python3-venv \
    # RISC-V cross-compilation toolchain
    gcc-riscv64-linux-gnu \
    g++-riscv64-linux-gnu \
    binutils-riscv64-linux-gnu \
    # For native RISC-V execution via QEMU
    qemu-user-static \
    qemu-system-misc \
    binfmt-support \
    # Debug tools
    gdb-multiarch \
    # Editor (for manual inspection)
    vim-tiny \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set up RISC-V environment variables for cross-compilation
ENV CC=riscv64-linux-gnu-gcc
ENV CXX=riscv64-linux-gnu-g++
ENV AR=riscv64-linux-gnu-ar
ENV RANLIB=riscv64-linux-gnu-ranlib
ENV STRIP=riscv64-linux-gnu-strip
ENV OBJCOPY=riscv64-linux-gnu-objcopy
ENV OBJDUMP=riscv64-linux-gnu-objdump
ENV NM=riscv64-linux-gnu-nm
ENV LD=riscv64-linux-gnu-ld

# CMake toolchain file for RISC-V cross-compilation
RUN mkdir -p /etc/cmake && echo '\
set(CMAKE_SYSTEM_NAME Linux)\n\
set(CMAKE_SYSTEM_PROCESSOR riscv64)\n\
\n\
set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)\n\
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)\n\
set(CMAKE_AR riscv64-linux-gnu-ar)\n\
set(CMAKE_RANLIB riscv64-linux-gnu-ranlib)\n\
\n\
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)\n\
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n\
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)\n\
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)\n\
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)\n\
' > /etc/cmake/riscv64-toolchain.cmake

# Create workspace directories
RUN mkdir -p /workspace/repos /workspace/output /workspace/logs /workspace/patches

# Set working directory
WORKDIR /workspace

# Volume for persistent output
VOLUME /workspace/output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD riscv64-linux-gnu-gcc --version || exit 1

# Keep container running
CMD ["tail", "-f", "/dev/null"]
