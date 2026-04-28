"""
Static knowledge base for RISC-V porting on Alpine Linux (musl/riscv64).
Covers tool installation, architecture detection, common porting patterns, and known issues.
"""

# ============================================================================
# ALPINE LINUX (riscv64) PACKAGE MAP
# ============================================================================

ALPINE_TOOL_MAP = {
    # Build systems
    "cmake": "cmake",
    "make": "make",
    "ninja": "ninja",
    "meson": "meson",
    "autotools": "automake autoconf libtool",
    "gcc": "build-base",
    "g++": "build-base",
    "pkgconfig": "pkgconf",
    # Languages
    "go": "go",
    "rust": "rust cargo",
    "python3": "python3 py3-pip",
    "node": "nodejs npm",
    "java": "openjdk17",
    # Libraries
    "zlib": "zlib-dev",
    "openssl": "openssl-dev",
    "curl": "curl-dev",
    "git": "git",
    "libatomic": "libatomic",
}

# ============================================================================
# RISC-V ARCHITECTURE KNOWLEDGE
# ============================================================================

RISCV_PREPROCESSOR_MACROS = {
    "__riscv": "Defined on any RISC-V target",
    "__riscv_xlen": "Register width: 32 or 64",
    "__riscv_flen": "FP register width: 32 (F ext) or 64 (D ext), absent if no FP",
    "__riscv_atomic": "Defined when A (atomic) extension is present",
    "__riscv_mul": "Defined when M (multiply/divide) extension is present",
    "__riscv_vector": "Defined when V (vector) extension is present",
    "__riscv_compressed": "Defined when C (compressed) extension is present",
}

RISCV_ARCH_DETECTION_SNIPPET = """\
#if defined(__riscv)
  #if __riscv_xlen == 64
    /* RISC-V 64-bit */
  #elif __riscv_xlen == 32
    /* RISC-V 32-bit */
  #endif
#endif"""

# Common error patterns when porting to RISC-V on Alpine/musl
COMMON_PORTING_ISSUES = {
    "x86_intrinsics": {
        "symptoms": [
            "error: unknown type name '__m128'",
            "error: unknown type name '__m256'",
            "undefined reference to '_mm_*'",
            "undefined reference to '_mm256_*'",
            "'xmmintrin.h' file not found",
            "'immintrin.h' file not found",
        ],
        "root_cause": "x86 SSE/AVX SIMD intrinsics not available on RISC-V",
        "solutions": [
            "Use SIMDe (SIMD Everywhere) header-only library for portable SIMD",
            "Provide scalar C fallback under #if !defined(__x86_64__) && !defined(__aarch64__)",
            "Use RISC-V Vector (RVV) intrinsics if V extension is available (#ifdef __riscv_vector)",
            "Disable SIMD via build option (e.g., -DENABLE_SSE=OFF, -DWITH_SIMD=0)",
        ],
    },
    "inline_assembly": {
        "symptoms": [
            "unknown register name",
            "invalid instruction mnemonic",
            "error in asm",
        ],
        "root_cause": "x86/ARM inline assembly incompatible with RISC-V ISA",
        "solutions": [
            "Guard with #if defined(__x86_64__) / #elif defined(__riscv)",
            "Replace with portable C code or compiler builtins",
            "Provide RISC-V assembly implementation for performance-critical paths",
        ],
    },
    "builtin_functions": {
        "symptoms": [
            "__builtin_ia32_",
            "__rdtsc",
            "__builtin_cpu_supports",
        ],
        "root_cause": "x86-specific GCC/Clang builtin functions",
        "solutions": [
            "Replace __rdtsc with __builtin_readcyclecounter() or clock_gettime()",
            "Replace __builtin_ia32_* with portable C equivalents",
            "Use C11 atomics instead of __sync_* builtins where possible",
        ],
    },
    "memory_ordering": {
        "symptoms": [
            "data races under RISC-V that pass on x86",
            "lock-free algorithms failing",
        ],
        "root_cause": "RISC-V has a weaker memory model (RVWMO) than x86 (TSO)",
        "solutions": [
            "Use C11/C++11 atomics with explicit memory orderings",
            "Add __atomic_thread_fence() where x86 relied on implicit ordering",
            "Audit lock-free data structures for RISC-V memory model",
        ],
    },
    "musl_libc": {
        "symptoms": [
            "execinfo.h: No such file or directory",
            "undefined reference to 'backtrace'",
            "undefined reference to 'backtrace_symbols'",
            "undefined reference to 'mallinfo'",
            "error: 'REG_RIP' undeclared",
            "error: 'REG_EIP' undeclared",
            "'sys/cdefs.h' file not found",
        ],
        "root_cause": "Alpine uses musl libc, not glibc; some GNU extensions are absent",
        "solutions": [
            "Guard backtrace usage with #ifdef __GLIBC__",
            "Replace mallinfo with platform-independent memory tracking",
            "Install musl-compatible alternatives (e.g., libexecinfo-dev for backtrace)",
            "Use portable signal handling instead of x86 register names",
        ],
    },
    "config_guess": {
        "symptoms": [
            "machine 'riscv64' not recognized",
            "unsupported host cpu type",
            "invalid configuration",
            "config.sub: too many arguments",
        ],
        "root_cause": "Stale config.guess/config.sub that predate RISC-V support",
        "solutions": [
            "Update config.guess and config.sub: apk add config-guess-wrapper || cp /usr/share/automake-*/config.* .",
            "Specify --build/--host explicitly: ./configure --build=riscv64-alpine-linux-musl",
            "For CMake: no action needed (CMake auto-detects RISC-V)",
        ],
    },
    "atomics_linking": {
        "symptoms": [
            "undefined reference to '__atomic_load'",
            "undefined reference to '__atomic_store'",
            "undefined reference to '__atomic_compare_exchange'",
        ],
        "root_cause": "Some atomic operations require libatomic on RISC-V",
        "solutions": [
            "Link with -latomic (add to LDFLAGS or CMake target_link_libraries)",
            "For CMake: target_link_libraries(target PRIVATE atomic)",
            "For Makefile: LDFLAGS += -latomic",
        ],
    },
    "cache_line_size": {
        "symptoms": [
            "hardcoded CACHELINE_SIZE 64",
            "performance regression due to false sharing",
        ],
        "root_cause": "RISC-V cache line size varies by implementation (32/64/128 bytes)",
        "solutions": [
            "Use runtime detection or conservative default",
            "Replace hardcoded values with sysconf or macro-based detection",
        ],
    },
}

# RISC-V target triplets for different configurations
RISCV_TRIPLETS = {
    "alpine_native": "riscv64-alpine-linux-musl",
    "gnu_linux": "riscv64-unknown-linux-gnu",
    "bare_metal": "riscv64-unknown-elf",
}


def get_system_knowledge_summary() -> str:
    """Produce a compact knowledge summary for injection into agent prompts."""
    lines = [
        "## RISC-V Porting Knowledge (Alpine Linux / musl / riscv64)",
        "",
        "### Package Installation",
        "Use `apk add <package>`. Common mappings:",
    ]
    for tool, pkg in ALPINE_TOOL_MAP.items():
        lines.append(f"  - {tool} → `{pkg}`")

    lines.append("")
    lines.append("### Architecture Detection (C/C++ preprocessor)")
    for macro, desc in RISCV_PREPROCESSOR_MACROS.items():
        lines.append(f"  - `{macro}`: {desc}")

    lines.append("")
    lines.append("### Common Porting Issues")
    for issue_key, issue in COMMON_PORTING_ISSUES.items():
        lines.append(f"  **{issue_key}**: {issue['root_cause']}")
        lines.append(f"    Fix: {issue['solutions'][0]}")

    lines.append("")
    lines.append("### Key Facts")
    lines.append("  - Alpine uses **musl libc**, not glibc — some GNU extensions are missing")
    lines.append("  - Native target triplet: `riscv64-alpine-linux-musl`")
    lines.append("  - Do NOT use cross-compilation flags when building natively inside the sandbox")
    lines.append("  - RISC-V memory model (RVWMO) is weaker than x86 (TSO) — explicit fences may be needed")
    lines.append("  - RISC-V ISA extensions vary by hardware — detect, do not assume")

    return "\n".join(lines)
