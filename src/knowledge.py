"""
Static knowledge base for RISC-V porting, tool installation, and architecture-specific fixes.
"""

from .runtime import get_runtime_settings

# Alpine Linux (riscv64) Package Knowledge
# Based on Alpine 3.23+ which has robust riscv64 support
ALPINE_TOOL_MAP = {
    # Build System -> Package
    "cmake": "cmake",
    "make": "make",
    "ninja": "ninja",
    "meson": "meson",
    "autotools": "automake autoconf libtool",
    "gcc": "build-base",
    "g++": "build-base",
    "pkgconfig": "pkgconf",
    
    # Language -> Package
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
}

def get_install_command(tool: str) -> str:
    """Get the apk add command for a given tool/language."""
    pkg = ALPINE_TOOL_MAP.get(tool.lower(), tool.lower())
    return f"apk add --no-cache {pkg}"

def get_system_knowledge_summary() -> str:
    """Get a summary of installation knowledge for the agent."""
    snapshot = get_runtime_settings().knowledge_snapshot_label
    summary = f"## RISC-V Tool Installation Knowledge ({snapshot})\n"
    summary += "For Alpine Linux (riscv64), use 'apk add <package>'.\n"
    summary += "Common Tool mappings:\n"
    for tool, pkg in ALPINE_TOOL_MAP.items():
        summary += f"- {tool}: {pkg}\n"
    return summary

# Architecture-Specific Code Patterns & Solutions
ARCH_PATTERNS = {
    "x86_simd": {
        "patterns": [r"__SSE\d?__", r"__AVX\d?__", r"_mm_\w+"],
        "solution": "Replace with RISC-V Vector (RVV) intrinsics or generic scalar fallback.",
        "severity": "high"
    },
    "arm_simd": {
        "patterns": [r"__ARM_NEON", r"vld\d", r"vst\d"],
        "solution": "Replace with RVV intrinsics or generic scalar fallback.",
        "severity": "high"
    },
    "inline_asm": {
        "patterns": [r"__asm__", r"asm\s*\("],
        "solution": "Rewrite assembly in C or provide RISC-V assembly variant.",
        "severity": "critical"
    }
}
