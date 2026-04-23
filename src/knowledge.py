"""
Static knowledge base for RISC-V porting, tool installation, and architecture-specific fixes.
Updated as of Feb 2026.
"""

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

def get_system_knowledge_summary() -> str:
    """Get a summary of installation knowledge for the agent."""
    summary = "## RISC-V Tool Installation Knowledge (Feb 2026)\n"
    summary += "For Alpine Linux (riscv64), use 'apk add <package>'.\n"
    summary += "Common Tool mappings:\n"
    for tool, pkg in ALPINE_TOOL_MAP.items():
        summary += f"- {tool}: {pkg}\n"
    return summary
