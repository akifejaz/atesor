# Atesor AI: Smart Multi-stage Agentic System for RISC-V Software Porting 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)](https://github.com/langchain-ai/langgraph)

---

## Overview

Atesor AI is a state-of-the-art **multi-agent AI system** designed to automate the complex process of porting software packages from x86/ARM to RISC-V architecture. Built on modern agentic design patterns and powered by LangGraph, it intelligently handles:

- **Automated Analysis**: Deep understanding of build systems and dependencies
- **Smart Building**: Intelligent compilation with RISC-V optimization
- **Self-Healing**: Automatic error detection and fixing
<!-- - **Cost Optimized**: 60-70% reduction in LLM API calls through scripted operations
- **High Success Rate**: 95% success on simple packages, 60% on complex ones -->

---

## Architecture

### 3-Tier Hierarchical Design

```
┌─────────────────────────────────────────────────────────────┐
│               ORCHESTRATION LAYER                           │
│  ┌──────────┐   ┌────────────┐   ┌────────────┐             │
│  │ Planner  │──→│ Supervisor │──→│ Summarizer │             │
│  └──────────┘   └────────────┘   └────────────┘             │
├─────────────────────────────────────────────────────────────┤
│              INTELLIGENCE LAYER                             │
│  ┌────────┐      ┌─────────┐      ┌────────┐                │
│  │ Scout  │      │ Builder │      │ Fixer  │                │
│  └────────┘      └─────────┘      └────────┘                │
├─────────────────────────────────────────────────────────────┤
│           SCRIPTED OPERATIONS LAYER                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

```

The Scripted Operations Layer is a cost-saving layer that handles deterministic operations such as:
- Repository cloning
- File operations
- Build system detection
- Dependency extraction
- Architecture code scanning
- Caching

---

## Quick Start

### Prerequisites

- API key for LLM provider (OpenAI, Gemini, or OpenRouter), set .env file

### Installation

```bash
# Clone the repository
git clone https://github.com/akifejaz/atesor-ai
cd atesor-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# NOTE: Edit .env and add your API keys
```

### Docker Setup

```bash
# Build the RISC-V development container
docker build -t atesor-ai .

# Run the container
docker run -d --name atesor_sandbox atesor-ai
```

### Basic Usage

```bash
python3 main.py --help
usage: main.py [-h] [--repo REPO] [--max-attempts MAX_ATTEMPTS] [--verbose] [--cleanup] [--setup-only]

RISC-V Porting Foundry - Automated software porting agent

options:
  -h, --help            show this help message and exit
  --repo REPO, -r REPO  GitHub/GitLab repository URL to port
  --max-attempts MAX_ATTEMPTS, -m MAX_ATTEMPTS
                        Maximum fix attempts before escalation (default: 5)
  --verbose, -v         Enable verbose output
  --cleanup             Clean up Docker container and exit
  --setup-only          Only set up the Docker environment, don't run agent

Examples:
  python main.py --repo https://github.com/madler/zlib
  python main.py --repo https://github.com/sqlite/sqlite --max-attempts 10 --verbose
  python main.py --cleanup
        
```

---

## Configuration

### LLM Provider Configuration

The agent supports multiple LLM providers with automatic fallback:

```python
# In .env file
OPENAI_API_KEY=your_key        # OpenAI GPT-3.5/4
GOOGLE_API_KEY=your_key        # Gemini (Free tier available!)
OPENROUTER_API_KEY=your_key    # OpenRouter (Free models available!)
```

```python
# models.py
PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # openai, gemini, openrouter

MODEL_CONFIG = {
    "openai": {
        "model": "gpt-3.5-turbo",
        "temperature": {"planner": 0.1, "supervisor": 0.0, ...}
    },
    "gemini": {
        "model": "gemini-pro",
        "temperature": {"planner": 0.1, "supervisor": 0.0, ...}
    },
    "openrouter": {
        "model": "openrouter/free",  # Uses free models!
        "temperature": {"planner": 0.1, "supervisor": 0.0, ...}
    }
}
```

---

## Supported Build Systems

- C/C++ Makefiles/CMake
- Go 

More loading ... :)

---

## Example Workflows

After running the agent, you can check the results in the output folder. 

TODO: Add some example agent runs 

---

## Future Work

- Integrate the RISC-V DB for more robust analysis and patching
- Add Examples for Agent to understand better porting patterns
- Add more build systems support

Any other suggestions are welcome!


---

## Contributing

We welcome contributions! Please open PR or create an issue.

---

## Learning Resources

Built using concepts from:

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Andrew Ng's Agentic AI Course](https://www.deeplearning.ai/)
- [RISC-V ISA Manual](https://riscv.org/technical/specifications/)
- [Agentic Design Patterns Book](https://www.amazon.com/Agentic-Design-Patterns-Hands-Intelligent/dp/3032014018)

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Support

- [Issue Tracker](https://github.com/akifejaz/atesor/issues)
- [Akif Ejaz | Email](mailto:akifejaz40@gmail.com)

---

**Built with ❤️ for the RISC-V community**

*Making RISC-V software ecosystem as rich as x86/ARM, one package at a time.*
