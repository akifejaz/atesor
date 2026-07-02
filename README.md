# Atesor AI

**Agentic System that autonomously ports x86/ARM Packages to RISC-V (riscv64).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-green.svg)](https://github.com/langchain-ai/langgraph)

Atesor AI takes a source code url (i.e GitHub repo link), builds the package natively inside a RISC-V Docker sandbox, fixes whatever breaks, and emits a reproducible porting recipe. Supported build systems cover the **C, C++, and Go** ecosystems (Make, CMake, Meson, autotools, Cargo, Go modules), more are coming.. It runs across both **Alpine (musl)** and **Debian/Ubuntu (glibc)** sandboxes (and **native/real RISC-V hardware support is coming** too..), so a package is verified on the two libc families that matter in practice.

**NOTE**: Native builds (on real RISC-V hardware) is in process and will be added in a future release. For now, QEMU/binfmt emulation is used for all builds. (see TODO)

---

## Table of Contents

- [Why](#why)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Key features](#key-features)
- [Development](#development)
- [Contributing](#contributing)

---

## Why

The RISC-V software ecosystem still has gaps that x86 and ARM solved years ago. Porting work is repetitive: clone, detect the build system, install the right packages, hit the same handful of issues (stale `config.guess`, x86-only SIMD, Go's `-buildvcs` trap, musl vs glibc headers and many more), patch, retry. Atesor AI automates that loop with a small team of specialized LLM agents and a deterministic scripted-ops layer that handles the boring up to 70% for free.

Output is a reproducible Markdown recipe a human or CI can replay, plus the ready-to-use RISC-V build artifacts.

---

## How it works

1. **Scripted analysis** clones the repo inside the sandbox and detects the build system, dependencies, and architecture-specific code - zero LLM cost.
2. **Planner** drafts a high-level `TaskPlan` from that analysis.
3. **Supervisor** routes work between Scout, Builder, and Fixer, watches for error loops, and decides when to escalate.
4. **Builder** runs the build natively on RISC-V via QEMU/binfmt. **Fixer** patches whatever breaks. **Scout** answers targeted questions about the source tree.
5. **Artifact scanner** verifies the produced binaries are real `riscv64` ELF files - not silent x86 fallthroughs.
6. **Recipe** is written to disk and cached, keyed by `(package, sandbox)`. A later cache hit re-renders `{repo}_recipe.md` from that entry and skips the pipeline entirely.

The supervisor → executor loop is built on [LangGraph](https://github.com/langchain-ai/langgraph), with a single `AgentState` carried between nodes.

---

## Quick start

### Prerequisites

- Docker, with RISC-V emulation enabled on x86/ARM hosts:
  ```bash
  docker run --privileged --rm tonistiigi/binfmt --install all
  ```
- Python 3.10+
- An API key for one of: Gemini, OpenAI, or OpenRouter.

### Install

```bash
git clone https://github.com/akifejaz/atesor-ai
cd atesor-ai
pip install -r requirements.txt
cp .env-example .env   # then add your API key
```

### Build the sandbox

```bash
python3 main.py --setup-only                       # Alpine (default)
python3 main.py --setup-only --platform debian     # Debian/Ubuntu
```

### Port a package

```bash
python3 main.py --repo https://github.com/madler/zlib --verbose
```

### Installation (from pre-built package)

A self-contained Debian package can be built for Ubuntu/Debian hosts - it
bundles a virtualenv and installs an `atesor-ai` launcher. See
[`packaging/deb/README.md`](packaging/deb/README.md):

```bash
packaging/deb/build_deb.sh                          # → dist/atesor-ai_*.deb
sudo apt-get install ./dist/atesor-ai_*.deb         # pulls docker/qemu deps
atesor-ai --help
atesor-ai --repo https://github.com/madler/zlib 
```

---


## Architecture (high-level)

Atesor AI is built in clear layers. A request enters at the top as a repo
URL and leaves at the bottom as verified `riscv64` artifacts plus a
replayable recipe. The middle is a **LangGraph** compiled `StateGraph` with
per-node conditional edges and an embedded build-fix subgraph - threading one
shared `AgentState` dataclass through every node.

```
   USER  --repo <url>  [--platform alpine | debian]
     |
     v
   +--------------------------------------------------------------+
   |  ORCHESTRATION   (main.py)                                   |
   |    (load .env -> verify keys -> provision riscv64 sandbox)   |
   +------------------------------+-------------------------------+
                  |
      recipe cache? --- HIT --->  render <pkg>_recipe.md  -->  EXIT
                  |
                  | MISS?   (carries one AgentState, mutated in place)
                  v
   +-------------------------------------------------------------+
   |  LANGGRAPH  (src/graph.py)   compiled StateGraph + subgraph |
   |                                                             |
   |   init_node                                                 |
   |     |  route_init_to_next(state)                            |
   |     v                                                       |
   |   planner_node  (LLM: TaskPlan + fallback)                  |
   |     |  route_planner_to_next(state)                         |
   |     v                                                       |
   |   scout_build_system  (0-LLM, reads build files)            |
   |     v                                                       |
   |   scout_deps  (0-LLM, reads deps)                           |
   |     v                                                       |
   |   scout_arch_issues  (0-LLM, counts arch patterns)          |
   |     v                                                       |
   |   scout_aggregator  (fan-in: default BuildPlan for          |
   |     |   well-known build systems, deps merged into setup)   |
   |     |  route_scout_aggregator_to_next(state)                |
   |     |----> scout_node  (LLM: BuildPlan, used when the       |
   |     |        aggregator defers: unknown/low-confidence      |
   |     |        build system or heavy arch-specific code)      |
   |     v                                                       |
   |   supervisor_node  (ZERO LLM - pure heuristic routing)      |
   |     |                                     |    |    |       |
   |     |  route_supervisor_to_next(state)    |    |    |       |
   |     |        |        |                   |    |    |       |
   |     |        |        |                   |    |    |       |
   |     v        v        v                   v    v    v       |
   |  planner  scout  build_fix_subgraph    finish  escalate     |
   |  (replan) (rescout)  (compiled subgraph)       (terminal)   |
   |                        |                                    |
   |                        v                                    |
   |                   build_node                                |
   |                     |  route_build_result(state)            |
   |                     v                                       |
   |               verify_node                                   |
   |                  |       route_verify_result(state)         |
   |                  |          |                   |           |
   |                  v          v                   v           |
   |              <verified>    fix_node          escalate       |
   |                  |          |  route_fix      (terminal)    |
   |                  |          |    |                          |
   |                  |          v    v                          |
   |                  | <fixed>  <can't fix>                     |
   |                  |    |         |                           |
   |                  v    v         v                           |
   |                 back to supervisor                          |
   |                                                             |
   +-------------------------------------------------------------+
       |                      |                      |
       v                      v                      v
   +-----------+        +----------+           +-----------+
   | FINISH    |        | ESCALATE |           | OUTPUTS   |
   | (LLM:     |        | (0-LLM)  |           | recipe.md |
   |  recipe)  |        |          |           | report    |
   +-----------+        +----------+           | state.json|
       |                                       | patches   |
       | self-learning: save few-shot examples +-----------+
       | + recipe cache
       v
   +----------------------------+
   |  MEMORY   (src/memory.py)  |
   |  examples + recipe_cache   |
   +----------------------------+
       |
       |-- few-shot prompts -->  PLANNER / SCOUT / FIXER
       |
       +-- fast-path -------->  next run's recipe-cache check  (top)
```

### Graph topology (LangGraph edges)

Every state has **per-node routing functions**. Each is unique to its source node and inspects real state fields:

- `route_init_to_next`: checks `state.build_status` → `planner_node` /
  `escalate_node`
- `route_planner_to_next`: checks `state.task_plan` → `scout_build_system`
  (entering the scout chain) / `escalate_node`
- `route_scout_aggregator_to_next`: checks `state.build_plan` →
  `supervisor_node` (heuristic plan built) / `scout_node` (defer to the
  LLM scout)
- `route_supervisor_to_next`: checks `state.build_status`, `state.task_plan`,
  `state.last_error_category`, `state.attempt_count` → 5 possible destinations
  (`planner_node`, `scout_node`, `build_fix_subgraph`, `finish_node`,
  `escalate_node`). A `SUCCESS` build always routes to `finish_node`,
  even at the attempt/cost ceiling.

The build-fix cycle is a compiled **subgraph** (`create_build_fix_subgraph()`)
with its own internal routing:

```
build_node ──> verify_node ──> __end__  (success, exits subgraph)
    ↑              │              ↑
    │              v              │
    └────── fix_node ─────────────┘  (retry, or exit if unfixable)
```

Routing inside the subgraph uses `route_build_result`, `route_verify_result`,
and `route_fix_result` - each inspecting build/error state directly.

### Three pillars

- **Scripted Operations Layer** (`src/scripted_ops.py`) - deterministic,
  zero-LLM repo inspection. Handles ~70% of analysis at zero cost.
- **LangGraph state machine** (`src/graph.py`) - compiled `StateGraph` with
  per-node routing, 13 nodes, a build-fix subgraph, and `@agent_node`-wrapped
  uniform error handling.
- **Platform abstraction** (`src/platforms.py`) - one `PlatformProfile` per
  distro. Adding a sandbox is a single `PROFILES` entry; the rest of the
  code stays distro-agnostic.

## Configuration

Edit `.env` (template in `.env-example`):

| Variable | Required when | Purpose |
|---|---|---|
| `LLM_PROVIDER` | always | `gemini` (default), `openai`, `openrouter` |
| `GOOGLE_API_KEY` | provider = `gemini` | |
| `OPENAI_API_KEY` | provider = `openai` | |
| `OPENROUTER_API_KEY` | provider = `openrouter` | |
| `LANGCHAIN_API_KEY` + `LANGCHAIN_TRACING_V2` | optional | LangSmith tracing |
| `ATESOR_PLATFORM` | optional | `alpine` / `debian` (overridden by `--platform`) |
| `ATESOR_CONTAINER` | optional | Override container name (overridden by `--container`) |
| `ATESOR_HOME` | optional | Base dir for runtime state (default `~/.local/share/atesor-ai`) |

NOTE: When using the pre-build binary (atesor-ai) the:
`.env` is loaded from the current directory first, then
`$ATESOR_HOME/.env` and `~/.config/atesor-ai/.env`, so an installed CLI
finds your keys regardless of the working directory.

Models are selected per agent role in `src/models.py` (`MODEL_CONFIG`). Each role has its own temperature - deterministic for Builder/Supervisor, slightly hotter for Fixer.

---

## Outputs

Runtime state is written under a **state home** - `$ATESOR_HOME` if set,
otherwise `~/.local/share/atesor-ai` for an installed CLI, or the repo's
`./workspace` when running from a source checkout. Below, `$WS` is that
`…/workspace` directory.

NOTE: When using the pre-build binary (atesor-ai) the state home is `$ATESOR_HOME` (default `~/.local/share/atesor-ai`) and does not include a `workspace` subdir. The CLI writes directly to `$ATESOR_HOME/output/`, `$ATESOR_HOME/logs/`, etc.

| Path | Content |
|---|---|
| `$WS/output/{repo}_recipe.md` | Final Markdown porting **recipe** (replayable) |
| `$WS/output/{repo}_report_*.md` | Detailed build report (per run) |
| `$WS/output/{repo}_state_*.json` | Full `AgentState` snapshot |
| `$WS/output/{repo}_patches_*/` | Patches applied during the run |
| `$WS/packages/{repo}-*-{platform}.zip` | Packaged artifact (with `--package`) |
| `$WS/logs/agent_{repo}.log` | Per-repo DEBUG log |
| `$WS/logs/agent-call_{repo}.log` | Full LLM call audit trail (prompt + response + cost) |
| `data/recipe_cache.json` | Successful builds, keyed by `{package: {sandbox: recipe}}` |
| `data/examples/*.json` | Few-shot examples per agent (auto-learning enabled) |

A cache hit short-circuits the pipeline - it re-renders `{repo}_recipe.md`
from the cached recipe and skips all LLM and Docker work - unless `--force`
is set. Cache entries are per-sandbox; Alpine and Debian builds populate
separate keys.

---

## Key features

- **Native RISC-V builds** - no cross-compilation, no surprises at deploy time.
- **Multiple Build Systems** - Make, CMake, Meson, autotools, Cargo, Go modules. More are coming.
- **Parallel batch runs** - `batch_test.py` allocates one container per worker (`atesor-ai-sandbox-w0..wN`) to avoid `apk`/`apt` lock contention.
- **Few-shot memory** - agents learn from past successes; up to 100 examples per agent, retrieved by keyword/regex.
- **Recipe cache** - successful builds are replayable and skip the LLM entirely.
- **ELF verification** - every produced binary is checked with `file` to confirm `RISC-V ELF`.
- **Cost-aware** - every LLM call is logged with token estimate and cost; hard cap at $1.00 per package.
- **Safe execution** - every shell command goes through a regex whitelist and runs inside the sandbox.

---

## Development

```bash
# Run the full test suite (PYTHONPATH=. is required)
PYTHONPATH=. pytest

# Single file or test case
PYTHONPATH=. pytest tests/test_graph_routing.py
PYTHONPATH=. pytest tests/test_state.py::TestState::test_add_error
```

Style: PEP 8, 79-char lines, Google-style docstrings, type hints on public APIs. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

---

## Contributing

Contributions are welcome - especially new platform profiles, additional few-shot examples from real porting runs, and bug reports for packages that fail in interesting ways. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, coding standards, and how to extend the system.

- [Open an issue](https://github.com/akifejaz/atesor-ai/issues)
- [License: MIT](LICENSE)

Built for the RISC-V community. Making the ecosystem catch up, one package at a time.
