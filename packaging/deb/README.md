# Atesor AI — Debian package

Builds a self-contained `.deb` for **Ubuntu 22.04 / Debian-based** systems
(targets `python3.10`, architecture `amd64`).

## Layout when installed

```
/opt/atesor-ai/app/     application code + Dockerfiles + seed data
/opt/atesor-ai/venv/    bundled virtualenv with all Python deps
/usr/bin/atesor-ai      launcher (sets PYTHONPATH, runs the venv python)
```

Runtime state (workspace, logs, recipe cache, learned examples) is written
to `$ATESOR_HOME` if set, otherwise `~/.local/share/atesor-ai` — never into
`/opt`.

## Build

```bash
packaging/deb/build_deb.sh            # -> dist/atesor-ai_<version>_amd64.deb
```

Requires Docker on the build host. The package is built inside a clean
`ubuntu:22.04` container so the bundled virtualenv matches the target
interpreter.

## Validate

```bash
packaging/deb/validate_deb.sh         # installs + runs the CLI in a clean container
```

## Install (on a target machine)

```bash
sudo apt-get install ./atesor-ai_<version>_amd64.deb
```

`apt` resolves the declared dependencies: `python3`, a Docker engine
(`docker.io` | `docker-ce`), `qemu-user-static`, and `binfmt-support`.

## Prerequisites that cannot be bundled

Atesor AI orchestrates Docker; these must exist on the target host:

- a running Docker daemon (and your user in the `docker` group),
- `qemu-user-static` + `binfmt-support` for `riscv64` emulation,
- an LLM API key (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`).

## Notes

- Targeting another distro/python version requires rebuilding with the
  matching base image (e.g. `packaging/deb/build_deb.sh debian:12`).
- The runtime dependency set is pinned in `requirements-runtime.txt`
  (dev-only tools are excluded).
