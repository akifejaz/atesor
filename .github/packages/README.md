# Package lists for `batch_test.py`

`batch_test.py` reads its workload from one of the JSON files in this
directory. The default is `smoke.json` (fast sanity check); pass
`--list full` for the full catalog.

## Files

| File         | Size       | When to use                                          |
| ------------ | ---------- | ---------------------------------------------------- |
| `smoke.json` | ~9 pkgs    | Default. Fast Go + C mix to catch regressions early. |
| `full.json`  | ~170 pkgs  | Nightly / pre-release full sweep.                    |

Add your own lists by dropping a new `*.json` file here that matches the
schema below — `batch_test.py --list <name>` will pick it up
(`<name>` is resolved against this directory, with `.json` appended if
missing).

## Schema (v1)

```json
{
  "$schema_version": 1,
  "description": "Free-form text shown in CLI banner; JSON has no comments.",
  "defaults": {
    "max_attempts": 5,
    "timeout_seconds": 3600
  },
  "packages": [
    {
      "name":  "anew",
      "url":   "https://github.com/tomnomnom/anew",
      "lang":  "go",
      "group": "tomnomnom",
      "tags":  ["cli", "small"]
    }
  ]
}
```

### Required fields per package

| Field  | Type   | Notes                                                  |
| ------ | ------ | ------------------------------------------------------ |
| `name` | string | Unique within the file. Used for log paths and CLI.    |
| `url`  | string | Clonable HTTPS git URL.                                |

### Optional fields per package

| Field   | Type           | Purpose                                                                   |
| ------- | -------------- | ------------------------------------------------------------------------- |
| `lang`  | string         | `go` \| `c` \| `cpp` \| `rust`. Enables `--lang <x>` filtering.            |
| `group` | string         | Logical bucket (replaces the old `# --- Batch N ---` comments).           |
| `tags`  | list\[string\] | Free-form labels for `--tag <x>` filtering.                                |

### Top-level fields

| Field             | Required | Purpose                                                                 |
| ----------------- | -------- | ----------------------------------------------------------------------- |
| `$schema_version` | yes      | Integer. Bump if you change required fields.                            |
| `description`     | no       | Shown in the CLI banner; helps operators know which list is running.    |
| `defaults`        | no       | Batch-wide knobs. Currently `max_attempts`, `timeout_seconds`.          |
| `packages`        | yes      | List of package objects (see above).                                    |

## Style rules for new entries

1. **One package per line** in `packages` — keep diffs reviewable.
2. **Sort by `group`, then alphabetically by `name`** within a group.
3. **`name` must be lowercase**, kebab-case allowed. It becomes the log
   filename and the recipe cache key.
4. **`url` must be the canonical upstream**, not a fork — the recipe
   cache is keyed on the repo identity.
5. **Tag SIMD-heavy / autotools-fragile packages** so we can isolate
   them quickly with `--tag simd` etc. when triaging.
6. **Do not add per-package timeout overrides** — bump `defaults` or
   move the slow package to its own list instead.

## CLI quick reference

```bash
# Default: smoke list
python batch_test.py

# Pick a list explicitly
python batch_test.py --list full
python batch_test.py --list smoke

# Run a subset by name (positional args are name filters)
python batch_test.py --list full anew gron zlib
```
