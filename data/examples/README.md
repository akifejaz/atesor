# How to Add New Few-Shot Examples

This guide explains how to add new examples to improve agent performance.

## Directory Structure

```
data/examples/
├── scout_examples.json      # Build plan generation examples
├── fixer_examples.json      # Error resolution examples
├── builder_examples.json    # Build execution examples
└── supervisor_examples.json # Routing decision examples
```

## Example Format

Each example file follows this JSON structure:

```json
{
  "version": "1.0",
  "description": "Description of this examples file",
  "examples": [
    {
      "id": "unique-id",
      "name": "Human-readable name",
      "tags": ["relevant", "tags", "for", "matching"],
      "context": {
        // Context that triggers this example
        "build_system": "go",
        "has_main": true,
        "main_path": ".",
        "module_dir": ""
      },
      "expected_output": {
        // What the agent should produce
        "build_system": "go",
        "phases": [...],
        "notes": ["..."]
      },
      "reasoning": "Why this solution works"
    }
  ]
}
```

## Adding Scout Examples (Build Plans)

Scout examples teach the agent how to create build plans for different project types.

```json
{
  "id": "scout-XXX",
  "name": "Descriptive Name",
  "tags": ["go", "tag2", "tag3"],
  "context": {
    "repo_name": "example-repo",
    "build_system": "go",
    "has_main": true,
    "main_path": "cmd/app",
    "module_dir": "",
    "dependencies": ["lib1", "lib2"]
  },
  "repo_structure": "Brief description of file structure",
  "expected_output": {
    "build_system": "go",
    "build_system_confidence": 0.95,
    "module_dir": "",
    "phases": [
      {
        "id": 1,
        "name": "install_dependencies",
        "commands": ["apk update", "apk add go"],
        "can_parallelize": false,
        "expected_duration": "30s"
      },
      {
        "id": 2,
        "name": "build",
        "commands": ["go build -buildvcs=false -v -o app ./cmd/app"],
        "can_parallelize": false,
        "expected_duration": "2m"
      }
    ],
    "total_estimated_duration": "3m",
    "notes": ["Key observations about this build"]
  },
  "reasoning": "Explain WHY this build plan is correct"
}
```

## Adding Fixer Examples (Error Resolution)

Fixer examples teach the agent how to resolve common build errors.

```json
{
  "id": "fixer-XXX",
  "name": "Error Type Description",
  "tags": ["go", "error-category", "tool"],
  "error_context": {
    "category": "DEPENDENCY",
    "error_message": "Full error message from build",
    "failed_command": "go build ."
  },
  "solution": {
    "analysis": "Root cause analysis",
    "strategy": "High-level fix approach",
    "actions": [
      {
        "type": "command",
        "command": "apk add missing-package"
      }
    ],
    "updated_build_command": "go build -buildvcs=false ."
  },
  "reasoning": "Why this fix works"
}
```

## Tips for Good Examples

1. **Be Specific**: Include actual error messages and real build commands
2. **Tag Well**: Use relevant tags that match the context
3. **Explain Reasoning**: The "reasoning" field helps the LLM understand WHY
4. **Keep It Real**: Use examples from actual porting sessions
5. **Cover Edge Cases**: Examples for Go modules in subdirectories, CGO, etc.

## Testing New Examples

```python
from src.memory import format_few_shot_examples

# Test Scout example
context = {
    'build_system': 'go',
    'has_main': True,
    'main_path': 'cmd/app',
    'module_dir': ''
}
examples = format_few_shot_examples('scout', context, max_examples=2)
print(examples)
```

## Example Relevance Scoring

The memory system scores examples based on:
- Build system match (+0.5)
- Error message tag matches (+0.2 per match)
- Main path match (+0.1)
- Module directory match (+0.15)

Higher scored examples are shown first in prompts.
