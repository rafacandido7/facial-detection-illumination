# Code Conventions

**Updated:** 2026-05-13
**Language:** Python 3.12 | Single file: `src/main.py`

---

## Naming

| Scope | Convention | Example |
|-------|-----------|---------|
| Constants | `SCREAMING_SNAKE_CASE` | `MODEL_PATH` |
| Functions/variables | `snake_case` | `run_pipeline` |
| Private helpers | `_underscore_prefix` | `_banner` |
| Domain functions | Portuguese names | `aplicar_clahe` |
| Utility functions | English names | `run_pipeline`, `_banner` |

## Return Conventions

Detection functions return consistent 2-tuples: `(boxes, elapsed_ms)`

## Error Handling

- Fatal errors: `print(..., file=sys.stderr)` + `sys.exit(1)`
- No try/except blocks
- No custom exception classes

## Output

- All output via `print()`
- User-facing messages in Portuguese
- CLI built with `argparse`, optional positional argument

## Tooling

- No formatter configured (no `.flake8`, `ruff.toml`, `pyproject.toml`, `.pylintrc`)
- No linter
- Academic MVP — style is informal but consistent within file

---

*Conventions inferred from single source file. No enforced style tooling.*
