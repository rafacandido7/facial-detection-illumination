# Testing

**Updated:** 2026-05-13

---

## Status

**No tests exist.** Academic MVP — no test infrastructure established.

## Coverage

| Area | Status |
|------|--------|
| Unit tests | None |
| Integration tests | None |
| Test framework | Not installed |
| Test config | None (`pytest.ini`, `tox.ini` absent) |

## Findings

- No `test_*.py` or `*_test.py` files anywhere
- No `tests/` directory
- `pytest` not in `requirements.txt`
- No CI pipeline

## Notes

Project is a research script comparing face detection algorithms. Correctness validated visually via side-by-side visualization output, not automated tests.
