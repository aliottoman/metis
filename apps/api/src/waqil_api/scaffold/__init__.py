"""Framework-owned code that Metis writes into generated projects.

The modules under ``scaffold/appkit`` lead a double life: they are imported
and tested here as ordinary ``waqil_api`` code, and they are copied verbatim
into every generated project as a top-level ``appkit`` package. That is the
whole trick — one implementation, validated by this repo's suite, vendored so
the generated app stays standalone. Internal imports are therefore relative
(``from .config import …``), which resolves identically in both homes, and
nothing in ``appkit`` may import from ``waqil_api``.
"""
