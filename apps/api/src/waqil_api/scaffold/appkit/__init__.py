"""Metis appkit — framework-owned runtime for a generated application.

These files are written by Metis, not by the model that builds the rest of
the project. They vendor the integrations Metis has already verified — OCI
Responses, lazy configuration, money arithmetic, upload handling — so a build
spends its nondeterminism on the application, never on infrastructure.

Do not hand-edit: Metis refuses model writes under appkit/ and an approved
scaffold upgrade replaces the canonical files it owns.
"""

SCAFFOLD_VERSION = "0.2.0"
