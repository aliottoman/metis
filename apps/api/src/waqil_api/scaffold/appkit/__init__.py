"""Metis appkit — framework-owned runtime for a generated application.

These files are written by Metis, not by the model that builds the rest of
the project. They vendor the integrations Metis has already verified — OCI
Responses, lazy configuration, money arithmetic, upload handling — so a build
spends its nondeterminism on the application, never on infrastructure.

Do not hand-edit: Metis refuses model writes under appkit/ and a scaffold
upgrade replaces the directory wholesale.
"""

SCAFFOLD_VERSION = "0.1.0"
