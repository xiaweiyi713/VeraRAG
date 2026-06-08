# Security Policy

## Supported Versions

VeraRAG is currently pre-1.0. Security fixes target the `main` branch.

## Reporting a Vulnerability

Please do not open a public issue for secrets exposure, authentication bypasses, code execution, path traversal, or unsafe file handling. Report privately to the maintainer listed in the repository metadata, or open a private security advisory on GitHub if available.

Include:

- affected commit or release;
- reproduction steps;
- impact and affected component;
- whether any API keys, local files, or user data may be exposed.

## Secret Handling

- Do not commit `.verarag_key`, SQLite databases, environment files, API keys, or raw private corpora.
- Web UI API keys are encrypted locally, but this is not a hosted multi-tenant security boundary.
- Demo and benchmark outputs may include model responses and document excerpts; review them before publishing.

