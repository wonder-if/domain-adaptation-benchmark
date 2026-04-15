# AGENTS

## Project

`dabench` is an installable Python package for domain adaptation benchmark support code.

Current focus:

- dataset download and verification
- loading data through libraries such as `datasets`, `torch`, and `transformers`
- evaluation-side utilities for benchmark workflows

Current non-focus:

- CLI design or expansion, unless it is required to preserve an existing user-facing command

Non-goals for now:

- method training code
- model zoo management
- environment-specific setup instructions

## Development Rules

- Keep package code under `src/dabench/`
- Keep development-only files outside the package
- Prefer small, composable Python modules over large scripts
- Treat shell scripts as thin wrappers, not the main implementation
- Prefer explicit paths and explicit configuration over hidden defaults
- Prefer a layered, deterministic loading flow: manifest/config provides the source of truth, storage prepares local files, data loads a single domain or split from a concrete path, and higher layers assemble tasks or suites on top
- Avoid guesswork or inference when the caller can pass the value explicitly; do not add aliases, auto-detection, filtering, or normalization unless it is required by a manifest-backed contract
- Preserve compatibility when replacing an existing user-facing command
- Keep dataset-specific logic inside dataset-specific modules

## Data And Config

- Repo-level development config may live under `config/`
- Generated analysis artifacts may live under `reports/`
- Do not put machine-specific environment instructions into repo docs; they belong in external memories

## Validation

- For package changes, verify import, CLI help, and editable install when relevant
- For dataset-loading changes, verify against real local data rather than only unit-style stubs
- Prefer writing reports or inspection outputs that make the dataset state easy to understand
