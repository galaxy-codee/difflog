# difflog

> Turn git history into a structured, AI-summarized changelog — in one command.

`difflog` analyzes what changed between two git refs at a _semantic_ level — not just "these lines changed" — and produces a structured `CHANGELOG.md`.

```
difflog --from v1.0.0 --to v1.2.0 --output CHANGELOG.md
difflog --from main --to dev --format markdown
difflog --from HEAD~10 --to HEAD --no-llm   # offline fallback
```

---

## Features

- **AST-based Python analysis** — detects added/removed/renamed functions and classes, not just line diffs
- **Message-first classification** — commit messages (`fix:`, `feat:`, `chore:`) are checked before file rules, so intent always wins
- **File-type defaults** — `.md/.rst/.txt` → Documentation, `.json/.toml/.yaml` → Chore, even without a commit prefix
- **Multi-label support** — a file touched by both a feature and a fix appears under both sections
- **Token-efficient LLM calls** — sends structured semantic context, not raw diffs
- **Keep a Changelog format** — sections: Breaking Changes · Features · Bug Fixes · Refactors · Documentation · Chores
- **Offline mode** — `--no-llm` produces rule-based summaries with no API calls
- **JSON output** — `--format json` for programmatic consumption

---

## Installation

```bash
# From source (recommended during development)
pip install gitpython openai rich click

# Install difflog itself
pip install -e .

# Set your OpenAI API key
set OPENAI_API_KEY=sk-...        # Windows
export OPENAI_API_KEY=sk-...     # Mac/Linux
```

**Requirements:** Python 3.10+, git

---

## Usage

```bash
# Basic — compare two version tags
difflog --from v1.0.0 --to v1.2.0

# From the very first commit
difflog --from f503868 --to HEAD

# Write to file
difflog --from v1.0.0 --to v1.2.0 --output CHANGELOG.md

# Compare branches
difflog --from main --to feature/my-branch

# Offline mode (no API key needed)
difflog --from HEAD~10 --to HEAD --no-llm

# Verbose: show per-file classification table
difflog --from v1.0.0 --to v1.2.0 --verbose

# JSON output
difflog --from v1.0.0 --to v1.2.0 --format json

# Custom version label in the changelog header
difflog --from v1.0.0 --to HEAD --version-label "2.0.0-beta"

# Point at a different repo
difflog --from v1.0.0 --to v1.1.0 --repo /path/to/other/repo
```

---

## Architecture

```
difflog/
├── cli.py              # Click CLI — orchestrates the pipeline
├── git_interface.py    # gitpython wrapper: commits, diffs, file stats
├── semantic_parser.py  # AST analysis (Python) + heuristics (all others)
├── classifier.py       # Message-first, multi-label change classifier
├── llm_summarizer.py   # OpenAI API integration (token-efficient)
└── renderer.py         # CHANGELOG.md (Keep a Changelog format)
```

### Pipeline

```
git refs
    │
    ▼
GitInterface          → DiffResult (commits + FileDiffs)
    │
    ▼
SemanticParser        → SemanticFileSummary[] (symbol changes, significance)
    │
    ▼
ChangeClassifier      → MultiClassifiedChange[] (multi-label per file)
    │
    ▼
LLMSummarizer         → LLMSummary (per-file one-liners + release paragraph)
    │
    ▼
ChangelogRenderer     → CHANGELOG.md / JSON
```

### Classification logic

The classifier runs in four passes per file:

1. **Commit messages first** — `fix:` beats file rules; a new file with a `fix:` commit goes to Bug Fixes, not Features
2. **Structural rules** — public symbol removed → Breaking; new file → Feature; etc.
3. **Extension defaults** — `.md/.txt/.rst` → Documentation; `.json/.toml/.yaml` → Chore
4. **Fallback** — `refactor` with LLM flag set

### Change labels

| Label | Triggered by |
|---|---|
| `breaking` | Public function/class removed or renamed; file deleted |
| `feature` | New file or new public symbol — unless commit says `fix:` |
| `bugfix` | `fix:` commit prefix; bug/patch/hotfix keywords |
| `refactor` | Only renames; `refactor:` commit prefix |
| `documentation` | `.md/.rst/.txt` files; `docs/` path; `docs:` prefix |
| `chore` | Config/infra paths (CI, lockfiles, `.gitignore`, etc.) |

A single file can appear under multiple sections if different commits touched it in different ways.

---

## Running tests

```bash
pip install pytest pytest-mock
pytest tests/ -v
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes (unless `--no-llm`) | OpenAI API key |

---

## Extending difflog

- **Add a new language parser** — extend `SemanticParser._parse_generic` with an `elif fd.extension == ".go":` branch
- **Change the LLM model** — edit `LLMSummarizer.DEFAULT_MODEL` in `llm_summarizer.py`
- **Add output formats** — add a branch in `cli.py` and a new renderer method
- **Tune classification rules** — edit `_BUGFIX_KEYWORDS`, `_EXTENSION_DEFAULTS`, etc. in `classifier.py`