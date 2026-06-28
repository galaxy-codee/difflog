"""Change classifier.

Three improvements over v1:
1. Message-first parsing — commit messages are checked before file rules,
   so a `fix:` commit on a new file correctly lands in Bug Fixes.
2. File-type defaults — extension-based fallbacks for docs, config, etc.
3. Multi-category mapping — a file can appear in multiple sections if
   different commits touched it in different ways.

Labels: breaking | feature | bugfix | refactor | chore | documentation
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from .semantic_parser import SemanticFileSummary

ChangeLabel = Literal["breaking", "feature", "bugfix", "refactor", "chore", "documentation"]
Confidence = Literal["high", "medium", "low"]

_BUGFIX_KEYWORDS = {
    "fix", "bug", "patch", "hotfix", "issue", "error", "crash", "regression",
    "typo", "correct", "resolve",
}
_FEATURE_KEYWORDS = {
    "feat", "feature", "add", "new", "implement", "introduce", "support", "enable",
}
_REFACTOR_KEYWORDS = {
    "refactor", "cleanup", "clean up", "restructure", "reorganize", "rename",
    "move", "extract", "simplify",
}
_CHORE_KEYWORDS = {
    "chore", "ci", "cd", "lint", "format", "bump", "version", "release",
    "dependency", "deps", "upgrade", "update", "comment", "test", "spec",
}
_DOC_KEYWORDS = {"doc", "docs", "readme", "documentation", "changelog"}

# Extension → default label when commit messages give no signal
_EXTENSION_DEFAULTS: dict[str, ChangeLabel] = {
    ".md":   "documentation",
    ".txt":  "documentation",
    ".rst":  "documentation",
    ".json": "chore",
    ".toml": "chore",
    ".yaml": "chore",
    ".yml":  "chore",
    ".lock": "chore",
    ".ini":  "chore",
    ".cfg":  "chore",
    ".env":  "chore",
}

_CHORE_PATH_FRAGMENTS = {
    ".github", "ci/", ".circleci", "Makefile", "Dockerfile", ".dockerignore",
    "requirements", "package.json", "package-lock.json", "poetry.lock",
    "pyproject.toml", ".pre-commit", "CHANGELOG", "LICENSE",
    ".gitignore", ".editorconfig",
}

_DOC_PATH_FRAGMENTS = {"README", "docs/", "documentation/"}


@dataclass
class ClassifiedChange:
    path: str
    label: ChangeLabel
    confidence: Confidence
    reason: str
    needs_llm: bool = False


@dataclass
class MultiClassifiedChange:
    """A file can carry multiple labels (e.g. feature + bugfix)."""
    path: str
    labels: list[ChangeLabel] = field(default_factory=list)
    confidence: Confidence = "low"
    reason: str = ""
    needs_llm: bool = False

    @property
    def label(self) -> ChangeLabel:
        """Primary label — highest-priority one present."""
        priority = ["breaking", "feature", "bugfix", "refactor", "documentation", "chore"]
        for p in priority:
            if p in self.labels:
                return p  # type: ignore[return-value]
        return self.labels[0] if self.labels else "chore"


class ChangeClassifier:
    """Classifies each SemanticFileSummary, supporting multi-label output."""

    def classify_all(
        self,
        summaries: list[SemanticFileSummary],
        commit_messages: list[str] | None = None,
    ) -> list[MultiClassifiedChange]:
        # Extract per-label commit scores once for the whole release window
        commit_hints = self._extract_commit_hints(commit_messages or [])
        return [self._classify_one(s, commit_hints) for s in summaries]

    # ------------------------------------------------------------------
    # Per-file classification
    # ------------------------------------------------------------------

    def _classify_one(
        self,
        summary: SemanticFileSummary,
        commit_hints: dict[ChangeLabel, int],
    ) -> MultiClassifiedChange:
        labels: list[ChangeLabel] = []

        # ── PASS 1: commit messages first (fixes the "train.py" bug) ──
        dominant = self._dominant_commit_hint(commit_hints)
        if dominant and commit_hints[dominant] >= 2:
            labels.append(dominant)

        # ── PASS 2: hard structural rules ─────────────────────────────

        # Doc path override
        if self._is_doc_path(summary.path):
            if "documentation" not in labels:
                labels.append("documentation")

        # Chore path override
        elif self._is_chore_path(summary.path):
            if "chore" not in labels:
                labels.append("chore")

        else:
            # Breaking: public symbol removed / renamed
            breaking_symbols = [
                sc for sc in summary.symbol_changes
                if sc.change in {"removed", "renamed"}
                and sc.kind in {"function", "class"}
                and not sc.name.startswith("_")
            ]
            if breaking_symbols:
                if "breaking" not in labels:
                    labels.append("breaking")

            # File deleted
            if summary.file_change_type == "deleted":
                if "breaking" not in labels:
                    labels.append("breaking")

            # New public symbols → feature (but only if commit didn't say fix)
            added_public = [
                sc for sc in summary.symbol_changes
                if sc.change == "added"
                and sc.kind in {"function", "class"}
                and not sc.name.startswith("_")
            ]
            if added_public and "bugfix" not in labels:
                if "feature" not in labels:
                    labels.append("feature")

            # New file → feature (but only if commit didn't say fix)
            if summary.file_change_type == "added" and "bugfix" not in labels:
                if "feature" not in labels:
                    labels.append("feature")

            # Only renames → refactor
            if (summary.symbol_changes
                    and all(sc.change == "renamed" for sc in summary.symbol_changes)
                    and not labels):
                labels.append("refactor")

        # ── PASS 3: extension-based default (fixes the "README" bug) ──
        if not labels:
            ext_default = _EXTENSION_DEFAULTS.get(summary.extension)
            if ext_default:
                labels.append(ext_default)

        # ── PASS 4: final fallback ─────────────────────────────────────
        needs_llm = False
        if not labels:
            labels.append("refactor")
            needs_llm = True

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_labels: list[ChangeLabel] = []
        for l in labels:
            if l not in seen:
                seen.add(l)
                unique_labels.append(l)

        confidence: Confidence = (
            "high" if len(unique_labels) == 1 and not needs_llm
            else "medium" if not needs_llm
            else "low"
        )

        return MultiClassifiedChange(
            path=summary.path,
            labels=unique_labels,
            confidence=confidence,
            reason=self._build_reason(unique_labels, commit_hints, summary),
            needs_llm=needs_llm,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reason(
        self,
        labels: list[ChangeLabel],
        commit_hints: dict[ChangeLabel, int],
        summary: SemanticFileSummary,
    ) -> str:
        parts = []
        if commit_hints.get(labels[0], 0) >= 2:
            parts.append("commit message signal")
        if summary.symbol_changes:
            parts.append(f"{len(summary.symbol_changes)} symbol change(s)")
        if summary.file_change_type in {"added", "deleted"}:
            parts.append(f"file {summary.file_change_type}")
        return "; ".join(parts) if parts else f"labels: {', '.join(labels)}"

    @staticmethod
    def _is_chore_path(path: str) -> bool:
        path_lower = path.lower()
        return any(f.lower() in path_lower for f in _CHORE_PATH_FRAGMENTS)

    @staticmethod
    def _is_doc_path(path: str) -> bool:
        return any(f.lower() in path.lower() for f in _DOC_PATH_FRAGMENTS)

    @staticmethod
    def _extract_commit_hints(messages: list[str]) -> dict[ChangeLabel, int]:
        hints: dict[ChangeLabel, int] = defaultdict(int)
        for msg in messages:
            lower = msg.lower()
            if lower.startswith("feat"):
                hints["feature"] += 2
            elif lower.startswith("fix"):
                hints["bugfix"] += 2
            elif lower.startswith("refactor"):
                hints["refactor"] += 2
            elif lower.startswith("chore") or lower.startswith("ci"):
                hints["chore"] += 2
            elif lower.startswith("docs") or lower.startswith("doc"):
                hints["documentation"] += 2
            elif lower.startswith("break") or "breaking change" in lower:
                hints["breaking"] += 2
            else:
                words = set(lower.split())
                for kw in _BUGFIX_KEYWORDS:
                    if kw in words or kw in lower:
                        hints["bugfix"] += 1
                for kw in _FEATURE_KEYWORDS:
                    if kw in words or kw in lower:
                        hints["feature"] += 1
                for kw in _REFACTOR_KEYWORDS:
                    if kw in words or kw in lower:
                        hints["refactor"] += 1
                for kw in _CHORE_KEYWORDS:
                    if kw in words or kw in lower:
                        hints["chore"] += 1
                for kw in _DOC_KEYWORDS:
                    if kw in words or kw in lower:
                        hints["documentation"] += 1
        return dict(hints)

    @staticmethod
    def _dominant_commit_hint(hints: dict[ChangeLabel, int]) -> ChangeLabel | None:
        if not hints:
            return None
        best = max(hints, key=lambda k: hints[k])
        return best if hints[best] > 0 else None  # type: ignore[return-value]