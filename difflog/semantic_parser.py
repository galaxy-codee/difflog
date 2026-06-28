"""Semantic parser.

For Python files: uses ast to detect added/removed/renamed functions,
classes, and top-level imports.

For all other file types: falls back to line-level diff stats and a
lightweight heuristic scan of the unified diff.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field
from typing import Literal

from .git_interface import FileDiff


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------

SymbolKind = Literal["function", "class", "method", "import"]
ChangeKind = Literal["added", "removed", "modified", "renamed"]


@dataclass
class SymbolChange:
    kind: SymbolKind
    change: ChangeKind
    name: str
    old_name: str | None = None   # for renames
    context: str = ""             # e.g. enclosing class for methods


@dataclass
class SemanticFileSummary:
    path: str
    extension: str
    file_change_type: str          # from FileDiff
    lines_added: int
    lines_deleted: int
    symbol_changes: list[SymbolChange] = field(default_factory=list)
    significance: Literal["high", "medium", "low"] = "low"
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

class SemanticParser:
    """Converts a list of FileDiffs into SemanticFileSummarys."""

    RENAME_THRESHOLD = 0.6   # similarity ratio for treating symbols as renamed

    def parse(self, file_diffs: list[FileDiff]) -> list[SemanticFileSummary]:
        summaries = []
        for fd in file_diffs:
            if fd.extension == ".py" and not fd.is_binary:
                summary = self._parse_python(fd)
            else:
                summary = self._parse_generic(fd)
            summary.significance = self._score_significance(summary)
            summaries.append(summary)
        return summaries

    # ------------------------------------------------------------------
    # Python-specific path
    # ------------------------------------------------------------------

    def _parse_python(self, fd: FileDiff) -> SemanticFileSummary:
        summary = SemanticFileSummary(
            path=fd.path,
            extension=fd.extension,
            file_change_type=fd.change_type,
            lines_added=fd.lines_added,
            lines_deleted=fd.lines_deleted,
        )

        old_src, new_src = self._split_diff_sources(fd.raw_diff)
        old_symbols = self._extract_symbols(old_src, fd.path)
        new_symbols = self._extract_symbols(new_src, fd.path)

        summary.symbol_changes = self._diff_symbols(old_symbols, new_symbols)

        if fd.change_type == "added":
            summary.notes.append("New file added")
        elif fd.change_type == "deleted":
            summary.notes.append("File deleted")

        return summary

    def _split_diff_sources(self, raw_diff: str) -> tuple[str, str]:
        """Reconstruct approximate old and new file contents from a unified diff."""
        old_lines: list[str] = []
        new_lines: list[str] = []

        for line in raw_diff.splitlines():
            if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
                continue
            if line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith("+"):
                new_lines.append(line[1:])
            else:
                # context line — present in both
                old_lines.append(line[1:] if line.startswith(" ") else line)
                new_lines.append(line[1:] if line.startswith(" ") else line)

        return "\n".join(old_lines), "\n".join(new_lines)

    def _extract_symbols(self, source: str, path: str) -> dict[str, SymbolKind]:
        """Parse Python source and return {qualified_name: kind}."""
        symbols: dict[str, SymbolKind] = {}
        if not source.strip():
            return symbols
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return symbols

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                # Determine if it's a method (parent is a class)
                symbols[node.name] = "function"
            elif isinstance(node, ast.ClassDef):
                symbols[node.name] = "class"
                for item in node.body:
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                        symbols[f"{node.name}.{item.name}"] = "method"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    symbols[alias.asname or alias.name] = "import"
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    symbols[alias.asname or alias.name] = "import"

        return symbols

    def _diff_symbols(
        self,
        old: dict[str, SymbolKind],
        new: dict[str, SymbolKind],
    ) -> list[SymbolChange]:
        changes: list[SymbolChange] = []
        old_names = set(old)
        new_names = set(new)

        added = new_names - old_names
        removed = old_names - new_names

        # Detect renames: pair up removed/added symbols of the same kind
        # with high name-similarity
        matched_added: set[str] = set()
        matched_removed: set[str] = set()

        for r in list(removed):
            for a in list(added):
                if old.get(r) == new.get(a):  # same kind
                    ratio = difflib.SequenceMatcher(None, r, a).ratio()
                    if ratio >= self.RENAME_THRESHOLD:
                        changes.append(
                            SymbolChange(
                                kind=new[a],
                                change="renamed",
                                name=a,
                                old_name=r,
                            )
                        )
                        matched_added.add(a)
                        matched_removed.add(r)
                        break  # one rename partner per removed symbol

        for name in added - matched_added:
            changes.append(SymbolChange(kind=new[name], change="added", name=name))

        for name in removed - matched_removed:
            changes.append(SymbolChange(kind=old[name], change="removed", name=name))

        # Symbols present in both — count as "modified" if the file itself changed
        # (we can't easily detect body changes without full source, so we skip
        # this for now to avoid false positives)

        return changes

    # ------------------------------------------------------------------
    # Generic (non-Python) path
    # ------------------------------------------------------------------

    def _parse_generic(self, fd: FileDiff) -> SemanticFileSummary:
        summary = SemanticFileSummary(
            path=fd.path,
            extension=fd.extension,
            file_change_type=fd.change_type,
            lines_added=fd.lines_added,
            lines_deleted=fd.lines_deleted,
        )

        if fd.is_binary:
            summary.notes.append("Binary file — content diff not available")
            return summary

        # Light heuristics on the diff text
        if fd.extension in {".json", ".yaml", ".yml", ".toml", ".ini", ".env"}:
            summary.notes.append("Configuration file changed")
        elif fd.extension in {".md", ".rst", ".txt"}:
            summary.notes.append("Documentation file changed")
        elif fd.extension in {".ts", ".tsx", ".js", ".jsx"}:
            summary.notes.append("JavaScript/TypeScript file changed")
        elif fd.extension in {".go"}:
            summary.notes.append("Go source file changed")
        elif fd.extension in {".java"}:
            summary.notes.append("Java source file changed")
        elif fd.extension in {".sh", ".bash"}:
            summary.notes.append("Shell script changed")

        return summary

    # ------------------------------------------------------------------
    # Significance scoring
    # ------------------------------------------------------------------

    def _score_significance(self, s: SemanticFileSummary) -> Literal["high", "medium", "low"]:
        if s.file_change_type in {"added", "deleted"}:
            return "high"

        high_symbol_changes = sum(
            1 for sc in s.symbol_changes
            if sc.change in {"removed", "renamed"} and sc.kind in {"function", "class"}
        )
        if high_symbol_changes > 0:
            return "high"

        total_lines = s.lines_added + s.lines_deleted
        if total_lines >= 50 or len(s.symbol_changes) >= 3:
            return "medium"

        return "low"