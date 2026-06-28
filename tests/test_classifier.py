"""Tests for classifier.py"""

from __future__ import annotations

import pytest

from difflog.classifier import ChangeClassifier
from difflog.semantic_parser import SemanticFileSummary, SymbolChange


def make_summary(
    path="foo.py",
    ext=".py",
    file_change_type="modified",
    lines_added=5,
    lines_deleted=2,
    symbol_changes=None,
    notes=None,
) -> SemanticFileSummary:
    s = SemanticFileSummary(path, ext, file_change_type, lines_added, lines_deleted)
    s.symbol_changes = symbol_changes or []
    s.notes = notes or []
    return s


CLF = ChangeClassifier()


# ------------------------------------------------------------------
# Chore detection by path
# ------------------------------------------------------------------

class TestChoreDetection:
    @pytest.mark.parametrize("path", [
        ".github/workflows/ci.yml",
        "Makefile",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        "CHANGELOG.md",
        ".gitignore",
    ])
    def test_infra_path_is_chore(self, path):
        s = make_summary(path=path, ext=".yml")
        result = CLF.classify_all([s])[0]
        assert result.label == "chore"
        assert result.confidence == "high"


# ------------------------------------------------------------------
# Breaking changes
# ------------------------------------------------------------------

class TestBreakingDetection:
    def test_public_function_removed_is_breaking(self):
        s = make_summary(symbol_changes=[
            SymbolChange("function", "removed", "do_thing")
        ])
        result = CLF.classify_all([s])[0]
        assert result.label == "breaking"

    def test_private_function_removed_is_not_breaking(self):
        s = make_summary(symbol_changes=[
            SymbolChange("function", "removed", "_internal")
        ])
        result = CLF.classify_all([s])[0]
        assert result.label != "breaking"

    def test_deleted_file_is_breaking(self):
        s = make_summary(file_change_type="deleted")
        result = CLF.classify_all([s])[0]
        assert result.label == "breaking"

    def test_public_function_renamed_is_breaking(self):
        s = make_summary(symbol_changes=[
            SymbolChange("function", "renamed", "new_name", old_name="old_name")
        ])
        result = CLF.classify_all([s])[0]
        assert result.label == "breaking"


# ------------------------------------------------------------------
# Feature detection
# ------------------------------------------------------------------

class TestFeatureDetection:
    def test_new_file_is_feature(self):
        s = make_summary(file_change_type="added")
        result = CLF.classify_all([s])[0]
        assert result.label == "feature"

    def test_new_public_function_is_feature(self):
        s = make_summary(symbol_changes=[
            SymbolChange("function", "added", "new_func")
        ])
        result = CLF.classify_all([s])[0]
        assert result.label == "feature"

    def test_new_private_function_not_feature(self):
        s = make_summary(symbol_changes=[
            SymbolChange("function", "added", "_private")
        ])
        result = CLF.classify_all([s])[0]
        # private additions don't guarantee feature — classifier should NOT force feature
        assert result.label != "feature" or result.confidence != "high"


# ------------------------------------------------------------------
# Commit message hints
# ------------------------------------------------------------------

class TestCommitHints:
    def test_fix_prefix_hints_bugfix(self):
        s = make_summary()   # no strong symbol signals
        hints = CLF._extract_commit_hints(["fix: correct off-by-one error", "fix: null check"])
        dominant = CLF._dominant_commit_hint(hints)
        assert dominant == "bugfix"

    def test_feat_prefix_hints_feature(self):
        hints = CLF._extract_commit_hints(["feat: add new dashboard widget"])
        assert hints["feature"] >= 2

    def test_chore_prefix_hints_chore(self):
        hints = CLF._extract_commit_hints(["chore: bump version to 2.0"])
        assert hints["chore"] >= 2

    def test_empty_messages_returns_zero_hints(self):
        hints = CLF._extract_commit_hints([])
        assert all(v == 0 for v in hints.values())


# ------------------------------------------------------------------
# Refactor detection
# ------------------------------------------------------------------

class TestRefactorDetection:
    def test_only_renames_is_refactor(self):
        s = make_summary(symbol_changes=[
            SymbolChange("function", "renamed", "new_name", old_name="_old_name")
        ])
        # _old_name is private, so not breaking; should land as refactor
        result = CLF.classify_all([s])[0]
        assert result.label == "refactor"


# ------------------------------------------------------------------
# Low-confidence fallback
# ------------------------------------------------------------------

class TestLowConfidenceFallback:
    def test_no_signal_marks_needs_llm(self):
        s = make_summary()  # no symbols, no path hints, no commit hints
        result = CLF.classify_all([s])[0]
        assert result.needs_llm is True or result.confidence in {"low", "medium"}