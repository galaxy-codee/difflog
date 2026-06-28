"""Tests for semantic_parser.py"""

from __future__ import annotations

import pytest

from difflog.git_interface import FileDiff
from difflog.semantic_parser import SemanticParser, SemanticFileSummary, SymbolChange


def make_fd(
    path="foo.py",
    ext=".py",
    change_type="modified",
    lines_added=0,
    lines_deleted=0,
    raw_diff="",
    is_binary=False,
) -> FileDiff:
    return FileDiff(path, ext, change_type, None, lines_added, lines_deleted, raw_diff, is_binary)


PARSER = SemanticParser()


# ------------------------------------------------------------------
# Symbol extraction
# ------------------------------------------------------------------

class TestExtractSymbols:
    def test_functions_detected(self):
        src = "def foo(): pass\ndef _bar(): pass\n"
        symbols = PARSER._extract_symbols(src, "f.py")
        assert "foo" in symbols
        assert "_bar" in symbols
        assert symbols["foo"] == "function"

    def test_class_detected(self):
        src = "class MyClass:\n    def method(self): pass\n"
        symbols = PARSER._extract_symbols(src, "f.py")
        assert "MyClass" in symbols
        assert symbols["MyClass"] == "class"
        assert "MyClass.method" in symbols
        assert symbols["MyClass.method"] == "method"

    def test_imports_detected(self):
        src = "import os\nfrom pathlib import Path as P\n"
        symbols = PARSER._extract_symbols(src, "f.py")
        assert "os" in symbols
        assert "P" in symbols

    def test_invalid_syntax_returns_empty(self):
        symbols = PARSER._extract_symbols("def broken(:", "f.py")
        assert symbols == {}

    def test_empty_source_returns_empty(self):
        symbols = PARSER._extract_symbols("", "f.py")
        assert symbols == {}


# ------------------------------------------------------------------
# Diff symbol diffing (added / removed / renamed)
# ------------------------------------------------------------------

class TestDiffSymbols:
    def test_added_symbol(self):
        old = {"foo": "function"}
        new = {"foo": "function", "bar": "function"}
        changes = PARSER._diff_symbols(old, new)
        names = {sc.name for sc in changes}
        assert "bar" in names
        assert any(sc.change == "added" for sc in changes if sc.name == "bar")

    def test_removed_symbol(self):
        old = {"foo": "function", "bar": "function"}
        new = {"foo": "function"}
        changes = PARSER._diff_symbols(old, new)
        assert any(sc.change == "removed" and sc.name == "bar" for sc in changes)

    def test_renamed_symbol(self):
        # "process_data" → "process_dataset" — high similarity
        old = {"process_data": "function"}
        new = {"process_dataset": "function"}
        changes = PARSER._diff_symbols(old, new)
        assert any(sc.change == "renamed" for sc in changes)

    def test_no_rename_across_kinds(self):
        # A class and a function with similar names should NOT be paired as rename
        old = {"Foo": "class"}
        new = {"foo": "function"}
        changes = PARSER._diff_symbols(old, new)
        kinds = {sc.change for sc in changes}
        assert "renamed" not in kinds


# ------------------------------------------------------------------
# Significance scoring
# ------------------------------------------------------------------

class TestSignificance:
    def test_deleted_file_is_high(self):
        s = SemanticFileSummary("x.py", ".py", "deleted", 0, 0)
        assert PARSER._score_significance(s) == "high"

    def test_added_file_is_high(self):
        s = SemanticFileSummary("x.py", ".py", "added", 10, 0)
        assert PARSER._score_significance(s) == "high"

    def test_removed_public_function_is_high(self):
        s = SemanticFileSummary("x.py", ".py", "modified", 0, 5)
        s.symbol_changes = [SymbolChange("function", "removed", "do_thing")]
        assert PARSER._score_significance(s) == "high"

    def test_small_change_is_low(self):
        s = SemanticFileSummary("x.py", ".py", "modified", 3, 1)
        assert PARSER._score_significance(s) == "low"

    def test_large_diff_is_medium(self):
        s = SemanticFileSummary("x.py", ".py", "modified", 40, 20)
        assert PARSER._score_significance(s) == "medium"


# ------------------------------------------------------------------
# Generic (non-Python) parsing
# ------------------------------------------------------------------

class TestGenericParsing:
    def test_binary_file_noted(self):
        fd = make_fd("image.png", ".png", "modified", is_binary=True)
        result = PARSER._parse_generic(fd)
        assert any("Binary" in note for note in result.notes)

    def test_json_file_noted(self):
        fd = make_fd("config.json", ".json", "modified")
        result = PARSER._parse_generic(fd)
        assert any("Configuration" in note for note in result.notes)

    def test_markdown_file_noted(self):
        fd = make_fd("README.md", ".md", "modified")
        result = PARSER._parse_generic(fd)
        assert any("Documentation" in note for note in result.notes)