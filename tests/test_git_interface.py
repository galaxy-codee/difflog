"""Tests for git_interface.py"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from difflog.git_interface import GitInterface, FileDiff, CommitInfo, DiffResult


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_repo(tmp_path):
    """Return a GitInterface whose underlying repo is fully mocked."""
    with patch("difflog.git_interface.git.Repo") as MockRepo:
        instance = MockRepo.return_value
        gi = GitInterface(str(tmp_path))
        gi.repo = instance
        yield gi, instance


# ------------------------------------------------------------------
# validate_ref
# ------------------------------------------------------------------

class TestValidateRef:
    def test_valid_ref_returns_true(self, mock_repo):
        gi, repo = mock_repo
        repo.commit.return_value = MagicMock()
        assert gi.validate_ref("main") is True

    def test_invalid_ref_returns_false(self, mock_repo):
        import git
        gi, repo = mock_repo
        repo.commit.side_effect = git.BadName
        assert gi.validate_ref("nonexistent") is False


# ------------------------------------------------------------------
# _count_lines
# ------------------------------------------------------------------

class TestCountLines:
    def test_counts_correctly(self):
        diff = (
            "+++ b/foo.py\n"
            "--- a/foo.py\n"
            "+added line\n"
            "+another added\n"
            "-removed line\n"
            " context line\n"
        )
        added, deleted = GitInterface._count_lines(diff)
        assert added == 2
        assert deleted == 1

    def test_empty_diff(self):
        assert GitInterface._count_lines("") == (0, 0)

    def test_ignores_header_lines(self):
        diff = "+++ b/foo.py\n--- a/foo.py\n"
        assert GitInterface._count_lines(diff) == (0, 0)


# ------------------------------------------------------------------
# _change_type
# ------------------------------------------------------------------

class TestChangeType:
    def _make_diff_item(self, new=False, deleted=False, renamed=False):
        item = MagicMock()
        item.new_file = new
        item.deleted_file = deleted
        item.renamed_file = renamed
        return item

    def test_new_file(self):
        item = self._make_diff_item(new=True)
        assert GitInterface._change_type(item) == "added"

    def test_deleted_file(self):
        item = self._make_diff_item(deleted=True)
        assert GitInterface._change_type(item) == "deleted"

    def test_renamed_file(self):
        item = self._make_diff_item(renamed=True)
        assert GitInterface._change_type(item) == "renamed"

    def test_modified_file(self):
        item = self._make_diff_item()
        assert GitInterface._change_type(item) == "modified"


# ------------------------------------------------------------------
# DiffResult properties
# ------------------------------------------------------------------

class TestDiffResult:
    def test_total_files_changed(self):
        fd1 = FileDiff("a.py", ".py", "modified", None, 5, 2, "", False)
        fd2 = FileDiff("b.py", ".py", "added",    None, 10, 0, "", False)
        dr = DiffResult("v1", "v2", [], [fd1, fd2])
        assert dr.total_files_changed == 2

    def test_total_lines(self):
        fd1 = FileDiff("a.py", ".py", "modified", None, 5, 2, "", False)
        fd2 = FileDiff("b.py", ".py", "modified", None, 3, 1, "", False)
        dr = DiffResult("v1", "v2", [], [fd1, fd2])
        assert dr.total_lines_added == 8
        assert dr.total_lines_deleted == 3