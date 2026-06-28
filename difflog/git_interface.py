"""Git interface layer.

Wraps gitpython to extract structured diff data between two refs.
Supports commit hashes, tags, and branch names as --from / --to targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import git


@dataclass
class CommitInfo:
    sha: str
    short_sha: str
    message: str
    author: str
    date: str


@dataclass
class FileDiff:
    path: str
    extension: str
    change_type: str          # 'added' | 'deleted' | 'modified' | 'renamed'
    old_path: str | None      # populated for renames
    lines_added: int
    lines_deleted: int
    raw_diff: str             # unified diff text (may be empty for binary)
    is_binary: bool = False


@dataclass
class DiffResult:
    from_ref: str
    to_ref: str
    commits: list[CommitInfo] = field(default_factory=list)
    file_diffs: list[FileDiff] = field(default_factory=list)

    @property
    def total_files_changed(self) -> int:
        return len(self.file_diffs)

    @property
    def total_lines_added(self) -> int:
        return sum(f.lines_added for f in self.file_diffs)

    @property
    def total_lines_deleted(self) -> int:
        return sum(f.lines_deleted for f in self.file_diffs)


class GitInterface:
    """High-level interface for extracting diff data from a git repo."""

    def __init__(self, repo_path: str | Path = "."):
        try:
            self.repo = git.Repo(repo_path, search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            raise ValueError(f"No git repository found at or above: {repo_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_diff(self, from_ref: str, to_ref: str) -> DiffResult:
        """Return a DiffResult for the range [from_ref..to_ref]."""
        result = DiffResult(from_ref=from_ref, to_ref=to_ref)
        result.commits = self._get_commits(from_ref, to_ref)
        result.file_diffs = self._get_file_diffs(from_ref, to_ref)
        return result

    def validate_ref(self, ref: str) -> bool:
        """Return True if *ref* can be resolved in this repo."""
        try:
            self.repo.commit(ref)
            return True
        except (git.BadName, git.BadObject, ValueError):
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_commits(self, from_ref: str, to_ref: str) -> list[CommitInfo]:
        """List commits reachable from to_ref but not from_ref."""
        commits = []
        try:
            rev_range = f"{from_ref}..{to_ref}"
            for commit in self.repo.iter_commits(rev_range):
                commits.append(
                    CommitInfo(
                        sha=commit.hexsha,
                        short_sha=commit.hexsha[:7],
                        message=commit.message.strip(),
                        author=str(commit.author),
                        date=commit.committed_datetime.strftime("%Y-%m-%d"),
                    )
                )
        except git.GitCommandError as exc:
            raise ValueError(f"Failed to list commits ({from_ref}..{to_ref}): {exc}") from exc
        return commits

    def _get_file_diffs(self, from_ref: str, to_ref: str) -> list[FileDiff]:
        """Return per-file diff data between the two refs."""
        try:
            base_commit = self.repo.commit(from_ref)
            head_commit = self.repo.commit(to_ref)
        except (git.BadName, git.BadObject, ValueError) as exc:
            raise ValueError(f"Cannot resolve refs: {exc}") from exc

        diffs = base_commit.diff(head_commit)
        file_diffs: list[FileDiff] = []

        for diff_item in diffs:
            file_diffs.append(self._parse_diff_item(diff_item))

        return file_diffs

    def _parse_diff_item(self, item: git.Diff) -> FileDiff:
        change_type = self._change_type(item)
        path = item.b_path or item.a_path or "unknown"
        old_path = item.a_path if change_type == "renamed" else None
        ext = Path(path).suffix.lower()

        # Try to get the unified diff text
        raw_diff = ""
        is_binary = False
        try:
            raw_diff = item.diff.decode("utf-8", errors="replace") if item.diff else ""
        except Exception:
            is_binary = True

        lines_added, lines_deleted = self._count_lines(raw_diff)

        return FileDiff(
            path=path,
            extension=ext,
            change_type=change_type,
            old_path=old_path,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            raw_diff=raw_diff,
            is_binary=is_binary,
        )

    @staticmethod
    def _change_type(item: git.Diff) -> str:
        if item.new_file:
            return "added"
        if item.deleted_file:
            return "deleted"
        if item.renamed_file:
            return "renamed"
        return "modified"

    @staticmethod
    def _count_lines(raw_diff: str) -> tuple[int, int]:
        added = deleted = 0
        for line in raw_diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                deleted += 1
        return added, deleted