"""difflog CLI entry point.

Usage examples:
    difflog --from v1.0.0 --to v1.2.0
    difflog --from v1.0.0 --to v1.2.0 --output CHANGELOG.md
    difflog --from main --to dev --format markdown
    difflog --from HEAD~10 --to HEAD --no-llm
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .classifier import ChangeClassifier, MultiClassifiedChange
from .git_interface import GitInterface
from .llm_summarizer import LLMSummary, LLMSummarizer
from .renderer import ChangelogRenderer
from .semantic_parser import SemanticParser

console = Console()
err_console = Console(stderr=True)


@click.command()
@click.option("--from", "from_ref", required=True, metavar="REF",
              help="Starting commit, tag, or branch (e.g. v1.0.0, main, abc1234)")
@click.option("--to", "to_ref", required=True, metavar="REF",
              help="Ending commit, tag, or branch (e.g. v1.2.0, dev, HEAD)")
@click.option("--output", "-o", default=None, metavar="FILE",
              help="Write changelog to FILE instead of stdout")
@click.option("--format", "fmt", default="markdown",
              type=click.Choice(["markdown", "json"], case_sensitive=False),
              show_default=True, help="Output format")
@click.option("--no-llm", is_flag=True, default=False,
              help="Skip LLM summarization (offline / faster mode)")
@click.option("--repo", default=".", metavar="PATH", show_default=True,
              help="Path to the git repository")
@click.option("--version-label", default=None, metavar="LABEL",
              help="Override version label in the changelog header")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show per-file classification details")
def main(
    from_ref: str,
    to_ref: str,
    output: str | None,
    fmt: str,
    no_llm: bool,
    repo: str,
    version_label: str | None,
    verbose: bool,
) -> None:
    """Generate a semantic changelog between two git refs."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=err_console,
    ) as progress:

        # ── 1. Git layer ──────────────────────────────────────────────
        task = progress.add_task("Reading git history…", total=None)
        try:
            git = GitInterface(repo)
        except ValueError as exc:
            err_console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)

        for ref in (from_ref, to_ref):
            if not git.validate_ref(ref):
                err_console.print(f"[red]Error:[/red] Cannot resolve ref: {ref!r}")
                sys.exit(1)

        try:
            diff_result = git.get_diff(from_ref, to_ref)
        except ValueError as exc:
            err_console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)

        progress.update(task, description=f"Found {diff_result.total_files_changed} changed file(s)…")

        if diff_result.total_files_changed == 0:
            err_console.print("[yellow]No files changed between the given refs.[/yellow]")
            sys.exit(0)

        # ── 2. Semantic parsing ───────────────────────────────────────
        progress.update(task, description="Parsing file semantics…")
        parser = SemanticParser()
        summaries = parser.parse(diff_result.file_diffs)

        # ── 3. Classification ─────────────────────────────────────────
        progress.update(task, description="Classifying changes…")
        classifier = ChangeClassifier()
        commit_messages = [c.message for c in diff_result.commits]
        classifications = classifier.classify_all(summaries, commit_messages)

        # ── 4. LLM summarization ──────────────────────────────────────
        if no_llm:
            progress.update(task, description="Generating rule-based summaries…")
            llm_summary = LLMSummarizer.fallback_summary(summaries, classifications)
        else:
            progress.update(task, description="Asking Claude for summaries…")
            try:
                summarizer = LLMSummarizer()
                llm_summary = summarizer.summarize(
                    summaries, classifications, from_ref, to_ref
                )
            except EnvironmentError as exc:
                err_console.print(f"[yellow]Warning:[/yellow] {exc}")
                err_console.print("[yellow]Falling back to rule-based summaries.[/yellow]")
                llm_summary = LLMSummarizer.fallback_summary(summaries, classifications)
            except Exception as exc:
                err_console.print(f"[yellow]LLM call failed:[/yellow] {exc}")
                llm_summary = LLMSummarizer.fallback_summary(summaries, classifications)

        # ── 5. Render ─────────────────────────────────────────────────
        progress.update(task, description="Rendering changelog…")
        renderer = ChangelogRenderer()

        if fmt == "json":
            import json
            payload = {
                "from_ref": from_ref,
                "to_ref": to_ref,
                "release_summary": llm_summary.release_summary,
                "files": [
                    {
                        "path": c.path,
                        "labels": c.labels,
                        "confidence": c.confidence,
                        "summary": llm_summary.file_summaries.get(c.path, c.reason),
                    }
                    for c in classifications
                ],
                "commits": [
                    {"sha": cm.short_sha, "message": cm.message.splitlines()[0], "author": cm.author}
                    for cm in diff_result.commits
                ],
            }
            changelog_text = json.dumps(payload, indent=2)
        else:
            changelog_text = renderer.render(
                diff_result, classifications, llm_summary, version_label
            )

        progress.update(task, description="Done.")

    # ── Output ────────────────────────────────────────────────────────
    if output:
        Path(output).write_text(changelog_text, encoding="utf-8")
        console.print(f"[green]✓[/green] Changelog written to [bold]{output}[/bold]")
    else:
        console.print(changelog_text)

    # ── Verbose classification table ──────────────────────────────────
    if verbose:
        table = Table(title="Per-file Classification", show_lines=True)
        table.add_column("File", style="cyan", no_wrap=False)
        table.add_column("Label", style="bold")
        table.add_column("Confidence")
        table.add_column("Reason")

        label_colors = {
            "breaking": "red",
            "feature": "green",
            "bugfix": "yellow",
            "refactor": "blue",
            "documentation": "cyan",
            "chore": "dim",
        }
        for c in classifications:
            labels_str = ", ".join(
                f"[{label_colors.get(l, 'white')}]{l}[/{label_colors.get(l, 'white')}]"
                for l in c.labels
            )
            table.add_row(c.path, labels_str, c.confidence, c.reason)
        err_console.print(table)


if __name__ == "__main__":
    main()