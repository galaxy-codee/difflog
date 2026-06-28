"""LLM Summarizer.

Sends structured diff context (not raw diffs) to the OpenAI API and
returns per-file one-liners plus an overall release summary.

Design goals:
- Token-efficient: only sends semantic summaries, not raw patch text
- Graceful degradation: if the API is unavailable, returns rule-based fallbacks
- Configurable: can be bypassed with --no-llm
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openai import OpenAI

from .classifier import ClassifiedChange
from .semantic_parser import SemanticFileSummary


@dataclass
class LLMSummary:
    file_summaries: dict[str, str]   # path → one-line summary
    release_summary: str              # overall paragraph
    model_used: str


_SYSTEM_PROMPT = """\
You are a technical writer helping developers understand what changed between
two versions of a software project.

You will receive structured metadata about changed files — NOT raw diffs.
Your job is to produce:
1. A one-line plain-English summary for each file (25 words or fewer).
2. A concise overall release summary (2–4 sentences) written in the style
   of a professional CHANGELOG entry.

Return ONLY valid JSON in this exact schema (no markdown fences):
{
  "file_summaries": {"<path>": "<one-liner>", ...},
  "release_summary": "<paragraph>"
}
"""


def _build_user_message(
    summaries: list[SemanticFileSummary],
    classifications: list[ClassifiedChange],
    from_ref: str,
    to_ref: str,
) -> str:
    class_map = {c.path: c for c in classifications}
    files_payload = []

    for s in summaries:
        c = class_map.get(s.path)
        entry = {
            "path": s.path,
            "change_type": s.file_change_type,
            "label": c.labels[0] if c.labels else "unknown",
            "lines_added": s.lines_added,
            "lines_deleted": s.lines_deleted,
            "significance": s.significance,
            "symbol_changes": [
                {
                    "kind": sc.kind,
                    "change": sc.change,
                    "name": sc.name,
                    **({"old_name": sc.old_name} if sc.old_name else {}),
                }
                for sc in s.symbol_changes
            ],
            "notes": s.notes,
        }
        files_payload.append(entry)

    payload = {
        "from_ref": from_ref,
        "to_ref": to_ref,
        "files": files_payload,
    }
    return json.dumps(payload, indent=2)


class LLMSummarizer:
    """Wraps the OpenAI API to produce changelog summaries."""

    DEFAULT_MODEL = "gpt-4o-mini"
    MAX_TOKENS = 1024

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. "
                "Export the variable or pass --no-llm to skip AI summaries."
            )
        self.client = OpenAI(api_key=key)

    def summarize(
        self,
        summaries: list[SemanticFileSummary],
        classifications: list[ClassifiedChange],
        from_ref: str,
        to_ref: str,
    ) -> LLMSummary:
        user_msg = _build_user_message(summaries, classifications, from_ref, to_ref)

        response = self.client.chat.completions.create(
            model=self.DEFAULT_MODEL,
            max_tokens=self.MAX_TOKENS,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )

        raw_text = response.choices[0].message.content.strip()
        data = json.loads(raw_text)

        return LLMSummary(
            file_summaries=data.get("file_summaries", {}),
            release_summary=data.get("release_summary", ""),
            model_used=self.DEFAULT_MODEL,
        )

    # ------------------------------------------------------------------
    # Fallback (no-LLM mode)
    # ------------------------------------------------------------------

    @staticmethod
    def fallback_summary(
        summaries: list[SemanticFileSummary],
        classifications: list[ClassifiedChange],
    ) -> LLMSummary:
        """Generate simple rule-based summaries when LLM is disabled."""
        class_map = {c.path: c for c in classifications}
        file_summaries: dict[str, str] = {}

        for s in summaries:
            c = class_map.get(s.path)
            label = c.labels[0] if c and c.labels else "chore"
            reason = c.reason if c else ""

            if s.symbol_changes:
                desc = "; ".join(
                    f"{sc.change} {sc.kind} `{sc.name}`" for sc in s.symbol_changes[:2]
                )
                file_summaries[s.path] = f"{label.capitalize()}: {desc}."
            elif reason:
                file_summaries[s.path] = f"{label.capitalize()}: {reason}."
            else:
                file_summaries[s.path] = (
                    f"{label.capitalize()}: {s.lines_added}+ / {s.lines_deleted}- lines."
                )

        # Overall summary
        from collections import Counter
        label_counts = Counter(c.label for c in classifications)
        parts = [f"{v} {k}(s)" for k, v in label_counts.most_common()]
        release_summary = (
            f"This release contains {', '.join(parts)}. "
            f"Run with LLM enabled for a richer summary."
        )

        return LLMSummary(
            file_summaries=file_summaries,
            release_summary=release_summary,
            model_used="none (offline fallback)",
        )