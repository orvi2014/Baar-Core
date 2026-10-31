"""
baar/core/cache.py — reading prompt-cache usage.

Providers bill a prompt prefix they've seen recently at a discount (OpenAI and
DeepSeek automatically; Anthropic with cache_control markers; Gemini implicitly).
In an agent loop most of each prompt repeats, so the discount is large ("Don't
Break the Cache", arXiv 2601.06007: 41–80%). Baar reads how many prompt tokens a
call billed as cached, records streams at the cache rate, and reports the hit
rate. Reservations always use the uncached price, so caching never weakens the
budget guarantee.
"""

from __future__ import annotations

from typing import Any


def cached_tokens(usage: Any) -> int:
    """Cached prompt tokens in a usage object (OpenAI's or Anthropic's field), else 0."""
    details = getattr(usage, "prompt_tokens_details", None)
    value = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    if not isinstance(value, int) or isinstance(value, bool):
        value = getattr(usage, "cache_read_input_tokens", None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0
