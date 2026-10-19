"""
type_suggest.py
Optional fallback for classifying assessment names that the keyword rules
could not place (for example "Kahoot 3" or "Viva"). It is OFF by default.

Enable with ATRISK_LLM_SUGGESTIONS=1 and an Anthropic API key
(ANTHROPIC_API_KEY), plus `pip install anthropic`. Only the assessment
NAMES are sent, never marks or student identifiers. A suggestion is only a
pre-filled value: the admin confirms or changes every type on the mapping
screen, which shows where each suggestion came from.
"""

import json
import logging

from config import LLM_SUGGESTIONS, LLM_MODEL, TYPES

log = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "type": {"type": "string", "enum": TYPES}},
                "required": ["name", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

_PROMPT = (
    "Classify each university course assessment name into one broad type.\n"
    "Coursework: assignments, projects, essays, reports, presentations, portfolios, participation.\n"
    "Quiz: short regular quizzes or topic checks.\n"
    "Test: in-semester tests such as a mid semester test.\n"
    "Practical: labs, practicals, workshops, fieldwork, clinical or studio work.\n"
    "Exam: final or end of semester examinations.\n"
    "Other: anything that fits none of these.\n\n"
    "Assessment names, one per line:\n"
)


def suggest(names):
    """Returns {name: type} for the names it could classify. Never raises."""
    if not LLM_SUGGESTIONS or not names:
        return {}
    try:
        import anthropic
    except ImportError:
        log.warning("LLM type suggestions are enabled but the anthropic package is not installed.")
        return {}
    try:
        client = anthropic.Anthropic(timeout=30.0, max_retries=1)
        response = client.beta.messages.create(
            model=LLM_MODEL,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user", "content": _PROMPT + "\n".join(names)}],
        )
        if response.stop_reason == "refusal":
            return {}
        text = next((b.text for b in response.content if b.type == "text"), "")
        items = json.loads(text).get("items", [])
        wanted = set(names)
        return {i["name"]: i["type"] for i in items if i.get("name") in wanted and i.get("type") in TYPES}
    except anthropic.APIConnectionError:
        log.warning("LLM type suggestion skipped: could not reach the API.")
    except anthropic.APIStatusError as exc:
        log.warning("LLM type suggestion skipped: API error %s.", exc.status_code)
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("LLM type suggestion skipped: unreadable response (%s).", exc)
    return {}
