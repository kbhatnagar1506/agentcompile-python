"""Turning create() keyword arguments into the JSON the decision call sends."""

from __future__ import annotations

from typing import Any

# What the engine reads; everything else stays between the customer and their provider.
_FIELDS = ("model", "messages", "system", "tools")
# A Responses API call (OpenAI's `responses.create`): the same, in that API's own words.
_RESPONSES_FIELDS = (
    "model",
    "input",
    "instructions",
    "tools",
    "previous_response_id",
    "conversation",
)


def jsonable(value: Any) -> Any:
    """Plain JSON from what agent loops put in messages: dicts, lists, and the SDKs' own
    response objects appended back into the history (pydantic models)."""
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def payload(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: jsonable(kwargs[k]) for k in _FIELDS if k in kwargs and kwargs[k] is not None}


def request_of(kwargs: dict[str, Any]) -> dict[str, Any]:
    """What capture keeps of a call: `payload`, or a Responses API call's own fields."""
    if "input" in kwargs and "messages" not in kwargs:
        return {
            k: jsonable(kwargs[k])
            for k in _RESPONSES_FIELDS
            if k in kwargs and kwargs[k] is not None
        }
    return payload(kwargs)
