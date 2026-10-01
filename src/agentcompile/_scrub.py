"""Scrubbing on your machine: personal data in captured calls becomes keyed tokens before
anything is sent (on by default with capture; `wrap(..., scrub=False)` turns it off).

The same rules and token format as AgentCompile's server (`<email:3f9a1c2e>`: the first 8 hex
characters of HMAC-SHA256(key, normalized value)), so the same value always becomes the same
token and AgentCompile can still match values across a conversation without ever seeing them.
Emails always; payment cards (13-19 digits passing Luhn); phone-shaped numbers; and values
under keys that name a card, phone or account outright. Ids are left alone.

The key never leaves your machine: AGENTCOMPILE_SCRUB_KEY (set the same one on every server,
so tokens match across your fleet), else a random key created once in ~/.agentcompile/.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD_RE = re.compile(
    r"(?<![\d-])(?:\d{13,19}|\d{4}(?:[ -]\d{4}){2,3}(?:[ -]\d{1,3})?)(?![\d-])"
)
_PHONE_RE = re.compile(
    r"(?<![\w+])(?:"
    r"\+\d{1,3}[ .-]?(?:\(\d{1,4}\)[ .-]?)?\d{2,4}(?:[ .-]\d{2,4}){1,4}"
    r"|\+\d{10,15}"
    r"|\(\d{3}\)[ .-]?\d{3}[ .-]\d{4}"
    r"|\d{3}[.-]\d{3}[.-]\d{4}"
    r")(?![\w])"
)
_ID_KEY = re.compile(r"(^|_)(id|ids|number|no|sku|code|ref|reference|timestamp|ts)$", re.I)
_PHONE_KEY = re.compile(r"(phone|mobile|cell|tel|fax|whatsapp)", re.I)
_ACCOUNT_KEY = re.compile(
    r"(account_?(number|no|num)|accountnumber|iban|routing|sort_?code|ssn|social_?security)",
    re.I,
)


def load_key() -> bytes:
    """AGENTCOMPILE_SCRUB_KEY, else ~/.agentcompile/scrub.key (created once, owner-only)."""
    env = os.environ.get("AGENTCOMPILE_SCRUB_KEY")
    if env:
        return env.encode()
    path = (
        Path(os.environ.get("AGENTCOMPILE_HOME", Path.home() / ".agentcompile")) / "scrub.key"
    )
    try:
        return path.read_bytes().strip()
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_hex(32)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(key)
        return key.encode()


def _token(kind: str, normalized: str, key: bytes) -> str:
    digest = hmac.new(key, normalized.encode(), hashlib.sha256).hexdigest()[:8]
    return f"<{kind}:{digest}>"


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        digit = int(ch)
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def scrub_text(text: str, key: bytes, *, numbers: bool = True) -> str:
    text = _EMAIL_RE.sub(lambda m: _token("email", m.group(0).strip().lower(), key), text)
    if not numbers:
        return text

    def card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return _token("card", digits, key)
        return match.group(0)

    def phone(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 10 <= len(digits) <= 15:
            return _token("phone", digits, key)
        return match.group(0)

    return _PHONE_RE.sub(phone, _CARD_RE.sub(card, text))


def _phone_field(value: str, key: bytes) -> str:
    digits = re.sub(r"\D", "", value)
    if 7 <= len(digits) <= 15 and not _EMAIL_RE.search(value):
        return _token("phone", digits, key)
    return scrub_text(value, key)


def _is_card_field(field: str) -> bool:
    words = {w.lower() for w in re.findall(r"[A-Za-z][a-z]*", field)}
    return bool(field) and bool(words & {"card", "cc", "pan"})


def _sensitive_field(value: Any, field: str, key: bytes) -> str | None:
    raw = (
        str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
    )
    if not isinstance(raw, str) or not field:
        return None
    digits = re.sub(r"\D", "", raw)
    if _ACCOUNT_KEY.search(field) and len(digits) >= 4:
        return _token("account", digits, key)
    if _is_card_field(field) and 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return _token("card", digits, key)
    if _PHONE_KEY.search(field) and 7 <= len(digits) <= 15 and not _EMAIL_RE.search(raw):
        return _token("phone", digits, key)
    return None


def scrub_value(value: Any, key: bytes, field: str = "") -> Any:
    sensitive = _sensitive_field(value, field, key)
    if sensitive is not None:
        return sensitive
    if isinstance(value, str):
        if field and _PHONE_KEY.search(field):
            return _phone_field(value, key)
        id_like = bool(field and _ID_KEY.search(field) and not _is_card_field(field))
        return scrub_text(value, key, numbers=not id_like)
    if isinstance(value, dict):
        return {k: scrub_value(v, key, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_value(v, key, field) for v in value]
    return value


def scrub_tool_output(output: str, key: bytes) -> str:
    """A tool result or arguments string: JSON is scrubbed field by field, text as text."""
    try:
        parsed = json.loads(output)
    except ValueError:
        return scrub_text(output, key)
    separators = (", ", ": ") if (", " in output or ": " in output) else (",", ":")
    return json.dumps(scrub_value(parsed, key), separators=separators)


def scrub_call(value: Any, key: bytes, field: str = "") -> Any:
    """A captured request or answer: every string scrubbed; a string holding JSON (tool
    arguments, tool results) scrubbed as JSON, field names included."""
    if isinstance(value, str) and value[:1] in ("{", "["):
        return scrub_tool_output(value, key)
    if isinstance(value, dict):
        return {k: scrub_call(v, key, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_call(v, key, field) for v in value]
    return scrub_value(value, key, field)
