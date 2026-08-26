"""A stand-in for an internal expense service.

The agent never reads this file. It only sees the responses that
submit() returns. The validation rules below play the part of the
undocumented behavior of a real internal API: the agent has to find
them through rejections, the same way a person would.

POLICY_VERSION 1 is active at import time. set_policy(2) switches the
date format, which models a policy change that nobody tells the agent
about.
"""

from __future__ import annotations

import re
from datetime import datetime

POLICY_VERSION = 1

ALLOWED_FIELDS = ["merchant", "date", "amount_minor", "category", "needs_approval", "note"]
CATEGORIES = ["MEALS", "TRAVEL", "SUPPLIES", "SOFTWARE", "OTHER"]
APPROVAL_LIMIT_MINOR = 20000

_next_id = 1041


def set_policy(version: int) -> None:
    global POLICY_VERSION
    POLICY_VERSION = version


def _check_date(value: object) -> str | None:
    if POLICY_VERSION == 1:
        fmt, label = "%d.%m.%Y", "DD.MM.YYYY"
    else:
        fmt, label = "%Y-%m-%d", "YYYY-MM-DD"
    if not isinstance(value, str):
        return f"expected a string in format {label}"
    try:
        datetime.strptime(value, fmt)
    except ValueError:
        return f"expected format {label}"
    return None


def submit(record: object) -> tuple[int, dict]:
    """POST /expenses. Returns (status_code, body)."""
    global _next_id
    if not isinstance(record, dict):
        return 422, {"error": "validation_failed", "fields": {"body": "expected a JSON object"}}

    errors: dict[str, str] = {}

    for key in record:
        if key not in ALLOWED_FIELDS:
            errors[key] = (
                f"unknown field; allowed fields: {', '.join(ALLOWED_FIELDS)}"
            )

    merchant = record.get("merchant")
    if not isinstance(merchant, str) or not merchant.strip():
        errors["merchant"] = "required; non-empty string"

    if "date" not in record:
        errors["date"] = "required"
    else:
        problem = _check_date(record["date"])
        if problem:
            errors["date"] = problem

    amount = record.get("amount_minor")
    if amount is None:
        errors["amount_minor"] = "required; integer amount in minor units (cents)"
    elif not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
        errors["amount_minor"] = "expected a positive integer (minor units, no decimals)"

    category = record.get("category")
    if category not in CATEGORIES:
        errors["category"] = f"must be one of {', '.join(CATEGORIES)}"

    if isinstance(amount, int) and not isinstance(amount, bool) and amount > APPROVAL_LIMIT_MINOR:
        if record.get("needs_approval") is not True:
            errors["needs_approval"] = (
                f"required: true when amount_minor > {APPROVAL_LIMIT_MINOR}"
            )

    note = record.get("note")
    if note is not None and not isinstance(note, str):
        errors["note"] = "expected a string"

    if errors:
        return 422, {"error": "validation_failed", "fields": errors}

    _next_id += 1
    return 201, {"id": f"EXP-{_next_id}", "status": "accepted"}
