"""Mubit feedback-loop demo — one agent, one terminal, no UI.

An agent submits expense reports to an internal API (expense_api.py)
whose validation rules it was never given. The transcript shows every
piece of data that moves between the agent and Mubit:

    -> lines: data going into Mubit or the API
    <- lines: data coming back

The loop, per expense:

    1. RECALL   — read stored rules from Mubit. Rules below the
                  confidence gate are not applied.
    2. ACT      — draft the submission with an LLM. Applied rules go
                  into the prompt.
    3. FEEDBACK — the API accepts or rejects. Each attempt's outcome
                  is recorded to Mubit as a step outcome.
    4. LEARN    — after an accept that needed retries, the agent
                  distills one rule per rejected field and stores each
                  as a lesson (upserted, one entry per rule).
    5. CLOSE    — record_outcome reports how each APPLIED rule fared
                  back to that lesson. Mubit moves its confidence up
                  or down. A rule the API contradicted is retired and
                  its corrected replacement re-earns confidence from
                  the initial value. The next RECALL reads the result.

Run:  ./run_demo.sh            scripted acts
      ./run_demo.sh --chat     scripted acts, then type your own expenses
      ./run_demo.sh --step     pause between acts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass

from mubit import Client

import expense_api

MODEL = os.environ.get("MODEL", "gpt-5-mini")
CONFIDENCE_GATE = 0.5  # rules below this are quarantined: kept, not applied
BAND_HIGH = 0.75

# ---------------------------------------------------------------------------
# Terminal output. Fixed gutter, one color per channel, no UI.
# ---------------------------------------------------------------------------

_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


DIM, BOLD = "2", "1"
CYAN, GREEN, RED, YELLOW, MAGENTA = "36", "32", "31", "33", "35"


def line(channel: str, arrow: str, text: str, color: str = "0") -> None:
    gutter = c(DIM, f"{channel:>10}")
    print(f"  {gutter} {arrow} {c(color, text)}")


def jshort(obj: object, limit: int = 150) -> str:
    s = json.dumps(obj, separators=(", ", ": "))
    return s if len(s) <= limit else s[: limit - 12] + " ... " + s[-6:]


# ---------------------------------------------------------------------------
# LLM calls (OpenAI). Reasoning models spend reasoning tokens inside
# max_completion_tokens, so the budget carries headroom.
# ---------------------------------------------------------------------------

LLM = {"calls": 0, "tokens_in": 0, "tokens_out": 0}
_openai_client = None


def llm_json(system: str, user: str) -> dict:
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI()
    for attempt in (1, 2):
        LLM["calls"] += 1
        kwargs = dict(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=2048,
        )
        if MODEL.startswith("gpt-5"):
            kwargs["reasoning_effort"] = "minimal"
        resp = _openai_client.chat.completions.create(**kwargs)
        LLM["tokens_in"] += resp.usage.prompt_tokens
        LLM["tokens_out"] += resp.usage.completion_tokens
        raw = (resp.choices[0].message.content or "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            user = user + "\nReturn only one valid JSON object."
    return {}


# ---------------------------------------------------------------------------
# Mubit wrapper. Every method prints what it sends and what returns.
# ---------------------------------------------------------------------------

RULE_TAG = re.compile(r"^\[rule:([a-z_]+)\]\s*(.*)$", re.S)


def bucket_for_field(field: str) -> str:
    """The memory key an error field files under. Unknown-field errors
    all teach the same rule (the allowed field list), so they share
    one bucket."""
    known = {
        "merchant": "merchant",
        "date": "date",
        "amount_minor": "amount",
        "category": "category",
        "needs_approval": "approval",
        "note": "note",
    }
    return known.get(field, "schema")


class Memory:
    def __init__(self, client=None) -> None:
        # server.py injects a traced client here; the terminal demo
        # builds a plain one.
        self.client = client or Client(
            endpoint=os.environ.get("MUBIT_ENDPOINT", "http://127.0.0.1:3970"),
            api_key=os.environ["MUBIT_API_KEY"],
            transport="http",
        )
        self.run_id = f"loop-demo-{uuid.uuid4().hex[:8]}"
        self.client.set_run_id(self.run_id)
        self._direct = True  # direct search path; falls back if the instance gates it

    def recall_rules(self, quiet: bool = False) -> dict[str, dict]:
        query = "expense API submission formatting rules"
        if not quiet:
            line("recall", "->", f'query="{query}"  entry_types=[lesson]', CYAN)
        kwargs = dict(
            query=query,
            limit=12,
            entry_types=["lesson"],
            evidence_only=True,
            prefer_current_run=True,
            include_working_memory=False,
        )
        try:
            if self._direct:
                out = self.client.recall(mode="direct_bypass", **kwargs)
            else:
                out = self.client.recall(**kwargs)
        except Exception:
            self._direct = False
            out = self.client.recall(**kwargs)
        rules: dict[str, dict] = {}
        for e in out.get("evidence") or []:
            text = (e.get("content") or e.get("text") or "").strip()
            m = RULE_TAG.match(text)
            if not m:
                continue
            key, body = m.group(1), m.group(2).strip()
            try:
                meta = json.loads(e.get("metadata_json") or "{}")
            except json.JSONDecodeError:
                meta = {}
            row = {
                "id": e.get("id"),
                "text": body,
                # A lesson with no recorded outcome sits at the server's
                # initial value, 0.5.
                "confidence": float(meta.get("confidence", 0.5)),
                "fresh": "confidence" not in meta,
                "reinforcement": meta.get("reinforcement_count"),
                "stored_at": meta.get("ingested_at") or "",
            }
            prev = rules.get(key)
            if prev is None or row["stored_at"] > prev["stored_at"]:
                rules[key] = row
        if not quiet:
            if rules:
                summary = "  ".join(
                    f"[{k} {r['confidence']:.2f}]" for k, r in sorted(rules.items())
                )
                line("recall", "<-", f"{len(rules)} stored rules  {summary}", CYAN)
            else:
                line("recall", "<-", "0 stored rules (cold start)", CYAN)
        return rules

    def store_rule(self, key: str, text: str) -> None:
        content = f"[rule:{key}] {text}"[:800]
        line("remember", "->", f'lesson "{content}"  upsert_key=rule:{key}', CYAN)
        self.client.remember(
            content=content,
            intent="lesson",
            lesson_type="success",
            lesson_scope="run",
            lesson_importance="high",
            upsert_key=f"rule:{key}",
            metadata={"rule": key},
            wait=True,
        )

    def report_outcome(self, key: str, rule: dict, ok: bool, why: str) -> None:
        # The signal is signed: positive reinforces, negative erodes.
        # Server side: confidence += signal * 0.1, clamped to [0, 1].
        payload = {
            "reference_id": rule["id"],
            "outcome": "success" if ok else "failure",
            "signal": 0.9 if ok else -0.9,
            "rationale": why,
        }
        line("outcome", "->", f"record_outcome {jshort(payload, 200)}", CYAN)
        r = self.client.record_outcome(**payload)
        conf = r.get("updated_confidence")
        count = r.get("reinforcement_count")
        if conf is not None:
            note = f"confidence {rule['confidence']:.2f} -> {conf:.2f}  (reinforcement {count})"
            if not ok and conf < CONFIDENCE_GATE:
                note += "  — below the gate, quarantined"
            line("outcome", "<-", f"[{key}] {note}", GREEN if ok else RED)

    def retire_rule(self, key: str, rule: dict) -> None:
        line("retire", "->",
             f"delete_lesson id={rule['id']}  — the [{key}] rule no longer matches the API",
             MAGENTA)
        try:
            self.client.delete_lesson({"run_id": self.run_id, "lesson_id": rule["id"]})
        except Exception as exc:
            line("retire", "<-", f"failed ({exc})", RED)

    def record_step(self, step_id: str, name: str, ok: bool, rationale: str) -> None:
        payload = {
            "step_id": step_id,
            "step_name": name,
            "outcome": "success" if ok else "failure",
            "signal": 0.9 if ok else 0.2,
            "rationale": rationale,
        }
        line("outcome", "->", f"record_step_outcome {jshort(payload, 170)}", CYAN)
        self.client.record_step_outcome(**payload, directive_hint=rationale)

    def reflect(self) -> None:
        line("reflect", "->", "last_n_items=24  (server-side distillation)", CYAN)
        try:
            r = self.client.reflect(last_n_items=24)
            lessons = r.get("lessons")
            if isinstance(lessons, list):
                n, examples = len(lessons), lessons[:2]
            else:
                n, examples = r.get("lessons_created", 0), []
            line("reflect", "<-",
                 f"server distilled {n} lesson(s) from the recorded outcomes", CYAN)
            for ex in examples:
                if isinstance(ex, dict) and ex.get("content"):
                    line("reflect", "  ", f'e.g. "{ex["content"][:90]}"', DIM)
        except Exception as exc:  # optional beat; the loop does not depend on it
            line("reflect", "<-", f"skipped ({exc})", DIM)


# ---------------------------------------------------------------------------
# The agent: draft with the LLM, submit, retry on rejection, learn.
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    text: str


DRAFT_SYSTEM = (
    "You prepare expense submissions for an internal expense API. "
    "Return only the JSON object to POST to /expenses. No commentary."
)


def draft(task: Task, applied: dict[str, dict], attempts: list[dict]) -> dict:
    system = DRAFT_SYSTEM
    if applied:
        rules = "\n".join(f"- {r['text']}" for r in applied.values())
        system += f"\nFollow these rules exactly when composing the payload:\n{rules}"
    user = f"Expense: {task.text}\nCurrent year: 2026."
    if attempts:
        last = attempts[-1]
        user += (
            "\nThe API rejected the previous attempt."
            f"\nPayload sent: {json.dumps(last['payload'])}"
            f"\nField errors: {json.dumps(last['errors'])}"
            "\nCorrect every rejected field and resubmit. The API's error"
            " message overrides any stored rule."
        )
    return llm_json(system, user)


DISTILL_SYSTEM = (
    "You maintain a rulebook for submitting expenses to an internal API. "
    "Write each rule as one short imperative line addressed to the person "
    "composing the next submission. Return JSON only."
)


def distill_rules(task: Task, attempts: list[dict], accepted: dict, keys: dict[str, list[str]]) -> dict[str, str]:
    """One LLM call: turn this task's rejections into per-key rules."""
    want = {k: fields for k, fields in keys.items()}
    user = (
        "The API rejected these attempts, then accepted the final payload.\n"
        f"Rejections: {json.dumps([a['errors'] for a in attempts])}\n"
        f"Accepted payload: {json.dumps(accepted)}\n"
        f"Write one rule per key. Keys and the rejected fields they cover: {json.dumps(want)}\n"
        'Return {"rules": {"<key>": "<one line>"}}'
    )
    out = llm_json(DISTILL_SYSTEM, user)
    rules = out.get("rules") or {}
    result = {}
    for key, fields in keys.items():
        text = rules.get(key)
        if not isinstance(text, str) or not text.strip():
            # fall back to the API's own words for this field
            text = "; ".join(f"{f}: {attempts[-1]['errors'].get(f, '')}" for f in fields)
        result[key] = text.strip()
    return result


MAX_TRIES = 4


def handle(task: Task, memory: Memory, stats: dict) -> None:
    print()
    print(f"  {c(BOLD, task.id)}  {task.text}")

    # 1. RECALL
    rules = memory.recall_rules()
    applied, quarantined = {}, {}
    for key, r in rules.items():
        if r["confidence"] >= CONFIDENCE_GATE:
            applied[key] = r
        else:
            quarantined[key] = r
    if quarantined:
        names = ", ".join(f"[{k}] {r['confidence']:.2f}" for k, r in quarantined.items())
        line("recall", "  ", f"below the {CONFIDENCE_GATE} gate, not applied: {names}", YELLOW)

    # 2. ACT / 3. FEEDBACK — draft, submit, retry on rejection
    attempts: list[dict] = []
    accepted = None
    active = dict(applied)  # rules still trusted within this task
    for attempt in range(1, MAX_TRIES + 1):
        note = " (rules in prompt)" if active and attempt == 1 else ""
        note = " (rejection attached)" if attempts else note
        line("llm", "  ", f"draft {attempt} — {MODEL}{note}")
        payload = draft(task, active, attempts)
        line("submit", "->", f"POST /expenses {jshort(payload, 200)}", YELLOW)
        status, body = expense_api.submit(payload)
        if status == 201:
            line("api", "<-", f"201 accepted  id={body['id']}", GREEN)
            memory.record_step(f"{task.id}/try{attempt}", "submit_expense", True,
                               f"{task.id} accepted on attempt {attempt}")
            accepted = payload
            break
        errors = body["fields"]
        line("api", "<-", f"422 rejected — {len(errors)} field error(s)", RED)
        for f, msg in errors.items():
            line("api", "  ", f"{f}: {msg}", RED)
        memory.record_step(f"{task.id}/try{attempt}", "submit_expense", False,
                           f"{task.id} rejected: {', '.join(errors)}")
        attempts.append({"payload": payload, "errors": errors})
        # A rule whose field the API just rejected is contradicted.
        # Stop applying it for the rest of this task.
        contradicted = {bucket_for_field(f) for f in errors} & set(active)
        if contradicted:
            for k in contradicted:
                active.pop(k)
            line("agent", "  ",
                 f"rule(s) contradicted by the API, dropped for the retry: "
                 f"{', '.join(sorted(contradicted))}", YELLOW)

    # Which memory keys saw an error this task?
    errored_keys: dict[str, list[str]] = {}
    for a in attempts:
        for f in a["errors"]:
            errored_keys.setdefault(bucket_for_field(f), []).append(f)
    for k in errored_keys:
        errored_keys[k] = sorted(set(errored_keys[k]))

    # 5. CLOSE — report how each applied rule fared, back to its lesson.
    # A rule whose field was rejected while applied takes a failure
    # outcome; when the corrected submission was then accepted, the
    # contradicted rule is retired so the distilled replacement can
    # take its key.
    for key, r in applied.items():
        if r["id"] is None:
            continue
        failed = key in errored_keys
        why = (
            f"{task.id}: field(s) {', '.join(errored_keys[key])} rejected while this rule was applied"
            if failed
            else f"{task.id}: applied, submission accepted"
        )
        memory.report_outcome(key, r, not failed, why)
        if failed and accepted is not None:
            memory.retire_rule(key, r)

    # 4. LEARN — distill one rule per rejected field, store each
    if accepted is not None and errored_keys:
        line("llm", "  ", f"distill — {len(errored_keys)} rule(s) from the rejections")
        new_rules = distill_rules(task, attempts, accepted, errored_keys)
        for key, text in new_rules.items():
            memory.store_rule(key, text)

    # Task summary
    tries = len(attempts) + (1 if accepted else 0)
    stats["tasks"] += 1
    stats["tries"] += tries
    if accepted is not None and not attempts:
        stats["first_try"] += 1
    verdict = (
        f"accepted on try {tries}" if accepted is not None else f"unresolved after {MAX_TRIES} tries"
    )
    parts = [verdict]
    if errored_keys and accepted is not None:
        replaced = sorted(set(errored_keys) & set(applied))
        minted = sorted(set(errored_keys) - set(applied))
        if minted:
            parts.append(f"{len(minted)} rule(s) learned")
        if replaced:
            parts.append(f"{len(replaced)} rule(s) replaced")
    reused = [k for k in applied if k not in errored_keys]
    if reused:
        parts.append(f"reused: {', '.join(sorted(reused))}")
    line("=", "  ", "  ·  ".join(parts), BOLD)


# ---------------------------------------------------------------------------
# Lesson board: what memory holds, with the confidence band.
# ---------------------------------------------------------------------------

def band(conf: float) -> tuple[str, str]:
    if conf >= BAND_HIGH:
        return "high — trusted", GREEN
    if conf >= CONFIDENCE_GATE:
        return "medium — applied", YELLOW
    return "low — quarantined", RED


def show_board(memory: Memory) -> None:
    rules = memory.recall_rules(quiet=True)
    print()
    print(f"  {c(BOLD, 'MEMORY')}  {c(DIM, f'lessons in run {memory.run_id}')}")
    if not rules:
        print(c(DIM, "    (empty)"))
        return
    for key, r in sorted(rules.items()):
        conf = r["confidence"]
        label, color = band(conf)
        if r["fresh"]:
            label += " · no outcomes yet"
        elif r["reinforcement"] is not None:
            label += f" · x{r['reinforcement']}"
        filled = round(conf * 10)
        bar = "#" * filled + "-" * (10 - filled)
        text = r["text"] if len(r["text"]) <= 58 else r["text"][:55] + "..."
        print(f"    {key:<9} {text:<60} {conf:.2f}  {bar}  {c(color, label)}")
    print(c(DIM, f"    band: >= {BAND_HIGH} trusted · >= {CONFIDENCE_GATE} applied in prompts · "
                 f"< {CONFIDENCE_GATE} quarantined (kept, not applied)"))


# ---------------------------------------------------------------------------
# Acts.
# ---------------------------------------------------------------------------

ACT1 = [
    Task("E-01", "Team lunch with the client at Nobu Bombay, 12 Aug, $84.20"),
    Task("E-02", "Return flight BOM-SFO for the customer visit, booked 14 Aug, $612.50"),
    Task("E-03", "A 27-inch monitor for the new hire, ordered 18 Aug, $189.00"),
]
ACT2 = [
    Task("E-04", "Airport taxi, 21 Aug, $23.75"),
    Task("E-05", "Annual design-tool subscription renewal, 22 Aug, $340.00"),
    Task("E-06", "Team dinner after the release, 22 Aug, $156.40"),
]
ACT3 = [
    Task("E-07", "Client coffee at Blue Tokai, 25 Aug, $12.80"),
    Task("E-08", "Monitor arm for the new hire's desk, 26 Aug, $45.00"),
]


def act_header(title: str, note: str) -> None:
    print()
    print(c(BOLD, f"  {'=' * 74}"))
    print(c(BOLD, f"  {title}"))
    if note:
        print(c(DIM, f"  {note}"))
    print(c(BOLD, f"  {'=' * 74}"))


def act_summary(name: str, stats: dict, llm_before: dict) -> dict:
    calls = LLM["calls"] - llm_before["calls"]
    tok = (LLM["tokens_in"] - llm_before["tokens_in"]) + (LLM["tokens_out"] - llm_before["tokens_out"])
    s = dict(stats, llm_calls=calls, tokens=tok)
    print()
    line("act", "  ",
         f"{name}: {stats['first_try']}/{stats['tasks']} accepted first try · "
         f"{stats['tries']} submissions · {calls} LLM calls · {tok} tokens", BOLD)
    return s


def run_act(name: str, note: str, tasks: list[Task], memory: Memory) -> dict:
    act_header(name, note)
    stats = {"tasks": 0, "first_try": 0, "tries": 0}
    before = dict(LLM)
    for t in tasks:
        handle(t, memory, stats)
    summary = act_summary(name, stats, before)
    show_board(memory)
    return summary


def pause(step: bool) -> None:
    if step and sys.stdin.isatty():
        input(c(DIM, "\n  [Enter] for the next act "))


def chat(memory: Memory) -> None:
    act_header("CHAT", "Type an expense in plain words. Commands: board · quit")
    n = 0
    stats = {"tasks": 0, "first_try": 0, "tries": 0}
    while True:
        try:
            text = input(c(BOLD, "\n  expense> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break
        if text.lower() == "board":
            show_board(memory)
            continue
        n += 1
        handle(Task(f"U-{n:02d}", text), memory, stats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chat", action="store_true", help="open a chat prompt after the acts")
    ap.add_argument("--fresh-chat", action="store_true", help="chat only, cold memory, no acts")
    ap.add_argument("--step", action="store_true", help="pause between acts")
    args = ap.parse_args()

    memory = Memory()
    print()
    print(c(BOLD, "  MUBIT FEEDBACK-LOOP DEMO"))
    print(c(DIM, f"  model {MODEL} · mubit {os.environ.get('MUBIT_ENDPOINT', 'http://127.0.0.1:3970')}"
                 f" · run {memory.run_id}"))
    print(c(DIM, "  The agent submits expenses to an internal API. It was never given the"))
    print(c(DIM, "  API's validation rules. Every -> and <- line is real data in or out."))

    if args.fresh_chat:
        chat(memory)
        return 0

    a1 = run_act("ACT 1 — COLD START", "No stored rules. The API teaches by rejection.",
                 ACT1, memory)
    memory.reflect()
    pause(args.step)

    a2 = run_act("ACT 2 — WARM", "Same API, new expenses. Stored rules ride in every prompt.",
                 ACT2, memory)
    pause(args.step)

    expense_api.set_policy(2)
    a3 = run_act("ACT 3 — THE API CHANGES",
                 "The expense service deploys v2: dates switch to YYYY-MM-DD. "
                 "Nobody tells the agent.", ACT3, memory)

    print()
    print(c(BOLD, f"  {'=' * 74}"))
    print(c(BOLD, "  SUMMARY"))
    print(c(BOLD, f"  {'=' * 74}"))
    rows = [("", "tasks", "first-try", "submissions", "LLM calls", "tokens")]
    for name, s in (("act 1 cold", a1), ("act 2 warm", a2), ("act 3 change", a3)):
        rows.append((name, str(s["tasks"]), f"{s['first_try']}/{s['tasks']}",
                     str(s["tries"]), str(s["llm_calls"]), str(s["tokens"])))
    for r in rows:
        print(f"    {r[0]:<14}{r[1]:>6}{r[2]:>11}{r[3]:>13}{r[4]:>11}{r[5]:>9}")
    print()
    print(c(DIM, "  The loop: rejection -> lesson -> applied on the next task -> outcome"))
    print(c(DIM, "  written back to the lesson. Confidence rises with each accepted reuse"))
    print(c(DIM, "  and falls when the API contradicts a rule; a contradicted rule is"))
    print(c(DIM, "  retired and its replacement re-earns trust. The band decides what"))
    print(c(DIM, "  the agent applies."))

    if args.chat:
        chat(memory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
