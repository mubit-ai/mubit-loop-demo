"""Agent-managed memory variant of the support scenario.

The harness no longer calls Mubit. A separate MEMORY AGENT (its own LLM
loop) owns every read and write:

- briefing: when a ticket arrives, the memory agent decides what to
  recall and which lessons the support agent should apply
- debrief: when the ticket closes, the memory agent reads the verified
  events (backend verdicts, tier-2 notes, resolution) and decides what
  to store, which lessons to reinforce or fail, and what to retire

The support agent, the ticket datasets, the backend, and the customer
follow-up triggers are shared with support.py unchanged. Guardrails
here only validate the memory agent's tool calls (known lesson ids,
key format, caps); the decisions are the model's.
"""

from __future__ import annotations

import json
import re

import support
from demo import llm_json, MODEL

GATE = support.GATE
KEY_RE = re.compile(r"^[a-z]+:[a-z0-9_-]+$")


# ---------------------------------------------------------------------------
# Raw SDK helpers (also used for the UI board refresh).
# ---------------------------------------------------------------------------

def compact_recall(client, query: str, limit: int = 16) -> dict[str, dict]:
    """Run one recall and compact the evidence to one row per lesson key
    (newest ingested_at wins — remember(upsert_key) appends)."""
    out = client.recall(
        query=query, limit=limit, entry_types=["lesson"], evidence_only=True,
        mode="direct_bypass", prefer_current_run=True, include_working_memory=False,
    )
    rows: dict[str, dict] = {}
    for e in out.get("evidence") or []:
        text = (e.get("content") or e.get("text") or "").strip()
        m = support.TAG.match(text)
        if not m:
            continue
        key, body = m.group(1), m.group(2).strip()
        try:
            meta = json.loads(e.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            meta = {}
        row = {
            "key": key, "id": e.get("id"), "text": body,
            "confidence": float(meta.get("confidence", 0.5)),
            "fresh": "confidence" not in meta,
            "reinforcement": meta.get("reinforcement_count"),
            "stored_at": meta.get("ingested_at") or "",
            "window": meta.get("window"),
        }
        prev = rows.get(key)
        if prev is None or row["stored_at"] > prev["stored_at"]:
            rows[key] = row
    return rows


# ---------------------------------------------------------------------------
# The memory agent.
# ---------------------------------------------------------------------------

BRIEF_SYSTEM = (
    "You are the memory agent on a support desk for {product}. You manage the "
    "team's long-term memory, stored in Mubit. A support ticket just arrived; "
    "brief the support agent with what memory holds about it.\n"
    "Tool: recall(query, limit) — searches stored lessons. Each result row "
    "has key, text, confidence, id, window.\n"
    "Confidence bands: >=0.75 trusted, >=0.5 applied, <0.5 quarantined. Every "
    "lesson at or above 0.5 is usable — brief it when its topic matches the "
    "ticket. Never brief a lesson below 0.5.\n"
    "The ticket text usually does not name the customer's plan; the support "
    "agent discovers it from the account lookup. So brief every policy lesson "
    "whose topic matches and let each lesson's own text say which plan it "
    "covers — do not guess the plan yourself.\n"
    "First call recall with a query you build from the ticket's topic. When "
    "you have the results, set done=true, list in apply_keys the keys the "
    "support agent should apply on THIS ticket, and summarize the briefing "
    "in one line. An empty apply_keys is correct when memory holds nothing "
    "on the ticket's topic.\n"
    "apply_keys is the only channel to the support agent: a lesson you do "
    "not list there is invisible on this ticket, no matter what the summary "
    "says. List every usable lesson whose topic matches the ticket.\n"
    'Return only JSON: {"actions": [{"tool": "recall", "args": {"query": '
    '"...", "limit": 8}}], "apply_keys": ["..."], "summary": "...", '
    '"done": true|false}'
)

DEBRIEF_SYSTEM = (
    "You are the memory agent on a support desk for {product}. A ticket just "
    "closed; decide what the team's memory should keep from it. Only verified "
    "events may create or change memory: backend tool verdicts, tier-2 notes, "
    "and the ticket's resolution — never the agent's or customer's prose.\n"
    "Tools:\n"
    "- remember(key, text, window): store or update one lesson under a stable "
    "key such as kb:invoice-location, fix:error-1017, policy:refund-monthly. "
    "Write text a support agent can apply on the next ticket; keep exact menu "
    "paths, button names, error codes, plan names, and numbers verbatim. Pass "
    "window (days, integer) only for policy:refund-* lessons.\n"
    "- record_outcome(lesson_id, outcome, rationale): report how an APPLIED "
    "lesson fared — 'success' when it helped resolve the ticket, 'failure' "
    "when a verified result contradicted it.\n"
    "- delete_lesson(lesson_id): retire a contradicted lesson. Store the "
    "corrected lesson with remember under the same key.\n"
    "Rules: one lesson per key.\n"
    "- A refund verdict names its plan and window. If memory holds no "
    "policy:refund-<plan> lesson for that exact plan, remember one from the "
    "verdict — a denial teaches the window as much as an acceptance does.\n"
    "- A lesson is contradicted only by a verified event about the SAME key: "
    "the plan named in the verdict must match the plan in the lesson's key, "
    "in both directions — a monthly verdict says nothing about "
    "policy:refund-annual, and an annual verdict says nothing about "
    "policy:refund-monthly. Each plan has its own window; different windows "
    "on different plans are not a contradiction. Only on a real "
    "contradiction (same plan, accepted N days out with the stored window "
    "smaller than N): record failure on that lesson, delete it, and "
    "remember the corrected window from the verdict.\n"
    "- When a ticket merely confirms a lesson (the verdict or the resolution "
    "matches what the lesson already says), record_outcome success is the "
    "ONLY update. Never delete or re-store a confirmed lesson — a new "
    "example is not a change; only a different window or different steps "
    "is.\n"
    "- Leave lessons that were not applied and not contradicted untouched. "
    "If the ticket verified nothing new, do nothing.\n"
    "Only the actions array changes memory — an update described in the "
    "summary but missing from actions never happens. The summary must "
    "describe only actions you actually issued.\n"
    'Return only JSON: {"actions": [{"tool": "...", "args": {...}}], '
    '"summary": "...", "done": true|false} — set done=true when memory is up '
    "to date, with a one-line summary of what changed."
)


class MemoryAgent:
    def __init__(self, client, run_id: str, emit_evt) -> None:
        self.client = client
        self.run_id = run_id
        self.emit_evt = emit_evt          # emit_evt(dict) -> UI event stream
        self.known: dict[str, dict] = {}  # key -> last recalled row (has id)
        client.set_run_id(run_id)

    # -- plumbing ----------------------------------------------------------

    def work(self, kind: str, text: str) -> None:
        self.emit_evt({"t": "memwork", "kind": kind, "text": text})

    def board(self) -> dict[str, dict]:
        return compact_recall(self.client, support.QUERY)

    def _loop(self, system: str, parts: list[str], exec_action, max_turns: int) -> dict:
        """Run the memory agent's JSON tool loop; returns the final turn."""
        transcript = list(parts)
        out: dict = {}
        for turn in range(1, max_turns + 1):
            self.work("llm", f"turn {turn} — {MODEL}")
            raw = llm_json(system, "\n".join(transcript) + "\nReturn your next JSON turn.")
            out = raw if isinstance(raw, dict) else {}
            results = []
            for a in (out.get("actions") or [])[:4]:
                tool = str(a.get("tool") or "")
                args = a.get("args") or {}
                results.append({"tool": tool, "result": exec_action(tool, args)})
            if results:
                transcript.append("Tool results: " + json.dumps(results))
            if out.get("done") or not results:
                break
        return out

    # -- briefing ----------------------------------------------------------

    def briefing(self, ticket: support.Ticket) -> dict[str, dict]:
        self.emit_evt({"t": "mem", "state": "open", "phase": "briefing"})
        recalled: dict[str, dict] = {}

        def act(tool: str, args: dict):
            if tool != "recall":
                return {"error": f"unknown tool {tool}"}
            query = str(args.get("query") or support.QUERY)[:200]
            limit = min(max(int(args.get("limit") or 8), 4), 16)
            rows = compact_recall(self.client, query, limit)
            recalled.update(rows)
            self.known.update(rows)
            if rows:
                names = "  ".join(f"[{k} {r['confidence']:.2f}]" for k, r in sorted(rows.items()))
                self.work("recall", f'recall(query="{query[:60]}") -> {len(rows)} lessons  {names}')
            else:
                self.work("recall", f'recall(query="{query[:60]}") -> nothing stored (cold start)')
            return [{"key": r["key"], "text": r["text"], "confidence": r["confidence"],
                     "id": r["id"], "window": r["window"]} for r in rows.values()]

        out = self._loop(
            BRIEF_SYSTEM.replace("{product}", support.CURRENT.product),
            [f"Ticket {ticket.id} ({ticket.kind}) from {ticket.customer}:",
             ticket.opening],
            act, max_turns=3,
        )
        apply: dict[str, dict] = {}
        for key in out.get("apply_keys") or []:
            row = recalled.get(key)
            if row is None:
                continue
            if row["confidence"] < GATE:
                self.work("recall", f"[{key}] is below the {GATE} gate — not briefed")
                continue
            apply[key] = row
        summary = (out.get("summary") or "").strip() or (
            f"apply {', '.join(sorted(apply))}" if apply
            else "memory holds nothing relevant — the agent goes in cold")
        self.emit_evt({"t": "mem", "state": "close", "text": summary})
        return apply

    # -- debrief -----------------------------------------------------------

    def debrief(self, ticket: support.Ticket, applied: dict[str, dict],
                events: list[dict], tool_results: list[dict],
                resolved: bool, escalated: bool) -> dict:
        self.emit_evt({"t": "mem", "state": "open", "phase": "debrief"})
        id2key = {r["id"]: k for k, r in self.known.items() if r.get("id")}
        stored: list[str] = []
        deleted: list[str] = []

        def act(tool: str, args: dict):
            if tool == "remember":
                key = str(args.get("key") or "").strip()
                text = str(args.get("text") or "").strip()
                if not KEY_RE.match(key) or not text:
                    return {"error": "remember needs key like kb:some-topic and text"}
                if len(stored) >= 3:
                    return {"error": "at most 3 lessons per ticket"}
                meta = {"key": key}
                window = args.get("window")
                if isinstance(window, (int, float)) and key.startswith("policy:"):
                    meta["window"] = int(window)
                self.client.remember(
                    content=f"[{key}] {text}"[:800], intent="lesson",
                    lesson_type="success", lesson_scope="run",
                    lesson_importance="high", upsert_key=key, metadata=meta, wait=True,
                )
                stored.append(key)
                self.work("store", f'stored lesson [{key}] "{text[:110]}"')
                return {"ok": True, "key": key}
            if tool == "record_outcome":
                lid = str(args.get("lesson_id") or "")
                key = id2key.get(lid)
                if key is None:
                    return {"error": "unknown lesson_id — use an id from the briefing"}
                ok = str(args.get("outcome") or "") == "success"
                r = self.client.record_outcome(
                    reference_id=lid, outcome="success" if ok else "failure",
                    signal=0.9 if ok else -0.9,
                    rationale=str(args.get("rationale") or "")[:300],
                )
                conf = r.get("updated_confidence")
                word = "reinforced" if ok else "contradicted"
                if conf is not None:
                    prev = self.known[key]["confidence"]
                    self.work("outcome", f"[{key}] {word}: confidence {prev:.2f} -> {conf:.2f}")
                return {"ok": True, "key": key, "updated_confidence": conf}
            if tool == "delete_lesson":
                lid = str(args.get("lesson_id") or "")
                key = id2key.get(lid)
                if key is None:
                    return {"error": "unknown lesson_id — use an id from the briefing"}
                self.client.delete_lesson({"run_id": self.run_id, "lesson_id": lid})
                deleted.append(key)
                self.work("retire", f"retired [{key}] — no longer matches a verified outcome")
                return {"ok": True, "key": key}
            return {"error": f"unknown tool {tool}"}

        parts = [
            f"Ticket {ticket.id} ({ticket.kind}) from {ticket.customer} just closed.",
            f"Outcome: {'resolved' if resolved else 'unresolved'}, "
            f"escalated to tier 2: {'yes' if escalated else 'no'}.",
            "Lessons the briefing applied on this ticket: "
            + (json.dumps([{"key": k, "id": r["id"], "confidence": r["confidence"],
                            "window": r["window"]} for k, r in applied.items()]) or "[]"),
            "All keys memory currently holds: " + json.dumps(sorted(self.known)),
            "Verified events on this ticket: "
            + json.dumps([{k: v for k, v in e.items() if k != "meta"} for e in events]),
            "Backend tool results: " + json.dumps(tool_results)[:1800],
            "Update memory now.",
        ]
        out = self._loop(DEBRIEF_SYSTEM.replace("{product}", support.CURRENT.product),
                         parts, act, max_turns=3)
        summary = (out.get("summary") or "").strip() or (
            "memory updated" if (stored or deleted) else "nothing to change")
        self.emit_evt({"t": "mem", "state": "close", "text": summary})
        replaced = [k for k in stored if k in deleted]
        return {"stored": [k for k in stored if k not in deleted],
                "replaced": replaced, "deleted": deleted}


# ---------------------------------------------------------------------------
# Ticket orchestration — the support loop from support.py with all memory
# decisions handed to the memory agent.
# ---------------------------------------------------------------------------

def run_ticket(ticket: support.Ticket, mem: MemoryAgent, emit, stop=None) -> dict:
    emit({"t": "ticket_start", "id": ticket.id, "title": ticket.opening[:60]})
    emit({"t": "customer", "name": ticket.customer, "text": ticket.opening})

    convo = [{"who": ticket.customer, "text": ticket.opening}]
    tool_log: list[dict] = []
    events: list[dict] = []
    applied = mem.briefing(ticket)
    if applied:
        emit({"t": "chips", "items": [{"kind": "recalled", "label": k}
                                      for k in sorted(applied)]})

    note_sent = False
    pushback_sent = False
    miss_sent = False
    escalated = False
    final_reply = ""
    turns = 0
    replies = 0
    note: str | None = None
    note_nudge: str | None = None
    dispute_nudge: str | None = None

    def send_note() -> None:
        nonlocal note, note_sent, escalated, note_nudge
        escalated = True
        note_sent = True
        emit({"t": "work", "kind": "escalate", "text": "escalated to tier 2"})
        note = support.CURRENT.tier2_notes[ticket.note_key]
        convo.append({"who": "Tier-2", "text": note})
        note_nudge = ("Tier 2 has sent the resolution note above. Write the "
                      "reply to the customer now, using the note's exact "
                      "steps. Do not escalate again and do not call tools.")
        emit({"t": "note", "text": note})
        events.append({"key": ticket.note_key, "evidence": note})

    while turns < support.MAX_TURNS:
        if stop is not None and stop.is_set():
            break
        turns += 1
        emit({"t": "work", "kind": "llm", "text": f"agent turn {turns} — {MODEL}"})
        out = support.agent_turn(convo, applied, tool_log, note,
                                 dispute_nudge or note_nudge)
        note = None

        refund_result = None
        escalate_called = False
        for action in out.get("actions", [])[:4]:
            tool = str(action.get("tool") or "")
            args = action.get("args") or {}

            def skip(reason: str) -> None:
                emit({"t": "work", "kind": "tool",
                      "text": f"{tool} repeat skipped — {reason}"})
                if not any(t["tool"] == tool and t["args"] == args
                           and "skipped" in t["result"] for t in tool_log):
                    tool_log.append({"tool": tool, "args": args,
                                     "result": {"skipped": reason}})

            if tool == "lookup_account" and any(
                    t["tool"] == tool and t["args"] == args
                    and "skipped" not in t["result"] for t in tool_log):
                skip("result already on the ticket; reuse it")
                continue
            if tool == "credit" and any(
                    t["tool"] == "credit" and "skipped" not in t["result"]
                    for t in tool_log):
                skip("one goodwill credit per ticket")
                continue
            if tool == "refund" and any(
                    t["tool"] == "refund" and t["result"].get("ok")
                    and t["args"].get("order_id") == args.get("order_id")
                    for t in tool_log):
                skip("this order is already refunded")
                continue
            if tool == "escalate" and note_sent:
                skip("the ticket is already with tier 2; answer from its note")
                continue
            result = support.tool_call(tool, args)
            tool_log.append({"tool": tool, "args": args, "result": result})
            emit({"t": "work", "kind": "tool",
                  "text": f"{tool}({json.dumps(args)}) -> {json.dumps(result)[:130]}"})
            if tool == "escalate":
                escalate_called = True
            if tool == "refund":
                refund_result = result
                if result.get("denied"):
                    events.append({
                        "key": f"policy:refund-{result['plan']}",
                        "evidence": f"refund denied: {result['reason']}",
                        "meta": {"window": result["window"]},
                    })
                elif result.get("ok"):
                    events.append({
                        "key": f"policy:refund-{result['plan']}",
                        "evidence": f"refund accepted at day {result['age_days']} on "
                                    f"the {result['plan']} plan (window "
                                    f"{result['window']} days)",
                        "meta": {"window": result["window"]},
                    })

        if refund_result is not None:
            dispute_nudge = None

        reply = (out.get("reply") or "").strip()
        if reply and note_sent:
            note_nudge = None
        resolution = out.get("resolution")
        if escalate_called:
            resolution = "escalate"

        if resolution == "pending":
            if turns >= 3 and ticket.note_key and not note_sent:
                send_note()
            continue

        if resolution == "escalate":
            if ticket.note_key and not note_sent:
                if reply:
                    final_reply = reply
                    replies += 1
                    convo.append({"who": "Agent", "text": reply})
                    emit({"t": "reply", "text": reply})
                send_note()
                continue
            resolution = "resolved"

        final_reply = reply
        if reply:
            replies += 1
            convo.append({"who": "Agent", "text": reply})
            emit({"t": "reply", "text": reply})

        low = reply.lower()
        tokens_ok = all(t in low for t in ticket.verify_tokens) if ticket.verify_tokens else True
        refund_open = (ticket.order_id and support.would_refund(ticket.order_id))

        if refund_open and ticket.pushback and not pushback_sent:
            pushback_sent = True
            dispute_nudge = ("The customer's dispute contains a concrete, "
                             "checkable claim. Call the refund tool for the "
                             "order now — the billing system's verdict is "
                             "authoritative and overrides your learned "
                             "knowledge.")
            convo.append({"who": ticket.customer, "text": ticket.pushback})
            emit({"t": "customer", "name": ticket.customer, "text": ticket.pushback})
            continue
        if not tokens_ok and ticket.miss_reply and not miss_sent:
            miss_sent = True
            convo.append({"who": ticket.customer, "text": ticket.miss_reply})
            emit({"t": "customer", "name": ticket.customer, "text": ticket.miss_reply})
            if ticket.note_key and not note_sent and ticket.note_key not in applied:
                send_note()
            continue
        if not tokens_ok and miss_sent and ticket.note_key and not note_sent:
            send_note()
            continue
        break

    tokens_ok = all(t in final_reply.lower() for t in ticket.verify_tokens) if ticket.verify_tokens else True
    refund_done = bool(ticket.order_id) and (
        support.CURRENT.orders[ticket.order_id]["refunded"]
        or not support.would_refund(ticket.order_id)
    )
    resolved = tokens_ok and (refund_done if ticket.order_id else True) and bool(final_reply)
    first_touch = (resolved and replies == 1 and not escalated
                   and not pushback_sent and not miss_sent)

    # The memory agent decides everything that changes memory.
    backend_facts = [t for t in tool_log if "skipped" not in t["result"]]
    result = mem.debrief(ticket, applied, events, backend_facts, resolved, escalated)
    chips = ([{"kind": "replaced", "label": k} for k in sorted(result["replaced"])]
             + [{"kind": "learned", "label": k} for k in sorted(result["stored"])])
    if chips:
        emit({"t": "chips", "items": chips})
    if (resolved and not escalated and ticket.note_key
            and ticket.note_key in applied):
        emit({"t": "chips", "items": [{"kind": "avoided", "label": "escalation avoided"}]})

    if resolved and ticket.confirm:
        emit({"t": "customer", "name": ticket.customer, "text": ticket.confirm})

    stats = {"resolved": resolved, "first_touch": first_touch,
             "escalated": escalated, "turns": turns,
             "lessons_stored": len(result["stored"]) + len(result["replaced"])}
    emit({"t": "ticket_done", "id": ticket.id, **stats})
    return stats
