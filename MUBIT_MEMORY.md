# How this demo uses Mubit memory

Every piece of product knowledge the agents show is stored in and
retrieved from Mubit. The application keeps no state of its own between
tickets. This document lists the SDK methods in use, what each one does,
and where each call site is. Line numbers are correct as of commit
`040c202`; the cited files are not changed by later commits unless noted.

## Where Mubit appears

| Demo | Files | Integration surface |
| --- | --- | --- |
| Support UI v1 (`./run_ui.sh`, :7874) | `support.py`, `server.py` | `SupportMemory` class — four methods, one SDK call each (`support.py:535-607`) |
| Support UI v2 (`./run_ui_agent.sh`, :7875) | `agentic.py`, `server_agent.py` | Same four SDK calls, exposed to the model as tools (`agentic.py:229-275`) |
| Terminal expense demo (`./run_demo.sh`) | `demo.py` | `Memory` class (`demo.py:137-271`) |

The client is constructed once per run from two environment variables
(`server.py:101-108`): `MUBIT_ENDPOINT` and `MUBIT_API_KEY`. Each run
gets its own `run_id` (`server.py:109`), and `set_run_id` scopes every
later call to it (`support.py:540`). The support agent and the backend
simulator never import the SDK; memory is a wrapper around them.

## SDK methods in use

### `recall` — read memory

```python
client.recall(query=..., limit=16, entry_types=["lesson"], evidence_only=True,
              mode="direct_bypass", prefer_current_run=True,
              include_working_memory=False)
```

Returns evidence rows for the query. `mode="direct_bypass"` with
`evidence_only=True` is the direct retrieval path: no server-side LLM,
typical latency tens of milliseconds. Lesson confidence rides inside
`metadata_json`, which arrives as a JSON string — parse it and read
`confidence` (absent until the first `record_outcome`; treat absent as
0.5). Call sites:

- `support.py:542-575` — `SupportMemory.recall`. One call, then ~20
  lines that compact the evidence into `{key: {text, confidence, id}}`.
- `support.py:732` — the single line at the top of every ticket where
  memory enters the prompt; the 0.5 confidence gate is applied at
  `support.py:736`.
- `agentic.py:34-66` — `compact_recall`, the same compaction as a free
  function.
- `agentic.py:182-196` — v2 briefing: the model writes its own query
  strings and this handler executes them.
- `demo.py:149-201` — `Memory.recall_rules` in the terminal demo.

### `remember` — write a lesson

```python
client.remember(content="[kb:invoice-location] ...", intent="lesson",
                lesson_type="success", lesson_scope="run",
                lesson_importance="high", upsert_key=key,
                metadata={"key": key, ...}, wait=True)
```

Stores one lesson. `upsert_key` appends a new row under the key rather
than replacing the old one, so readers deduplicate client-side by the
newest `ingested_at` in metadata (`support.py:566-568`,
`agentic.py:60-63`), and a replacement is completed by deleting the old
row (see `delete_lesson`). Call sites:

- `support.py:577-587` — `SupportMemory.store`; called from the ticket
  close-out at `support.py:948-954` after an LLM pass distills one
  lesson per verified event.
- `agentic.py:241-245` — v2: the model calls a `remember` tool; the
  handler validates the key format and a three-lessons-per-ticket cap,
  then makes this call.
- `demo.py:203-215` — `Memory.store_rule`.

### `record_outcome` — close the loop

```python
client.record_outcome(reference_id=lesson_id, outcome="success" | "failure",
                      signal=0.9 | -0.9, rationale=...)
```

Reports how an applied lesson fared. The signal is signed: the server
adds `signal * 0.1` to the lesson's confidence, clamped to [0, 1], so
+0.9 moves it up by 0.09 and -0.9 down by the same amount. The response
carries `updated_confidence`. Call sites:

- `support.py:589-600` — `SupportMemory.outcome`; called from the
  close-out at `support.py:936-946` (contradicted lessons get the
  negative signal, lessons that carried a resolved ticket get the
  positive one).
- `agentic.py:249-265` — v2: the model reports the outcome; the handler
  maps it to the signed signal.
- `demo.py:217-234` — `Memory.report_outcome`.

### `delete_lesson` — retire a contradicted lesson

```python
client.delete_lesson({"run_id": run_id, "lesson_id": lesson_id})
```

Removes one lesson row. The payload is a dict and `run_id` must be in
it. One contradiction moves confidence by 0.09, which cannot push a
trusted lesson below the apply gate on its own — so a lesson proven
wrong by a verified backend result is retired outright and the corrected
lesson is stored under the same key. Call sites:

- `support.py:602-607` — `SupportMemory.retire`; called at
  `support.py:942` when a verified refund result contradicts the applied
  policy lesson.
- `agentic.py:266-274` — v2: the model decides the retirement.
- `demo.py:236-243` — `Memory.retire_rule`.

### `record_step_outcome` — per-step signals (v1 and terminal demo only)

```python
client.record_step_outcome(step_id=..., step_name=..., outcome=...,
                           signal=..., rationale=..., directive_hint=...)
```

Reports individual tool calls and the ticket close as steps
(`support.py:609-614`, called at `support.py:813` and `support.py:929`;
`demo.py:245-254`). This feeds server-side reflection; the v2 demo omits
it so the right-hand pane shows only calls the model decided to make.

### `reflect` — server-side distillation (terminal demo only)

```python
client.reflect(last_n_items=24)
```

Asks the server to distill lessons from recent activity with its own
LLM and merge them into existing lesson ids (`demo.py:256-271`, called
at `demo.py:577`). The support UIs distill client-side instead, so the
transcript shows the exact text being stored.

## The loop in one place

For a single reading path, open `support.py` and read top to bottom:

1. `support.py:732` — recall at ticket start; gate at 736.
2. `support.py:936-946` — after the ticket closes, reinforce or
   contradict each applied lesson; retire the contradicted ones.
3. `support.py:948-954` — distill and store what the ticket taught.

Everything between those points is the support agent and the simulated
backend; none of it touches the SDK.

## The call pane

`server.py:57-98` — `TracedClient` wraps the SDK client with
`__getattr__`, times every method call, and reports
`{method, ms, request, response, error}` to the UI. Board refreshes set
`enabled=False` (`server.py:73-74`) so the pane shows agent-driven calls
only. This wrapper is presentation, not integration; removing it changes
nothing about the loop.
