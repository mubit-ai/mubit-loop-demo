# Mubit feedback-loop demo

The Mubit learning loop in its smallest runnable form. One agent, one
task type, one terminal. No UI.

An agent submits expense reports to an internal API. Nobody gave it the
API's validation rules. The API teaches by rejection; the agent stores
what it learns in Mubit; the next submission starts from what memory
holds. The transcript prints every piece of data that moves:

- `->` lines show data going into Mubit or the API
- `<-` lines show data coming back

## Run

```bash
cp .env.example .env    # fill in MUBIT_ENDPOINT, MUBIT_API_KEY, OPENAI_API_KEY
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_demo.sh           # three scripted acts
./run_demo.sh --step    # pause between acts
./run_demo.sh --chat    # scripted acts, then type your own expenses
./run_demo.sh --fresh-chat   # chat only, cold memory
```

Each run uses a fresh Mubit run id, so each run starts with empty
memory and learns from zero. The model is `gpt-5-mini` (override with
`MODEL=`).

## Chat UI

```bash
./run_ui.sh    # http://127.0.0.1:7874
```

A browser version of the same loop (`server.py` + `ui.html`; the agent
code is demo.py, unchanged). Left: a chat window with the sample
expenses one click away. Right: the lesson board with its confidence
band, and a feed of every SDK call the agent makes — method, latency,
and a one-line summary; a row expands to the full request and response
JSON. Header controls: the API policy switch (the same rule change as
act 3), Reflect (server-side distillation over the recorded outcomes),
and New run (fresh run id, empty memory).

## The three acts

1. **Cold start.** No stored rules. The first submission collects a
   stack of field errors; the retry passes; the agent distills one rule
   per rejected field and stores each as a lesson.
2. **Warm.** New expenses, same API. Stored rules go into every prompt.
   Submissions pass on the first try, and each accepted reuse is
   reported back to the lessons it used — confidence climbs.
3. **The API changes.** The service switches its date format and nobody
   tells the agent. The stored date rule gets contradicted: its
   confidence takes a real hit, the agent stops applying it inside the
   task, retires it once the corrected submission is accepted, and the
   replacement rule re-earns trust from the initial value.

## The loop

```mermaid
flowchart TB
    T[expense arrives] -->|recall| G{stored rules?}
    G -->|"rules >= 0.5 go into the prompt"| A[LLM drafts the submission]
    G -->|"cold: no rules"| A
    A -->|"POST /expenses"| V{API verdict}
    V -->|"422 — record_step_outcome, retry with the errors"| A
    V -->|"201 — record_step_outcome"| W["distill one rule per rejected field
    remember (upsert per rule key)"]
    W --> O["record_outcome per applied rule:
    accepted reuse raises confidence,
    contradiction lowers it and retires the rule"]
    O -.->|next expense starts at recall| T
```

## What goes in, what comes out

| Call | In | Out |
| --- | --- | --- |
| `recall` | a query, `entry_types=["lesson"]` | stored rules with `confidence` (in `metadata_json`) |
| `record_step_outcome` | one outcome per submission attempt: step id, success/failure, signal, rationale | acknowledgement; feeds server-side reflection |
| `remember` | the distilled rule as a lesson, `upsert_key=rule:<field>` | acknowledgement; the text recall returns next time |
| `record_outcome` | the lesson id, outcome, a **signed** signal | `updated_confidence`, `reinforcement_count` |
| `delete_lesson` | run id + lesson id of a contradicted rule | removal; the replacement takes the key |
| `reflect` | a window over recent activity | lessons the server distilled from the recorded outcomes |

## The confidence band

A lesson starts at 0.5. Each `record_outcome` moves it:
`confidence += signal * 0.1`, clamped to [0, 1], and the reinforcement
count steps up or down with the signal's sign.

| Band | Range | Effect |
| --- | --- | --- |
| high — trusted | >= 0.75 | applied in every prompt |
| medium — applied | >= 0.5 | applied in every prompt |
| low — quarantined | < 0.5 | kept in memory, not applied |

The band is not a display value: the recall gate reads it. A rule the
API contradicts takes a negative signal immediately, is dropped for the
rest of that task, and is retired and replaced once the corrected
submission is accepted. The replacement starts back at 0.5 and earns
its way up.

## What keeps the demo honest

- The pass/fail verdict is the API validator's accept or reject —
  deterministic code in `expense_api.py`, not a score over the agent's
  text.
- The agent never reads `expense_api.py`. It sees only the responses
  `submit()` returns.
- The stored rules are written by the agent's own LLM from the error
  messages it received. Nothing is seeded.
- Act 2's first-try acceptances depend on the stored rules alone; the
  cold attempt in act 1 shows what the same model does without them.

`sample_run.txt` is one full captured run.
