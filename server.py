"""UI server for the feedback-loop demo (customer-support scenario).

Serves ui.html and streams ticket playback over SSE. The scenario is
support.py; this file only routes its output:

- SupportMemory work lines become the compact activity block inside
  each agent chat bubble
- the Mubit client is wrapped in TracedClient, which records every SDK
  method call with its latency, request, and response (the side pane)

Run:  ./run_ui.sh          then open http://127.0.0.1:7874
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import demo as loop
import support

app = FastAPI()

_lock = threading.Lock()   # one playback at a time
_sink = None               # active event sink (queue or list collector)
_boot: list = []           # mubit events captured while no sink is active
_counter = {"n": 0}


def _emit_mubit(evt: dict) -> None:
    if _sink is not None:
        _sink.put(evt)
    else:
        _boot.append(evt)


def _emit_work(kind: str, text: str) -> None:
    if _sink is not None:
        _sink.put({"t": "work", "kind": kind, "text": text})


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return json.loads(json.dumps(obj, default=str))


class TracedClient:
    """Wraps the Mubit SDK client. Every method call is timed and
    reported with the data that went in and the data that came back."""

    def __init__(self, inner, emit):
        self._inner = inner
        self._emit = emit
        self._seq = 0
        self.enabled = True

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            if not self.enabled:
                return attr(*args, **kwargs)
            self._seq += 1
            seq = self._seq
            t0 = time.perf_counter()
            response, error = None, None
            try:
                response = attr(*args, **kwargs)
                return response
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                ms = round((time.perf_counter() - t0) * 1000, 1)
                request = {}
                if args:
                    request["args"] = _jsonable(list(args))
                if kwargs:
                    request.update(_jsonable(kwargs))
                self._emit({
                    "t": "mubit", "seq": seq, "method": name, "ms": ms,
                    "request": request, "response": _jsonable(response),
                    "error": error,
                })

        return wrapper


def _new_memory() -> support.SupportMemory:
    from mubit import Client

    inner = Client(
        endpoint=os.environ.get("MUBIT_ENDPOINT", "http://127.0.0.1:3970"),
        api_key=os.environ["MUBIT_API_KEY"],
        transport="http",
    )
    run_id = f"support-demo-{uuid.uuid4().hex[:8]}"
    return support.SupportMemory(TracedClient(inner, _emit_mubit), run_id, _emit_work)


MEMORY = _new_memory()


def _board() -> list[dict]:
    tc = MEMORY.client
    tc.enabled = False  # a UI refresh, not an agent call — keep the pane honest
    try:
        lessons = MEMORY.recall(quiet=True)
    finally:
        tc.enabled = True
    return [
        {
            "key": k,
            "text": r["text"],
            "confidence": round(r["confidence"], 2),
            "reinforcement": r["reinforcement"],
            "fresh": r["fresh"],
        }
        for k, r in sorted(lessons.items())
    ]


def _sse(evt: dict) -> str:
    return f"data: {json.dumps(evt)}\n\n"


class ListCollector:
    def __init__(self):
        self.items = []

    def put(self, evt):
        self.items.append(evt)


def _stream(worker) -> StreamingResponse:
    """Run `worker(q, stop)` in a thread; stream its queue as SSE."""
    global _sink
    if not _lock.acquire(blocking=False):
        def busy():
            yield _sse({"t": "error", "msg": "a run is already in progress"})
            yield _sse({"t": "done"})
        return StreamingResponse(busy(), media_type="text/event-stream")

    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    _sink = q

    def work():
        try:
            worker(q, stop)
        except Exception as exc:
            q.put({"t": "error", "msg": str(exc)})
        finally:
            q.put({"t": "__end__"})

    threading.Thread(target=work, daemon=True).start()

    def gen():
        global _sink
        try:
            while True:
                try:
                    evt = q.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                if evt.get("t") == "__end__":
                    break
                yield _sse(evt)
                if evt.get("t") == "ticket_done":
                    # tracing is off during the board read, so no pane
                    # events leak from this UI refresh
                    yield _sse({"t": "board", "rules": _board()})
            _sink = None
            yield _sse({"t": "done"})
        finally:
            stop.set()
            _sink = None
            _lock.release()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "ui.html"))


@app.get("/api/state")
def state():
    boot, _boot[:] = _boot[:], []
    return {
        "run_id": MEMORY.run_id,
        "model": loop.MODEL,
        "policy": support.POLICY_VERSION,
        "dataset": support.CURRENT.id,
        "datasets": [{"id": d.id, "label": d.label} for d in support.DATASETS.values()],
        "board": _board(),
        "boot_events": boot,
    }


@app.post("/api/dataset")
def switch_dataset(body: dict):
    global MEMORY
    if not _lock.acquire(blocking=False):
        return JSONResponse({"error": "a run is already in progress"}, status_code=409)
    try:
        ds = support.use_dataset(str(body.get("id") or "orbit"))
        _boot.clear()
        _counter["n"] = 0
        MEMORY = _new_memory()
        boot, _boot[:] = _boot[:], []
        return {"run_id": MEMORY.run_id, "policy": 1, "dataset": ds.id,
                "label": ds.label, "boot_events": boot}
    finally:
        _lock.release()


@app.get("/api/run")
def run_all():
    def worker(q, stop):
        ds = support.CURRENT
        totals = {"tickets": 0, "resolved": 0, "first_touch": 0,
                  "escalated": 0, "lessons": 0}
        q.put({"t": "run_start", "total": len(ds.tickets)})
        for tk in ds.tickets:
            if stop.is_set():
                break
            if tk.id == ds.policy_change_before:
                support.set_policy(2)
                q.put({"t": "policy_change", "version": 2,
                       "text": ds.policy_change_text})
            stats = support.run_ticket(tk, MEMORY, q.put, stop)
            totals["tickets"] += 1
            totals["resolved"] += int(stats["resolved"])
            totals["first_touch"] += int(stats["first_touch"])
            totals["escalated"] += int(stats["escalated"])
            totals["lessons"] += stats["lessons_stored"]
            time.sleep(0.8)
        q.put({"t": "run_done", **totals})

    return _stream(worker)


@app.get("/api/chat")
def chat(text: str):
    _counter["n"] += 1
    ticket = support.adhoc_ticket(_counter["n"], text.strip()[:500])

    def worker(q, stop):
        support.run_ticket(ticket, MEMORY, q.put, stop)

    return _stream(worker)


@app.post("/api/policy")
def policy(body: dict):
    version = 2 if body.get("version") == 2 else 1
    support.set_policy(version)
    return {"policy": version}


@app.post("/api/reflect")
def reflect():
    global _sink
    if not _lock.acquire(blocking=False):
        return JSONResponse({"error": "a run is already in progress"}, status_code=409)
    collector = ListCollector()
    _sink = collector
    try:
        _emit_work("reflect", "reflect(last_n_items=30) — server-side distillation")
        try:
            r = MEMORY.client.reflect(last_n_items=30)
            lessons = r.get("lessons")
            n = len(lessons) if isinstance(lessons, list) else r.get("lessons_created", 0)
            _emit_work("reflect", f"server distilled {n} lesson(s) from the recorded outcomes")
            if isinstance(lessons, list):
                for l in lessons[:2]:
                    if isinstance(l, dict) and l.get("content"):
                        _emit_work("reflect", f'e.g. "{l["content"][:100]}"')
        except Exception as exc:
            _emit_work("reflect", f"skipped ({exc})")
    finally:
        _sink = None
        _lock.release()
    return {"events": collector.items, "board": _board()}


@app.post("/api/reset")
def reset():
    global MEMORY, _sink
    if not _lock.acquire(blocking=False):
        return JSONResponse({"error": "a run is already in progress"}, status_code=409)
    try:
        _boot.clear()
        support.reset_backend()
        _counter["n"] = 0
        MEMORY = _new_memory()
        boot, _boot[:] = _boot[:], []
        return {"run_id": MEMORY.run_id, "policy": 1, "boot_events": boot}
    finally:
        _lock.release()
