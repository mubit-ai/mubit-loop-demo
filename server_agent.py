"""UI server for the agent-managed-memory variant (ui_agent.html).

Same shape as server.py, but memory is owned by agentic.MemoryAgent:
every Mubit call in the side pane was decided by the memory agent's own
tool loop, not by the harness.

Run:  ./run_ui_agent.sh    then open http://127.0.0.1:7875
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
import agentic

app = FastAPI()

_lock = threading.Lock()
_sink = None
_boot: list = []
_counter = {"n": 0}


def _emit(evt: dict) -> None:
    if _sink is not None:
        _sink.put(evt)
    elif evt.get("t") == "mubit":
        _boot.append(evt)


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


def _new_memory() -> agentic.MemoryAgent:
    from mubit import Client

    inner = Client(
        endpoint=os.environ.get("MUBIT_ENDPOINT", "http://127.0.0.1:3970"),
        api_key=os.environ["MUBIT_API_KEY"],
        transport="http",
    )
    run_id = f"support-agentic-{uuid.uuid4().hex[:8]}"
    return agentic.MemoryAgent(TracedClient(inner, _emit), run_id, _emit)


MEMORY = _new_memory()


def _board() -> list[dict]:
    tc = MEMORY.client
    tc.enabled = False  # a UI refresh, not an agent decision — keep the pane honest
    try:
        lessons = MEMORY.board()
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


def _stream(worker) -> StreamingResponse:
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
    return FileResponse(os.path.join(os.path.dirname(__file__), "ui_agent.html"))


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
            stats = agentic.run_ticket(tk, MEMORY, q.put, stop)
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
        agentic.run_ticket(ticket, MEMORY, q.put, stop)

    return _stream(worker)


@app.post("/api/policy")
def policy(body: dict):
    version = 2 if body.get("version") == 2 else 1
    support.set_policy(version)
    return {"policy": version}


@app.post("/api/reset")
def reset():
    global MEMORY
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
