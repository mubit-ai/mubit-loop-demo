"""UI server for the feedback-loop demo.

Serves ui.html and streams one chat message at a time over SSE. The
agent loop is the one in demo.py — this file only routes its output:

- demo.line() is redirected into the event stream (the chat transcript)
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
import expense_api

app = FastAPI()

_lock = threading.Lock()   # one task at a time
_sink = None               # active event sink (queue or list collector)
_boot: list = []           # mubit events captured while no sink is active
_counter = {"n": 0}


def _emit_mubit(evt: dict) -> None:
    if _sink is not None:
        _sink.put(evt)
    else:
        _boot.append(evt)


def _emit_line(channel: str, arrow: str, text: str, color: str = "0") -> None:
    if _sink is not None:
        _sink.put({"t": "line", "ch": channel.strip(), "arrow": arrow.strip(), "text": text})


loop.line = _emit_line  # the terminal display becomes the chat transcript


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


def _new_memory() -> loop.Memory:
    from mubit import Client

    inner = Client(
        endpoint=os.environ.get("MUBIT_ENDPOINT", "http://127.0.0.1:3970"),
        api_key=os.environ["MUBIT_API_KEY"],
        transport="http",
    )
    return loop.Memory(client=TracedClient(inner, _emit_mubit))


MEMORY = _new_memory()


def _board() -> list[dict]:
    tc = MEMORY.client
    tc.enabled = False  # a UI refresh, not an agent call — keep the pane honest
    try:
        rules = MEMORY.recall_rules(quiet=True)
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
        for k, r in sorted(rules.items())
    ]


def _sse(evt: dict) -> str:
    return f"data: {json.dumps(evt)}\n\n"


class ListCollector:
    def __init__(self):
        self.items = []

    def put(self, evt):
        self.items.append(evt)


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "ui.html"))


@app.get("/api/state")
def state():
    boot, _boot[:] = _boot[:], []
    return {
        "run_id": MEMORY.run_id,
        "model": loop.MODEL,
        "policy": expense_api.POLICY_VERSION,
        "board": _board(),
        "boot_events": boot,
    }


@app.get("/api/chat")
def chat(text: str):
    global _sink
    if not _lock.acquire(blocking=False):
        def busy():
            yield _sse({"t": "error", "msg": "a task is already running"})
            yield _sse({"t": "task_done"})
        return StreamingResponse(busy(), media_type="text/event-stream")

    q: queue.Queue = queue.Queue()
    _sink = q
    _counter["n"] += 1
    task = loop.Task(f"U-{_counter['n']:02d}", text.strip()[:400])
    stats = {"tasks": 0, "first_try": 0, "tries": 0}

    def work():
        try:
            q.put({"t": "task_start", "id": task.id, "text": task.text})
            loop.handle(task, MEMORY, stats)
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
            _sink = None
            yield _sse({"t": "board", "rules": _board()})
            yield _sse({"t": "task_done"})
        finally:
            _sink = None
            _lock.release()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.post("/api/policy")
def policy(body: dict):
    version = 2 if body.get("version") == 2 else 1
    expense_api.set_policy(version)
    return {"policy": version}


@app.post("/api/reflect")
def reflect():
    global _sink
    if not _lock.acquire(blocking=False):
        return JSONResponse({"error": "a task is already running"}, status_code=409)
    collector = ListCollector()
    _sink = collector
    try:
        MEMORY.reflect()
    finally:
        _sink = None
        _lock.release()
    return {"events": collector.items, "board": _board()}


@app.post("/api/reset")
def reset():
    global MEMORY, _sink
    if not _lock.acquire(blocking=False):
        return JSONResponse({"error": "a task is already running"}, status_code=409)
    try:
        _boot.clear()
        expense_api.set_policy(1)
        _counter["n"] = 0
        MEMORY = _new_memory()
        boot, _boot[:] = _boot[:], []
        return {"run_id": MEMORY.run_id, "policy": 1, "boot_events": boot}
    finally:
        _lock.release()
