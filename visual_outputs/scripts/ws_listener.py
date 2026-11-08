"""
Minimal WebSocket client for the backend's /ws endpoint. Records every message
the backend broadcasts (system_state, training_status, training_progress,
performance_update, training_complete, plc_code, ...) to a JSONL file.

This demonstrates the backend half of the "real-time" link independently of
the browser frontend (whose WebSocketManager never loads, see report).

Usage:
    .venv\\Scripts\\python.exe visual_outputs\\scripts\\ws_listener.py --url ws://127.0.0.1:8001/ws
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets

OUT = Path(__file__).resolve().parents[1] / "run" / "ws_messages.jsonl"


async def listen(url: str, timeout: float) -> None:
    OUT.unlink(missing_ok=True)
    counts = {}
    t0 = time.time()
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"action": "ping"}))
        while time.time() - t0 < timeout:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            msg["_received"] = time.time()
            counts[msg.get("type")] = counts.get(msg.get("type"), 0) + 1
            with open(OUT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(msg) + "\n")
            if msg.get("type") == "training_complete":
                break
    print("message counts:", json.dumps(counts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8001/ws")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    asyncio.run(listen(a.url, a.timeout))
