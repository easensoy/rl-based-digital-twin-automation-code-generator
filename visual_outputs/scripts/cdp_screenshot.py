"""
Headless Edge/Chrome screenshot via the Chrome DevTools Protocol, with a real
wall-clock wait (CSS animations and timers finish) and capture of console
messages and uncaught exceptions.

A throw-away browser profile is used, so the user's browser profile is never
touched.

Usage:
    .venv\\Scripts\\python.exe visual_outputs\\scripts\\cdp_screenshot.py URL OUT.png
        [--wait 6] [--width 1600] [--height 1000] [--eval "js"] [--log OUT.txt]
"""
import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request

import websockets

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


async def run(url, out, wait, width, height, js, log_path, port):
    exe = next(b for b in BROWSERS if __import__("os").path.exists(b))
    profile = tempfile.mkdtemp(prefix="cdp-profile-")
    proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={port}",
                             f"--user-data-dir={profile}", "--no-first-run",
                             "--no-default-browser-check", "--hide-scrollbars",
                             f"--window-size={width},{height}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        targets = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as r:
                    targets = [t for t in json.load(r) if t.get("type") == "page"]
                if targets:
                    break
            except OSError:
                pass
            time.sleep(0.2)
        ws_url = targets[0]["webSocketDebuggerUrl"]
        logs = []
        async with websockets.connect(ws_url, max_size=None) as ws:
            msg_id = 0
            pending = {}

            async def send(method, params=None):
                nonlocal msg_id
                msg_id += 1
                fut = asyncio.get_running_loop().create_future()
                pending[msg_id] = fut
                await ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
                return await fut

            async def reader():
                async for raw in ws:
                    m = json.loads(raw)
                    if "id" in m and m["id"] in pending:
                        pending.pop(m["id"]).set_result(m.get("result", m.get("error")))
                    elif m.get("method") == "Runtime.consoleAPICalled":
                        p = m["params"]
                        text = " ".join(str(a.get("value", a.get("description", ""))) for a in p["args"])
                        logs.append(f"[console.{p['type']}] {text}")
                    elif m.get("method") == "Runtime.exceptionThrown":
                        d = m["params"]["exceptionDetails"]
                        desc = d.get("exception", {}).get("description", d.get("text", ""))
                        logs.append(f"[uncaught] {desc.splitlines()[0]}  @ {d.get('url', '')}:{d.get('lineNumber', 0) + 1}")
                    elif m.get("method") == "Log.entryAdded":
                        e = m["params"]["entry"]
                        logs.append(f"[{e['source']}.{e['level']}] {e['text']} {e.get('url', '')}")

            rtask = asyncio.create_task(reader())
            await send("Runtime.enable")
            await send("Log.enable")
            await send("Page.enable")
            await send("Emulation.setDeviceMetricsOverride",
                       {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
            await send("Page.navigate", {"url": url})
            await asyncio.sleep(wait)
            if js:
                res = await send("Runtime.evaluate", {"expression": js, "awaitPromise": True,
                                                      "returnByValue": True})
                logs.append(f"[diagnostic eval] {json.dumps(res)[:500]}")
                await asyncio.sleep(2)
            shot = await send("Page.captureScreenshot", {"format": "png"})
            with open(out, "wb") as fh:
                fh.write(base64.b64decode(shot["data"]))
            try:
                await send("Browser.close")
            except Exception:
                pass
            rtask.cancel()
        if log_path:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(f"URL: {url}\nwait: {wait}s, viewport {width}x{height}\n\n")
                fh.write("\n".join(logs) + "\n")
        print(f"saved {out} ({len(logs)} console/log entries)")
    finally:
        proc.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--wait", type=float, default=6)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--eval", dest="js", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--port", type=int, default=9333)
    a = ap.parse_args()
    asyncio.run(run(a.url, a.out, a.wait, a.width, a.height, a.js, a.log, a.port))
