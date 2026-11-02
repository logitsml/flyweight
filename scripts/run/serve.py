"""FlyWeight live: the fly inside Qwen in your browser, while it runs.

    python scripts/run/serve.py [--port 8770] [--fly-only] [--memory 4] [--adapt]

Open http://localhost:8770 (on a remote GPU: ssh -L 8770:localhost:8770 host). Talk to Qwen (your
message goes into the fly's next reports, and Qwen's own answer drives the fly), or drive the fly's
descending neurons yourself, stimulate its senses, dust its antennae or reset it. Everything the
viewer shows is computed live by the one-stack model (flyweight/live.py); --fly-only runs the fly's
layers alone (no Qwen; about 4 GB of GPU memory).
"""

import argparse
import asyncio
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

from aiohttp import WSMsgType, web

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VIZ = ROOT / "viz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--fly-only", action="store_true")
    ap.add_argument("--memory", type=int, default=0, help="Qwen reads each report after its last N reports and answers")
    ap.add_argument("--adapt", action="store_true", help="the fly runs the adaptation variant (spike-frequency adaptation)")
    ap.add_argument("--history", type=int, default=1, help="recent windows a visitor who joins starts from (1: the newest; the rest is live)")
    a = ap.parse_args()
    from flyweight.live import LiveFly
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    from flyweight import fly_layers
    live = LiveFly(fly_only=a.fly_only, memory=a.memory, adapt=fly_layers.ADAPT if a.adapt else None)
    live.header["live"]["started"] = time.time()  # a reconnecting viewer reloads when the server restarted
    print(f"model ready in {time.time() - t0:.0f} s; open http://localhost:{a.port}", flush=True)
    clients, history = set(), deque(maxlen=a.history)
    loop = asyncio.new_event_loop()

    def broadcast(text):
        history.append(text)
        for ws in list(clients):
            if not ws.closed:
                asyncio.ensure_future(ws.send_str(text), loop=loop)

    def run():  # the model, in its own thread: one window at a time while anyone watches
        while True:
            if not clients:
                time.sleep(0.2)
                continue
            try:
                msg = live.window()
            except Exception as e:  # report and keep serving
                print("window failed:", repr(e), flush=True)
                time.sleep(1.0)
                continue
            if msg is None:
                time.sleep(0.1)
                continue
            loop.call_soon_threadsafe(broadcast, json.dumps(msg))

    async def index(request):
        html = (VIZ / "index.html").read_text()
        return web.Response(text=html.replace("<script>", '<script>window.FLYWEIGHT_LIVE = "ws";</script>\n<script>', 1),
                            content_type="text/html", headers={"Cache-Control": "no-store"})

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=1 << 20)
        await ws.prepare(request)
        await ws.send_str(json.dumps(dict(type="hello", **live.header, votes_b64=votes_b64)))
        for text in list(history):
            await ws.send_str(text)
        clients.add(ws)
        try:
            async for m in ws:
                if m.type == WSMsgType.TEXT:
                    try:
                        live.controls.put(json.loads(m.data))
                    except (ValueError, TypeError):
                        pass
        finally:
            clients.discard(ws)
        return ws

    votes_b64 = None
    if not a.fly_only:
        import base64
        import gzip
        from flyweight import viewdata
        payload, meta = viewdata.qwen_votes()
        live.header["votes"] = meta
        votes_b64 = base64.b64encode(gzip.compress(payload, 6)).decode()
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/", VIZ)
    threading.Thread(target=run, daemon=True).start()
    asyncio.set_event_loop(loop)
    web.run_app(app, host=a.host, port=a.port, loop=loop, print=None)


if __name__ == "__main__":
    main()
