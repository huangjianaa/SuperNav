"""Loopback HTTP bridge server for a single MuJoCo 4.0 + Unitree Go2 episode."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import signal
import threading
from typing import Any

from supernav.backends.mujoco.simulator import Go2MujocoSession


class _BridgeServer(HTTPServer):
    def __init__(self, server_address: tuple[str, int], session: Go2MujocoSession) -> None:
        super().__init__(server_address, _BridgeHandler)
        self.session = session
        self.lock = threading.Lock()


class _BridgeHandler(BaseHTTPRequestHandler):
    server: _BridgeServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:  # noqa: N802
        with self.server.lock:
            sess = self.server.session
            if self.path == "/healthz":
                self._send_json(
                    200,
                    {
                        "state": "ready",
                        "session_id": sess.session_id,
                        "terminal": sess.terminal,
                        "locomotion_mode": sess.locomotion.mode,
                    },
                )
                return
            if self.path == "/status":
                self._send_json(200, sess.status())
                return
        self._send_json(404, {"ok": False, "error": f"Unknown route: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_body()
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": f"Invalid JSON: {exc}"})
            return

        with self.server.lock:
            sess = self.server.session
            try:
                if self.path == "/observe":
                    self._send_json(200, sess.observe())
                    return
                if self.path == "/navigate":
                    res = sess.navigate(
                        point=body.get("point") or [0.5, 0.7],
                        direction=str(body.get("direction") or "front"),
                    )
                    self._send_json(200, res)
                    return
                if self.path == "/status":
                    self._send_json(200, sess.status())
                    return
                if self.path == "/stop":
                    self._send_json(200, sess.stop_movement())
                    return
                if self.path == "/claim":
                    res = sess.claim_goal(
                        description=str(body.get("description") or ""),
                        direction=str(body.get("direction") or "front"),
                        point=body.get("point"),
                    )
                    self._send_json(200, res)
                    return
                if self.path == "/evaluator-close":
                    status = sess.finish(reason=str(body.get("reason") or "evaluator_close"))
                    self._send_json(200, status)
                    return
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                return

        self._send_json(404, {"ok": False, "error": f"Unknown route: {self.path}"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a MuJoCo 4.0 + Unitree Go2 episode over loopback HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--scene", required=True, help="Path to scene_spec.json, scene package dir, or preset name.")
    parser.add_argument("--output", type=Path, required=True, help="Simulator evidence directory.")
    parser.add_argument("--task-id", default="mujoco_go2_task")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--spawn-json", default="")
    parser.add_argument("--ground-truth-json", default="")
    parser.add_argument("--locomotion", default="auto", choices=("auto", "rl_torque", "cpg_torque", "kinematic"))
    parser.add_argument("--max-actions", type=int, default=24)
    args = parser.parse_args(argv)

    spawn = json.loads(args.spawn_json) if args.spawn_json.strip() else None
    gt = json.loads(args.ground_truth_json) if args.ground_truth_json.strip() else None
    live_env = os.environ.get("SUPERNAV_LIVE_DIR", "").strip()
    live_dir = Path(live_env) if live_env else None

    session = Go2MujocoSession(
        scene_ref=args.scene,
        output_dir=args.output,
        task_id=args.task_id,
        instruction=args.instruction,
        spawn_override=spawn,
        ground_truth_override=gt,
        locomotion_mode=args.locomotion,
        max_actions=args.max_actions,
        live_dir=live_dir,
    )
    httpd = _BridgeServer((args.host, args.port), session)

    def _shutdown_handler(signum: int, _frame: Any) -> None:
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    try:
        httpd.serve_forever(poll_interval=0.2)
    finally:
        try:
            if not session.terminal:
                session.finish(reason="operator_interrupt")
        finally:
            session.close()
            httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
