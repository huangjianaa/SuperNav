"""End-to-end tests for Text2MuJoCo + MuJoCo 4.0 + Unitree Go2 + OpenAIMCPAgent."""

from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_menagerie")

from supernav.backends.mujoco.locomotion import Go2LocomotionController
from supernav.backends.mujoco.pixelgoal import Go2PixelGoalMap
from supernav.backends.mujoco.text2mujoco import (
    compile_go2_scene,
    export_scene_package,
    redact_prompt_text,
    synthesize_scene_spec_offline,
    validate_scene_spec,
)

ROOT = Path(__file__).resolve().parents[2]


def test_text2mujoco_spec_validation_and_secret_redaction() -> None:
    cleaned = redact_prompt_text("Use sk-maas-1234567890abcdef and /home/alice/secret in office")
    assert "sk-maas-" not in cleaned
    assert "/home/alice" not in cleaned

    spec = synthesize_scene_spec_offline("构建带隔断的茶水间场景", preset="office_pantry")
    assert validate_scene_spec(spec) == []
    assert spec["task"]["geodesic_distance_m"] > 3.5

    bad_penetration = copy.deepcopy(spec)
    bad_penetration["assets"][-1]["pose"]["position"][2] = -0.2
    errs = validate_scene_spec(bad_penetration)
    assert any("ground interpenetration" in e for e in errs)

    bad_spawn = copy.deepcopy(spec)
    sx, sy, _ = spec["robot"]["spawn"]["position"]
    bad_spawn["assets"][-1]["pose"]["position"] = [sx, sy, 0.5]
    errs_spawn = validate_scene_spec(bad_spawn)
    assert any("overlaps Go2 spawn" in e for e in errs_spawn)


@pytest.mark.parametrize("preset", ["office_pantry", "warehouse_bypass"])
def test_mujoco4_go2_compile_and_standalone_xml_export(preset: str, tmp_path: Path) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    spec = synthesize_scene_spec_offline(preset=preset)
    out = export_scene_package(spec, tmp_path / preset, render_preview=True)
    audit = out["audit"]
    assert audit["mujoco_version"].startswith("4.")
    assert audit["speculative_contacts_enabled"] is True
    assert audit["nu"] == 12
    assert audit["ncam"] == 6
    assert audit["deepest_initial_contact_m"] >= -5e-3

    # Verify standalone XML reload works with linked assets/ directory and preserves Go2 spawn
    reloaded = mujoco.MjModel.from_xml_path(out["mjcf"])
    assert reloaded.nu == 12
    assert abs(float(reloaded.key_qpos[0, 0]) - float(spec["robot"]["spawn"]["position"][0])) < 1e-4
    assert Path(out["tasks_manifest"]).is_file()


def test_go2_rl_locomotion_and_lidar_pixelgoal_astar() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    spec = synthesize_scene_spec_offline(preset="office_pantry")
    _, model, _ = compile_go2_scene(spec)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    loco = Go2LocomotionController(model, data, mode="auto")
    assert loco.mode in ("rl_torque", "cpg_torque")
    loco.settle_standing(0.25)

    pg = Go2PixelGoalMap()
    lidar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "go2_lidar_site")
    stats = pg.update_from_lidar(model, data, lidar_id)
    assert stats["explored_area_m2"] > 5.0
    assert stats["occupied_cells"] > 40

    renderer = mujoco.Renderer(model, height=240, width=320)
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="go2_left_cam")
    depth = renderer.render().copy()
    renderer.close()

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "go2_left_cam")
    x0, y0, _, _, _ = loco.base_pose()
    plan = pg.resolve_and_plan_pixel_goal(
        point_xy=(0.52, 0.74),
        direction="left",
        depth_m=depth,
        cam_pos=data.cam_xpos[cam_id],
        cam_xmat=data.cam_xmat[cam_id],
        fovy_deg=float(model.cam_fovy[cam_id]),
        robot_xy=(x0, y0),
    )
    assert plan["ok"] is True
    assert len(plan["path_xy"]) >= 2

    walk = loco.follow_waypoints(plan["path_xy"], target_yaw=plan["target_yaw"], max_duration_s=8.0)
    assert walk["distance_walked_m"] > 0.5
    assert walk["min_up_z"] > 0.80


def test_openai_mcp_agent_with_mock_qwen_server_end_to_end(tmp_path: Path) -> None:
    """Verify exact OpenAI/Qwen3.8-Flash-Next wire protocol (4 base64 images + tool calls + <think>)."""
    captured_requests: list[dict[str, Any]] = []

    class _MockQwenHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            captured_requests.append(
                {
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "body": body,
                }
            )
            turn = len(captured_requests)
            if turn == 1:
                tool_call = {
                    "id": "call_qwen_1",
                    "type": "function",
                    "function": {"name": "observe_surround", "arguments": "{}"},
                }
                content = "<think>先调用四路环视相机观察周围环境。</think>正在观察四周。"
            elif turn == 2:
                tool_call = {
                    "id": "call_qwen_2",
                    "type": "function",
                    "function": {
                        "name": "pixelgoal_navigate",
                        "arguments": json.dumps({"direction": "front", "point": [0.5, 0.68]}),
                    },
                }
                content = "<think>前方视野清晰，向目标点移动。</think>向目标靠近。"
            else:
                tool_call = {
                    "id": "call_qwen_3",
                    "type": "function",
                    "function": {
                        "name": "claim_goal",
                        "arguments": json.dumps(
                            {"description": "红色目标信标就在正前方", "direction": "front", "point": [0.5, 0.55]}
                        ),
                    },
                }
                content = "<think>已到达目标半径内，提交确认。</think>已到达目标位置。"

            resp = {
                "id": f"chatcmpl-mock-{turn}",
                "object": "chat.completion",
                "created": 1760000000,
                "model": body.get("model", "Qwen3.8-Flash-Next"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": [tool_call],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
            payload = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), _MockQwenHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        scene_dir = tmp_path / "scene_pkg"
        spec = synthesize_scene_spec_offline(preset="office_pantry")
        # Spawn Go2 facing the target counter from 1.7 m away so 1 hop reaches arrival radius
        spec["robot"]["spawn"] = {"position": [0.75, 1.35, 0.285], "yaw": 0.0}
        export_scene_package(spec, scene_dir, render_preview=False)

        cfg_path = tmp_path / "mock_qwen_exp.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "extends": [str(ROOT / "configs/experiments/mujoco-go2-pixelgoal.json")],
                    "environment": {"backend": "mujoco", "locomotion": "kinematic", "max_actions": 8},
                    "agents": {
                        "openai_mcp": {
                            "timeout_s": 120,
                            "max_turns": 6,
                            "provider": {
                                "id": "mock-qwen",
                                "type": "openai",
                                "base_url": f"http://127.0.0.1:{port}/v1",
                                "env_key": "SUPERNAV_MAAS_API_KEY",
                            },
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        env = dict(os.environ)
        env["MUJOCO_GL"] = "egl"
        env["SUPERNAV_MAAS_API_KEY"] = "sk-test-qwen-mock"
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["no_proxy"] = "127.0.0.1,localhost"

        out_dir = tmp_path / "runs"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "supernav",
                "run-one",
                "--config",
                str(cfg_path),
                "--arm",
                "default",
                "--slug",
                "mock-qwen-ep",
                "--task-id",
                "mock-qwen-ep",
                "--scene",
                str(scene_dir),
                "--instruction",
                "向前靠近红色目标台并确认到达。",
                "--output-dir",
                str(out_dir),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        assert len(captured_requests) == 3
        assert captured_requests[0]["auth"] == "Bearer sk-test-qwen-mock"
        assert captured_requests[0]["body"]["model"] == "Qwen3.8-Flash-Next"

        # Turn 2 request must carry the 4 base64 JPEG surround views from observe_surround
        turn2_msgs = captured_requests[1]["body"]["messages"]
        img_parts = [
            part
            for msg in turn2_msgs
            if isinstance(msg.get("content"), list)
            for part in msg["content"]
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        assert len(img_parts) == 4
        assert img_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

        metrics_files = list(out_dir.rglob("metrics.json"))
        assert len(metrics_files) == 1
        metrics = json.loads(metrics_files[0].read_text(encoding="utf-8"))
        assert metrics["success"] is True
        assert metrics["stop_called"] is True
        assert metrics["spl"] > 0.5
        assert metrics["tool_calls"] == {
            "observe_surround": 1,
            "pixelgoal_navigate": 1,
            "claim_goal": 1,
        }
    finally:
        server.shutdown()
        server.server_close()
