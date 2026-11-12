"""MuJoCo 4.0 + Text2MuJoCo + Unitree Go2 experiment backend (no simulator SDK imports)."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.request

from supernav.backends.base import BridgeStartupError, EpisodeTask
from supernav.backends.mujoco import DEFAULT_BRIDGE_PORT
from supernav.backends.mujoco.config import build_mcp_spec
from supernav.paths import python_paths, resolve_asset_path
from supernav.runtime.config import instruction_rows, load_json, resolve_path, write_json
from supernav.runtime.mcp import validate_mcp_config
from supernav.runtime.streams import write_canonical_events

DEFAULT_GO2_AGENT_INSTRUCTIONS = """# Unitree Go2 PixelGoal Navigation Agent (MuJoCo 4.0)

You control a physical 12-DoF Unitree Go2 quadruped robot equipped with:
- 4 surround RGB-D cameras returned in fixed order: `front`, `right`, `back`, `left`
- A 360° 3D LiDAR building a persistent 2D obstacle-inflated occupancy map + A* planner

## Workflow Rules
1. Start by calling `observe_surround` once to inspect all 4 directions (`front`, `right`, `back`, `left`).
2. Move by calling `pixelgoal_navigate(point=[x, y], direction=...)` where `x, y` are normalized image coordinates in `[0.0, 1.0]` (`[0, 0]` = top-left, `[1, 1]` = bottom-right):
   - Prefer clicking visible open floor (`y` around `0.62`–`0.85`) leading around walls/partitions or toward doorways.
   - Once the target object is clearly visible, click the floor directly in front of it (or the object's lower body; the planner automatically stands off safely in front of vertical surfaces).
3. Each `pixelgoal_navigate` returns updated 4-view images after Go2 finishes walking, so you do NOT need redundant `observe_surround` calls after every move.
4. Only when Go2 is close (`<= 1.0 m`) and clearly facing/beside the target object, call `claim_goal(description=..., direction=..., point=[x, y])` to finish the episode. Never guess or claim from afar.
"""


class _RunningMujocoEpisode:
    def __init__(self, port: int, path: Path) -> None:
        self.url = f"http://127.0.0.1:{port}"
        self.path = path
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, route: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.url}{route}",
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if body is None and route == "/healthz" else "POST",
        )
        with self.opener.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def finish(self, reason: str = "agent_returned_without_claim") -> dict[str, Any]:
        health = self.request("/healthz")
        if not health.get("terminal"):
            self.request("/evaluator-close", {"reason": reason})
        return load_json(self.path / "status.json")


def _normalize_env(config: dict[str, Any], root: Path) -> dict[str, Any]:
    env = dict(config.get("environment") or {})
    env["backend"] = "mujoco"
    env["python"] = os.path.expandvars(
        str(env.get("python") or os.environ.get("SUPERNAV_MUJOCO_PYTHON") or sys.executable)
    )
    env["locomotion"] = str(env.get("locomotion") or "auto")
    if env["locomotion"] not in ("auto", "rl_torque", "cpg_torque", "kinematic"):
        raise ValueError(f"Unsupported environment.locomotion={env['locomotion']!r}")
    env["max_actions"] = int(env.get("max_actions") or 20)
    if env.get("scene_package"):
        resolved = resolve_path(str(env["scene_package"]), base=root)
        if resolved is not None:
            env["scene_package"] = str(resolved)
    return env


class MujocoBackend:
    """SuperNav EnvironmentBackend for MuJoCo 4.0 + Text2MuJoCo + Unitree Go2."""

    name = "mujoco"
    default_port = 0

    def tasks(self, config: dict[str, Any], instructions: str | None, root: Path) -> list[dict[str, Any]]:
        env = _normalize_env(config, root)
        manifest = instructions or config.get("instructions_file")
        if manifest:
            manifest_path = resolve_path(str(manifest), base=root)
            if manifest_path is None or not manifest_path.is_file():
                raise ValueError(f"Task manifest not found: {manifest}")
            return instruction_rows(manifest_path)

        scene_pkg = env.get("scene_package")
        if scene_pkg:
            pkg_tasks = Path(scene_pkg) / "tasks.json"
            if pkg_tasks.is_file():
                return instruction_rows(pkg_tasks)

        raise ValueError(
            "MuJoCo Go2 sweep requires --instructions <tasks.json> (generate one with `supernav text2mujoco`)."
        )

    def prepare_task(self, config: dict[str, Any], args: Any, arm: dict[str, Any], root: Path) -> EpisodeTask:
        validate_mcp_config(config.get("mcp") or {})
        env = _normalize_env(config, root)
        config["environment"] = env
        config["benchmark_profile"] = str(config.get("benchmark_profile") or "mujoco_go2_pixelgoal")

        raw_scene = getattr(args, "scene", None) or env.get("scene_package") or env.get("preset") or "office_pantry"
        scene_str = str(raw_scene)
        maybe_path = resolve_path(scene_str, base=root)
        if maybe_path is not None and maybe_path.exists():
            scene_str = str(maybe_path)

        spawn: dict[str, Any] = {}
        raw_spawn = getattr(args, "spawn", None)
        if isinstance(raw_spawn, dict):
            spawn = dict(raw_spawn)
        elif isinstance(raw_spawn, str) and raw_spawn.strip():
            spawn = json.loads(raw_spawn)

        gt: dict[str, Any] | None = None
        raw_gt = getattr(args, "ground_truth", None)
        if isinstance(raw_gt, dict):
            gt = dict(raw_gt)
        elif isinstance(raw_gt, str) and raw_gt.strip():
            gt = json.loads(raw_gt)

        # Hydrate spawn/GT from scene_spec.json when pointing at a generated Text2MuJoCo package
        spec_candidate = Path(scene_str) / "scene_spec.json" if Path(scene_str).is_dir() else Path(scene_str)
        if spec_candidate.is_file() and spec_candidate.suffix == ".json":
            try:
                spec_data = load_json(spec_candidate)
                if not spawn and isinstance((spec_data.get("robot") or {}).get("spawn"), dict):
                    sp = spec_data["robot"]["spawn"]
                    spawn = {
                        "start_position": list(sp.get("position") or [0.0, 0.0, 0.285]),
                        "yaw": float(sp.get("yaw", 0.0)),
                    }
                if gt is None and isinstance(spec_data.get("task"), dict):
                    t_cfg = spec_data["task"]
                    if "goal_position" in t_cfg:
                        gt = {
                            "target_asset_id": t_cfg.get("target_asset_id"),
                            "goal_position": t_cfg.get("goal_position"),
                            "success_radius_m": t_cfg.get("success_radius_m", 1.15),
                            "geodesic_distance_m": t_cfg.get("geodesic_distance_m", 0.0),
                        }
            except Exception:
                pass

        task_id = str(getattr(args, "task_id", None) or getattr(args, "slug", None) or "mujoco_go2_ep")
        return EpisodeTask(
            task_id=task_id,
            instruction=str(args.instruction),
            scene=scene_str,
            spawn=spawn,
            ground_truth=gt,
        )

    def configure_agent(
        self,
        config: dict[str, Any],
        arm: dict[str, Any],
        task: EpisodeTask,
        root: Path,
        run_dir: Path,
    ) -> None:
        del task
        env = config["environment"]
        bridge = dict(config.get("bridge") or {})
        bridge.setdefault("host", "127.0.0.1")
        if bridge["host"] != "127.0.0.1":
            raise ValueError("MuJoCo Go2 backend uses a loopback 127.0.0.1 bridge")
        if not bridge.get("port"):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                bridge["port"] = sock.getsockname()[1]
        config["bridge"] = bridge

        attempt = 1
        while (run_dir / f"simulator-{attempt}").exists():
            attempt += 1
        config["_simulator_output"] = str(run_dir / f"simulator-{attempt}")

        spec = build_mcp_spec(workspace_root=root, config=config, arm_cfg=arm)
        config["_mcp_launch"] = {
            "name": spec.name,
            "command": spec.command,
            "args": list(spec.args),
            "environment": dict(spec.environment),
            "transport": spec.transport,
            "http_args": list(spec.http_args) if spec.http_args else None,
        }

        instructions_text = DEFAULT_GO2_AGENT_INSTRUCTIONS
        skill_name = str(arm.get("skill_name") or "").strip()
        if skill_name and str(arm.get("skill_mode") or "").lower() == "native":
            skill_file = resolve_asset_path(f"skills/{skill_name}/SKILL.md", base=root)
            if skill_file is not None and skill_file.is_file():
                instructions_text = instructions_text + "\n\n" + skill_file.read_text(encoding="utf-8")

        config.setdefault("agent_instructions", instructions_text)
        config.setdefault("agent_policy", "")
        write_json(
            run_dir / "backend.json",
            {
                "backend": self.name,
                "environment": env,
                "bridge": bridge,
                "simulator_output": config["_simulator_output"],
                "mcp": config["_mcp_launch"],
            },
        )

    def prompt(
        self,
        *,
        task: EpisodeTask,
        arm_name: str,
        arm_cfg: dict[str, Any],
        workspace_root: Path,
        prompts_dir: Path,
    ) -> str:
        del arm_name, workspace_root, prompts_dir
        skill_hint = ""
        if str(arm_cfg.get("skill_mode") or "").lower() == "native" and arm_cfg.get("skill_name"):
            skill_hint = f"\nActive skill: `{arm_cfg['skill_name']}`.\n"
        return (
            f"Navigation Task for Unitree Go2:\n{task.instruction.strip()}\n"
            f"{skill_hint}\n"
            "Inspect the 4 surround camera views (`observe_surround`), navigate safely around obstacles "
            "using `pixelgoal_navigate(point=[x, y], direction=...)`, and call `claim_goal(...)` "
            "once Go2 arrives right next to the target."
        )

    @contextmanager
    def episode(
        self,
        *,
        root: Path,
        config: dict[str, Any],
        task: EpisodeTask,
        run_dir: Path,
        dry_run: bool,
        live_context: dict[str, Any],
    ):
        if dry_run:
            yield None
            return

        env = config["environment"]
        port = int(config["bridge"]["port"] or DEFAULT_BRIDGE_PORT)
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise BridgeStartupError(f"Bridge port {port} is already occupied")

        sim_out = Path(config["_simulator_output"])
        command = [
            env["python"],
            "-m",
            "supernav.backends.mujoco.http_server",
            "--port",
            str(port),
            "--scene",
            task.scene,
            "--output",
            str(sim_out),
            "--task-id",
            str(task.task_id or "mujoco_go2_ep"),
            "--instruction",
            task.instruction,
            "--locomotion",
            str(env.get("locomotion") or "auto"),
            "--max-actions",
            str(env.get("max_actions") or 20),
        ]
        if task.spawn:
            command.extend(["--spawn-json", json.dumps(task.spawn)])
        if task.ground_truth:
            command.extend(["--ground-truth-json", json.dumps(task.ground_truth)])

        proc_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc_env.setdefault("MUJOCO_GL", "egl")
        if python_paths():
            proc_env["PYTHONPATH"] = os.pathsep.join(python_paths())
        if env.get("live_dir"):
            proc_env["SUPERNAV_LIVE_DIR"] = str(resolve_path(str(env["live_dir"]), base=root))
        proc_env["SUPERNAV_LIVE_CONTEXT"] = json.dumps(live_context, ensure_ascii=False)

        log_path = run_dir / "simulator.log"
        session = _RunningMujocoEpisode(port, sim_out)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=proc_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            ready = False
            finish_on_exit = True
            try:
                deadline = time.monotonic() + float(config["bridge"].get("startup_timeout_s", 60))
                while True:
                    if process.poll() is not None:
                        raise BridgeStartupError(f"MuJoCo Go2 bridge exited early; see {log_path}")
                    try:
                        health = session.request("/healthz")
                        if health.get("state") == "ready":
                            ready = True
                            break
                    except (OSError, ValueError):
                        pass
                    if time.monotonic() >= deadline:
                        raise BridgeStartupError(f"MuJoCo Go2 bridge startup timed out; see {log_path}")
                    time.sleep(0.15)
                try:
                    yield session
                except subprocess.TimeoutExpired:
                    session.finish("wall_timeout")
                    raise
                except BaseException:
                    finish_on_exit = False
                    raise
            finally:
                if ready and finish_on_exit and process.poll() is None:
                    try:
                        session.finish()
                    except (OSError, ValueError):
                        pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                write_json(
                    run_dir / "simulator-process.json",
                    {
                        "command": command,
                        "returncode": process.returncode,
                        "ready": ready,
                        "owned": True,
                    },
                )

    def collect(
        self,
        *,
        config: dict[str, Any],
        root: Path,
        run_dir: Path,
        task: EpisodeTask,
        args: Any,
        result: Any,
        events: list[dict[str, Any]],
        session: _RunningMujocoEpisode,
    ) -> dict[str, Any]:
        del config, root, task
        status = (
            session.finish("wall_timeout")
            if getattr(result, "timed_out", False)
            else session.finish()
        )
        write_canonical_events(run_dir / "canonical.jsonl", events)
        counts = Counter(e.get("name", "unknown") for e in events if e.get("type") == "tool_call")
        evaluation = status.get("evaluation") or {}
        claims_path = session.path / "evaluator/claims.jsonl"
        claim_rows = (
            [json.loads(line) for line in claims_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if claims_path.is_file()
            else []
        )

        return {
            "arm": args.arm,
            "slug": args.slug,
            "rep": args.rep,
            "session_id": status.get("session_id"),
            "success": bool(evaluation.get("success", False)) and getattr(result, "returncode", 1) == 0,
            "spl": evaluation.get("spl", 0.0),
            "final_goal_distance_m": evaluation.get("final_goal_distance_m"),
            "geodesic_distance_m": evaluation.get("geodesic_distance_m"),
            "path_length_m": status.get("path_length_m", 0.0),
            "min_up_z": status.get("min_up_z"),
            "locomotion_mode": evaluation.get("locomotion_mode"),
            "success_semantics": "stop_called_and_goal_within_radius_and_robot_upright",
            "action_count": status.get("action_count", 0),
            "stop_called": status.get("stop_called", False),
            "claim_count": len(claim_rows),
            "agent_terminal_claim": claim_rows[-1] if claim_rows else None,
            "tool_calls": dict(counts),
            "tool_calls_total": sum(counts.values()),
            "simulator_status": status,
            "simulator_evidence": str(session.path),
            "rollout_gif": str(session.path / "rollout.gif") if (session.path / "rollout.gif").is_file() else None,
            "observation_views": ["front", "right", "back", "left"],
        }

    def replay(self, run_dir: Path, config: dict[str, Any], root: Path) -> dict[str, Any]:
        del run_dir, config, root
        return {"replay_manifest_status": "mujoco_rollout_gif"}
