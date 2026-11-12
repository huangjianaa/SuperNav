"""MuJoCo 4.0 + Unitree Go2 simulation session with 4-way RGB-D, 360° LiDAR, and PixelGoal."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw

from supernav.backends.mujoco.locomotion import Go2LocomotionController
from supernav.backends.mujoco.pixelgoal import Go2PixelGoalMap
from supernav.backends.mujoco.text2mujoco import (
    CAMERA_DIRECTIONS,
    CAMERA_YAW_DEG,
    DEFAULT_CAM_HEIGHT,
    DEFAULT_CAM_WIDTH,
    compile_go2_scene,
    synthesize_scene_spec_offline,
)
from supernav.runtime.config import load_json, write_json
from supernav.runtime.live import LivePublisher


def _load_scene_spec(scene_ref: str) -> dict[str, Any]:
    ref = str(scene_ref or "").strip()
    if ref in ("", "office_pantry", "warehouse_bypass", "auto"):
        return synthesize_scene_spec_offline(preset=ref or "office_pantry")
    p = Path(ref).expanduser()
    if p.is_dir():
        cand = p / "scene_spec.json"
        if cand.is_file():
            return load_json(cand)
    if p.is_file():
        return load_json(p)
    return synthesize_scene_spec_offline(prompt=ref, preset="auto")


class Go2MujocoSession:
    """Single-episode Unitree Go2 navigation session in MuJoCo 4.0."""

    def __init__(
        self,
        *,
        scene_ref: str,
        output_dir: Path,
        task_id: str = "mujoco_go2_task",
        instruction: str = "",
        spawn_override: Mapping[str, Any] | None = None,
        ground_truth_override: Mapping[str, Any] | None = None,
        locomotion_mode: str = "auto",
        cam_width: int = DEFAULT_CAM_WIDTH,
        cam_height: int = DEFAULT_CAM_HEIGHT,
        max_actions: int = 24,
        live_dir: Path | None = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.session_id = str(uuid.uuid4())
        self.task_id = str(task_id or "mujoco_go2_task")
        self.output_dir = Path(output_dir).resolve()
        self.frames_dir = self.output_dir / "frames"
        self.maps_dir = self.output_dir / "maps"
        self.eval_dir = self.output_dir / "evaluator"
        for d in (self.output_dir, self.frames_dir, self.maps_dir, self.eval_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.spec = _load_scene_spec(scene_ref)
        if instruction:
            self.spec.setdefault("task", {})["instruction"] = str(instruction)
        if spawn_override:
            sp = self.spec.setdefault("robot", {}).setdefault("spawn", {})
            if "start_position" in spawn_override:
                sp["position"] = list(spawn_override["start_position"])
            elif "position" in spawn_override:
                sp["position"] = list(spawn_override["position"])
            if "yaw" in spawn_override:
                sp["yaw"] = float(spawn_override["yaw"])

        _, self.model, self.compile_audit = compile_go2_scene(self.spec)
        self.data = mujoco.MjData(self.model)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.cam_width = int(cam_width)
        self.cam_height = int(cam_height)
        self.rgb_renderer = mujoco.Renderer(self.model, height=self.cam_height, width=self.cam_width)
        self.depth_renderer = mujoco.Renderer(self.model, height=self.cam_height, width=self.cam_width)
        self.depth_renderer.enable_depth_rendering()

        self.locomotion = Go2LocomotionController(self.model, self.data, mode=locomotion_mode)
        self.locomotion.settle_standing(0.35)

        w_cfg = (self.spec.get("scene") or {}).get("world") or {}
        fsize = w_cfg.get("floor_size") or [9.0, 7.5]
        self.pixelgoal_map = Go2PixelGoalMap(
            half_extent_x_m=max(4.5, 0.55 * float(fsize[0])),
            half_extent_y_m=max(4.0, 0.55 * float(fsize[1])),
        )
        self.lidar_site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "go2_lidar_site")
        )
        self.cam_ids = {
            d: int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, f"go2_{d}_cam"))
            for d in CAMERA_DIRECTIONS
        }

        # Private ground-truth metadata (never returned in agent tool responses)
        task_meta = dict(self.spec.get("task") or {})
        if ground_truth_override:
            task_meta.update(dict(ground_truth_override))
        self._gt = {
            "target_asset_id": str(task_meta.get("target_asset_id") or ""),
            "goal_position": [float(v) for v in (task_meta.get("goal_position") or [0.0, 0.0, 0.0])],
            "success_radius_m": float(task_meta.get("success_radius_m") or 1.15),
            "geodesic_distance_m": float(task_meta.get("geodesic_distance_m") or 0.0),
        }

        self.max_actions = int(max_actions)
        self.action_count = 0
        self.capture_seq = 0
        self.path_length_m = 0.0
        x0, y0, z0, yaw0, up0 = self.locomotion.base_pose()
        self._prev_xy = (x0, y0)
        self.min_up_z = up0
        self.terminal = False
        self.stop_called = False
        self.close_reason: str | None = None
        self.claims: list[dict[str, Any]] = []
        self.last_nav_result: dict[str, Any] = {"state": "idle"}
        self.rollout_panels: list[Image.Image] = []

        self.live: LivePublisher | None = None
        if live_dir is not None:
            safe_sid = f"mujoco_{self.task_id}_{self.session_id[:8]}"
            safe_sid = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in safe_sid)[:96]
            self.live = LivePublisher(
                Path(live_dir),
                safe_sid,
                scene=str((self.spec.get("scene") or {}).get("name") or "mujoco_go2"),
            )

        # Initial LiDAR scan + rollout frame
        self.pixelgoal_map.update_from_lidar(self.model, self.data, self.lidar_site_id)
        self._record_rollout_panel(label="spawn")
        self._write_status()

    def _pose_dict(self) -> dict[str, float]:
        x, y, z, yaw, up_z = self.locomotion.base_pose()
        self.min_up_z = min(self.min_up_z, up_z)
        return {
            "x": round(x, 4),
            "y": round(y, 4),
            "z": round(z, 4),
            "yaw_rad": round(yaw, 4),
            "heading_deg": round(math.degrees(yaw), 2),
            "up_z": round(up_z, 4),
        }

    def _render_rgb(self, cam_name: str) -> np.ndarray:
        self.rgb_renderer.update_scene(self.data, camera=cam_name)
        return self.rgb_renderer.render().copy()

    def _render_depth(self, cam_name: str) -> np.ndarray:
        self.depth_renderer.update_scene(self.data, camera=cam_name)
        return self.depth_renderer.render().copy()

    def _record_rollout_panel(self, label: str = "") -> None:
        """Compose a side-by-side (Topdown 3D | Go2 Front RGB | 2D LiDAR A* Map) panel."""
        if len(self.rollout_panels) >= 80:
            return
        try:
            top_rgb = self._render_rgb("topdown_cam")
            front_rgb = self._render_rgb("go2_front_cam")
            x, y, _, yaw, _ = self.locomotion.base_pose()
            map_rgb = self.pixelgoal_map.render_topdown_rgb((x, y, yaw), scale=3)

            panel_h = 300
            def _fit(arr: np.ndarray) -> Image.Image:
                im = Image.fromarray(arr).convert("RGB")
                w = int(round(im.width * (panel_h / im.height)))
                return im.resize((w, panel_h), resample=Image.Resampling.BILINEAR)

            im_top = _fit(top_rgb)
            im_front = _fit(front_rgb)
            im_map = _fit(map_rgb)

            total_w = im_top.width + im_front.width + im_map.width + 16
            canvas = Image.new("RGB", (total_w, panel_h + 28), (15, 23, 42))
            draw = ImageDraw.Draw(canvas)
            ox = 4
            for title, tile in (
                ("MuJoCo 4.0 Top-Down (Go2)", im_top),
                (f"Go2 Front Cam ({label})", im_front),
                ("360 LiDAR + PixelGoal A*", im_map),
            ):
                draw.text((ox + 6, 6), title, fill=(226, 232, 240))
                canvas.paste(tile, (ox, 24))
                ox += tile.width + 4
            self.rollout_panels.append(canvas)
        except Exception:
            pass

    def observe(self) -> dict[str, Any]:
        """Capture 4-way surround RGB views and refresh the 360° LiDAR occupancy grid."""
        self.capture_seq += 1
        lidar_stats = self.pixelgoal_map.update_from_lidar(self.model, self.data, self.lidar_site_id)
        pose = self._pose_dict()

        views: list[dict[str, Any]] = []
        for direction in CAMERA_DIRECTIONS:
            rgb = self._render_rgb(f"go2_{direction}_cam")
            fname = f"seq_{self.capture_seq:03d}_{direction}.png"
            fpath = self.frames_dir / fname
            Image.fromarray(rgb).save(fpath)
            heading_deg = (pose["heading_deg"] + CAMERA_YAW_DEG[direction]) % 360.0
            views.append(
                {
                    "direction": direction,
                    "image_ref": f"{self.capture_seq}:{direction}",
                    "path": str(fpath),
                    "width": self.cam_width,
                    "height": self.cam_height,
                    "heading_deg": round(heading_deg, 2),
                }
            )

        map_rgb = self.pixelgoal_map.render_topdown_rgb(
            (pose["x"], pose["y"], pose["yaw_rad"]), scale=3
        )
        map_path = self.maps_dir / f"map_{self.capture_seq:03d}.png"
        Image.fromarray(map_rgb).save(map_path)

        if self.live is not None:
            self.live.panorama(views, self.capture_seq)
            self.live.overlay(str(map_path), {"kind": "pixelgoal_map", "step": self.action_count})
            self.live.sample(
                position=[pose["x"], pose["z"], pose["y"]],
                heading=pose["heading_deg"],
                step=self.action_count,
                path_length=round(self.path_length_m, 4),
                capture=lambda: self._render_rgb("go2_front_cam"),
                force=True,
            )

        return {
            "ok": True,
            "session_id": self.session_id,
            "capture_seq": self.capture_seq,
            "action_count": self.action_count,
            "remaining_actions": max(0, self.max_actions - self.action_count),
            "locomotion_mode": self.locomotion.mode,
            "pose": pose,
            "path_length_m": round(self.path_length_m, 4),
            "lidar": lidar_stats,
            "map_path": str(map_path),
            "views": views,
        }

    def navigate(self, *, point: list[float], direction: str = "front") -> dict[str, Any]:
        """Execute one Go2 PixelGoal hop: RGB-D unproject -> LiDAR A* -> physical quadruped walk."""
        if self.terminal:
            return {"ok": False, "error": "Episode is already closed", "terminal": True}
        if self.action_count >= self.max_actions:
            self.finish(reason="action_budget_exhausted")
            return {"ok": False, "error": "Action budget exhausted", "terminal": True}
        direction = str(direction or "front").lower().strip()
        if direction not in CAMERA_DIRECTIONS:
            return {"ok": False, "error": f"direction must be one of {CAMERA_DIRECTIONS}"}
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return {"ok": False, "error": "point must be [x, y] normalized in [0, 1]"}

        u_norm = float(np.clip(float(point[0]), 0.0, 1.0))
        v_norm = float(np.clip(float(point[1]), 0.0, 1.0))
        self.action_count += 1
        if self.live is not None:
            self.live.event(f"pixelgoal_navigate({direction})", "start")

        # Refresh LiDAR before planning
        self.pixelgoal_map.update_from_lidar(self.model, self.data, self.lidar_site_id)

        cam_name = f"go2_{direction}_cam"
        cam_id = self.cam_ids[direction]
        rgb = self._render_rgb(cam_name)
        depth = self._render_depth(cam_name)

        # Save annotated pixel target overlay for audit & web viewer
        ann = Image.fromarray(rgb).convert("RGB")
        draw = ImageDraw.Draw(ann)
        px = int(round(u_norm * (self.cam_width - 1)))
        py = int(round(v_norm * (self.cam_height - 1)))
        draw.ellipse((px - 10, py - 10, px + 10, py + 10), outline=(250, 204, 21), width=3)
        draw.line((px - 14, py, px + 14, py), fill=(239, 68, 68), width=2)
        draw.line((px, py - 14, px, py + 14), fill=(239, 68, 68), width=2)
        ann_path = self.frames_dir / f"act_{self.action_count:03d}_{direction}_target.png"
        ann.save(ann_path)
        if self.live is not None:
            self.live.overlay(
                str(ann_path),
                {"kind": "pixelgoal_click", "direction": direction, "point": [u_norm, v_norm]},
            )

        x0, y0, _, _, _ = self.locomotion.base_pose()
        plan = self.pixelgoal_map.resolve_and_plan_pixel_goal(
            point_xy=(u_norm, v_norm),
            direction=direction,
            depth_m=depth,
            cam_pos=self.data.cam_xpos[cam_id],
            cam_xmat=self.data.cam_xmat[cam_id],
            fovy_deg=float(self.model.cam_fovy[cam_id]),
            robot_xy=(x0, y0),
        )
        if not plan.get("ok"):
            self.last_nav_result = {
                "state": "blocked_pixel_target",
                "plan": plan,
                "action_count": self.action_count,
            }
            obs = self.observe()
            self._write_status()
            return {
                "ok": False,
                "navigation": self.last_nav_result,
                "observation": obs,
            }

        def _on_walk_step(step_idx: int, pose_tuple: tuple[float, float, float, float, float]) -> None:
            if step_idx % 20 == 0:
                self.pixelgoal_map.update_from_lidar(self.model, self.data, self.lidar_site_id)
                self._record_rollout_panel(label=f"act{self.action_count}")

        walk = self.locomotion.follow_waypoints(
            plan["path_xy"],
            target_yaw=plan["target_yaw"],
            step_callback=_on_walk_step,
        )
        self.path_length_m += float(walk["distance_walked_m"])
        self.min_up_z = min(self.min_up_z, float(walk["min_up_z"]))
        self._prev_xy = (walk["final_pose"]["x"], walk["final_pose"]["y"])

        self.last_nav_result = {
            "state": walk["state"],
            "reason": walk["reason"],
            "mode": plan["mode"],
            "direction": direction,
            "point": [round(u_norm, 4), round(v_norm, 4)],
            "goal_xy": plan["goal_xy"],
            "distance_walked_m": walk["distance_walked_m"],
            "sim_time_s": walk["sim_time_s"],
            "min_up_z": walk["min_up_z"],
        }
        self._record_rollout_panel(label=f"act{self.action_count}_done")
        obs = self.observe()
        if self.live is not None:
            self.live.event(
                f"pixelgoal_navigate({direction})",
                "end",
                ok=(walk["state"] == "succeeded"),
            )
        self._write_status()
        return {
            "ok": walk["state"] in ("succeeded", "partial"),
            "navigation": self.last_nav_result,
            "observation": obs,
        }

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "session_id": self.session_id,
            "terminal": self.terminal,
            "stop_called": self.stop_called,
            "action_count": self.action_count,
            "path_length_m": round(self.path_length_m, 4),
            "pose": self._pose_dict(),
            "last_navigation": self.last_nav_result,
        }

    def stop_movement(self) -> dict[str, Any]:
        self.locomotion.settle_standing(0.15)
        self.last_nav_result = {"state": "canceled", "reason": "agent_stop"}
        self._write_status()
        return self.status()

    def claim_goal(
        self,
        *,
        description: str = "",
        direction: str = "front",
        point: list[float] | None = None,
    ) -> dict[str, Any]:
        """Record agent arrival claim and cleanly terminate the episode."""
        pose = self._pose_dict()
        claim = {
            "timestamp": time.time(),
            "action_count": self.action_count,
            "description": str(description or ""),
            "direction": str(direction or "front"),
            "point": list(point) if isinstance(point, (list, tuple)) else None,
            "robot_pose": pose,
        }
        self.claims.append(claim)
        with (self.eval_dir / "claims.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")

        self.stop_called = True
        self.finish(reason="agent_claimed_goal")
        obs = self.observe()
        return {
            "ok": True,
            "session_id": self.session_id,
            "terminal": True,
            "stop_called": True,
            "claimed": True,
            "observation": obs,
        }

    def _compute_private_evaluation(self) -> dict[str, Any]:
        x, y, _, _, _ = self.locomotion.base_pose()
        gx, gy = float(self._gt["goal_position"][0]), float(self._gt["goal_position"][1])
        dist = math.hypot(x - gx, y - gy)
        radius = float(self._gt["success_radius_m"])
        success = bool(self.stop_called and dist <= radius and self.min_up_z >= 0.42)
        geodesic = max(float(self._gt["geodesic_distance_m"]), 1e-6)
        spl = (geodesic / max(self.path_length_m, geodesic)) if success else 0.0
        return {
            "evaluated": True,
            "success": success,
            "spl": round(float(spl), 4),
            "final_goal_distance_m": round(dist, 4),
            "success_radius_m": round(radius, 4),
            "geodesic_distance_m": round(geodesic, 4),
            "path_length_m": round(self.path_length_m, 4),
            "min_up_z": round(self.min_up_z, 4),
            "stop_called": self.stop_called,
            "claim_count": len(self.claims),
            "locomotion_mode": self.locomotion.mode,
        }

    def _write_status(self) -> dict[str, Any]:
        payload = {
            "state": "closed" if self.terminal else "ready",
            "session_id": self.session_id,
            "task_id": self.task_id,
            "terminal": self.terminal,
            "stop_called": self.stop_called,
            "close_reason": self.close_reason,
            "action_count": self.action_count,
            "capture_seq": self.capture_seq,
            "path_length_m": round(self.path_length_m, 4),
            "min_up_z": round(self.min_up_z, 4),
            "pose": self._pose_dict(),
            "compile_audit": self.compile_audit,
            "evaluation": self._compute_private_evaluation() if self.terminal else None,
        }
        write_json(self.output_dir / "status.json", payload)
        return payload

    def finish(self, reason: str = "evaluator_close") -> dict[str, Any]:
        if not self.terminal:
            self.terminal = True
            self.close_reason = str(reason)
            self._record_rollout_panel(label="final")
            if self.rollout_panels:
                gif_path = self.output_dir / "rollout.gif"
                try:
                    self.rollout_panels[0].save(
                        gif_path,
                        save_all=True,
                        append_images=self.rollout_panels[1:],
                        duration=220,
                        loop=0,
                    )
                except Exception:
                    pass
                try:
                    self.rollout_panels[-1].save(self.output_dir / "final_summary.png")
                except Exception:
                    pass
            if self.live is not None:
                self.live.finish("closed")
        return self._write_status()

    def close(self) -> None:
        try:
            self.rgb_renderer.close()
        except Exception:
            pass
        try:
            self.depth_renderer.close()
        except Exception:
            pass
