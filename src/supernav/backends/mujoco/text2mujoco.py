"""Text2MuJoCo scene specification, validator, LLM generator, and MuJoCo 4.0 Go2 compiler.

Implements a Text2MuJoCo-compatible ``scene_spec.json`` pipeline (SI meters, Z-up,
full metric dimensions, ``orientation_xyzw``) coupled with ``mujoco_menagerie``'s
``unitree_go2`` quadruped in MuJoCo 4.0 (``MjSpec`` + speculative contacts).
"""

from __future__ import annotations

import argparse
from collections import deque
import heapq
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from supernav.runtime.config import write_json
from supernav.runtime.support.config import load_dotenv_from_project

CAMERA_DIRECTIONS = ("front", "right", "back", "left")
CAMERA_YAW_DEG = {"front": 0.0, "right": -90.0, "back": 180.0, "left": 90.0}
DEFAULT_CAM_PITCH_DEG = 4.0
DEFAULT_CAM_FOVY_DEG = 66.0
DEFAULT_CAM_WIDTH = 640
DEFAULT_CAM_HEIGHT = 480
GO2_SPAWN_Z = 0.285
GO2_CLEARANCE_RADIUS_M = 0.36

_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._-]{12,}|/home/[A-Za-z0-9._/-]+)",
    re.IGNORECASE,
)


def redact_prompt_text(text: str) -> str:
    """Strip credential-like tokens or machine-local paths before persisting scene_spec.json."""
    return _SECRET_PATTERN.sub("[REDACTED]", str(text or "")).strip()


def normalize_quat_xyzw_to_wxyz(quat_xyzw: list[float] | tuple[float, ...]) -> list[float]:
    """Normalize an ``[x, y, z, w]`` quaternion and reorder to MuJoCo ``[w, x, y, z]``."""
    if len(quat_xyzw) != 4:
        raise ValueError(f"Expected 4-element orientation_xyzw, got {quat_xyzw!r}")
    x, y, z, w = (float(v) for v in quat_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"Degenerate quaternion: {quat_xyzw!r}")
    return [w / norm, x / norm, y / norm, z / norm]


def yaw_to_quat_wxyz(yaw_rad: float) -> list[float]:
    half = 0.5 * float(yaw_rad)
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def yaw_to_quat_xyzw(yaw_rad: float) -> list[float]:
    half = 0.5 * float(yaw_rad)
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def camera_xyaxes(yaw_deg: float, pitch_down_deg: float = DEFAULT_CAM_PITCH_DEG) -> list[float]:
    """Compute MuJoCo camera ``xyaxes`` (+X right, +Y image-up, viewing along -Z)."""
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_down_deg))
    fx = math.cos(yaw) * math.cos(pitch)
    fy = math.sin(yaw) * math.cos(pitch)
    fz = -math.sin(pitch)
    # right = normalize(fwd x [0, 0, 1]) = [fy, -fx, 0] / hypot(fx, fy)
    horiz = math.hypot(fx, fy) or 1.0
    rx, ry, rz = fy / horiz, -fx / horiz, 0.0
    # cam_up = normalize(right x fwd)
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    unorm = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
    return [rx, ry, rz, ux / unorm, uy / unorm, uz / unorm]


def _asset_xy_aabb(asset: Mapping[str, Any]) -> tuple[float, float, float, float]:
    pose = asset.get("pose") or {}
    pos = pose.get("position") or [0.0, 0.0, 0.0]
    quat_xyzw = pose.get("orientation_xyzw") or [0.0, 0.0, 0.0, 1.0]
    w, x, y, z = normalize_quat_xyzw_to_wxyz(quat_xyzw)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    geom = asset.get("geometry") or {}
    shape = str(geom.get("shape") or "box")
    dims = [float(v) for v in (geom.get("dimensions") or [0.2, 0.2, 0.2])]
    if shape in ("cylinder", "sphere"):
        hx = hy = 0.5 * dims[0]
    else:
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        hx = 0.5 * dims[0] * c + 0.5 * dims[1] * s
        hy = 0.5 * dims[0] * s + 0.5 * dims[1] * c
    cx, cy = float(pos[0]), float(pos[1])
    return cx - hx, cy - hy, cx + hx, cy + hy


def compute_grid_geodesic(
    spec: Mapping[str, Any],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    goal_stop_radius_m: float = 0.65,
    clearance_m: float = 0.26,
    resolution_m: float = 0.08,
) -> tuple[float | None, list[tuple[float, float]]]:
    """Compute 2D collision-free shortest path length on the ground plane."""
    world = (spec.get("scene") or {}).get("world") or {}
    floor_size = world.get("floor_size") or [10.0, 10.0]
    half_w = max(4.0, 0.5 * float(floor_size[0]))
    half_h = max(4.0, 0.5 * float(floor_size[1]))
    nx = int(math.ceil((2.0 * half_w) / resolution_m))
    ny = int(math.ceil((2.0 * half_h) / resolution_m))
    blocked = [[False] * nx for _ in range(ny)]

    target_id = str((spec.get("task") or {}).get("target_asset_id") or "")
    for asset in spec.get("assets") or []:
        if not isinstance(asset, Mapping):
            continue
        geom = asset.get("geometry") or {}
        dims = geom.get("dimensions") or [0.0, 0.0, 0.0]
        pos = (asset.get("pose") or {}).get("position") or [0.0, 0.0, 0.0]
        # Ignore flat decals or overhead objects above Go2 height (0.55 m)
        if float(pos[2]) + 0.5 * float(dims[2]) <= 0.03 or float(pos[2]) - 0.5 * float(dims[2]) >= 0.55:
            continue
        pad = 0.10 if str(asset.get("id")) == target_id else clearance_m
        x0, y0, x1, y1 = _asset_xy_aabb(asset)
        ix0 = max(0, int((x0 - pad + half_w) / resolution_m))
        ix1 = min(nx - 1, int((x1 + pad + half_w) / resolution_m))
        iy0 = max(0, int((y0 - pad + half_h) / resolution_m))
        iy1 = min(ny - 1, int((y1 + pad + half_h) / resolution_m))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                blocked[iy][ix] = True

    def to_cell(xy: tuple[float, float]) -> tuple[int, int]:
        ix = min(nx - 1, max(0, int((xy[0] + half_w) / resolution_m)))
        iy = min(ny - 1, max(0, int((xy[1] + half_h) / resolution_m)))
        return ix, iy

    def to_xy(ix: int, iy: int) -> tuple[float, float]:
        return (-half_w + (ix + 0.5) * resolution_m, -half_h + (iy + 0.5) * resolution_m)

    sx, sy = to_cell(start_xy)
    if blocked[sy][sx]:
        return None, []

    pq: list[tuple[float, float, int, int]] = [(0.0, 0.0, sx, sy)]
    best: dict[tuple[int, int], float] = {(sx, sy): 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    neighbors = (
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
    )
    while pq:
        _, g_cost, cx, cy = heapq.heappop(pq)
        if g_cost > best.get((cx, cy), float("inf")) + 1e-9:
            continue
        wx, wy = to_xy(cx, cy)
        if math.hypot(wx - goal_xy[0], wy - goal_xy[1]) <= goal_stop_radius_m:
            path_cells = [(cx, cy)]
            while path_cells[-1] in parent:
                path_cells.append(parent[path_cells[-1]])
            path_cells.reverse()
            return g_cost * resolution_m, [to_xy(ix, iy) for ix, iy in path_cells]
        for dx, dy, step_cost in neighbors:
            nx_i, ny_i = cx + dx, cy + dy
            if not (0 <= nx_i < nx and 0 <= ny_i < ny) or blocked[ny_i][nx_i]:
                continue
            ng = g_cost + step_cost
            if ng + 1e-9 < best.get((nx_i, ny_i), float("inf")):
                best[(nx_i, ny_i)] = ng
                parent[(nx_i, ny_i)] = (cx, cy)
                nwx, nwy = to_xy(nx_i, ny_i)
                h = math.hypot(nwx - goal_xy[0], nwy - goal_xy[1]) / resolution_m
                heapq.heappush(pq, (ng + h, ng, nx_i, ny_i))
    return None, []


def validate_scene_spec(spec: Mapping[str, Any]) -> list[str]:
    """Deterministically validate a Text2MuJoCo Go2 navigation scene specification."""
    errors: list[str] = []
    if str(spec.get("schema_version") or "") != "1.0":
        errors.append("schema_version must be '1.0'")
    if str(spec.get("backend") or "") != "mujoco":
        errors.append("backend must be 'mujoco'")

    scene = spec.get("scene")
    if not isinstance(scene, Mapping):
        return [*errors, "scene object is required"]
    world = scene.get("world") or {}
    if world.get("units") != "m":
        errors.append("scene.world.units must be 'm'")
    if world.get("up_axis") != "Z":
        errors.append("scene.world.up_axis must be 'Z'")

    robot = spec.get("robot") or {}
    spawn = robot.get("spawn") or {}
    spawn_pos = spawn.get("position") or [0.0, 0.0, GO2_SPAWN_Z]
    if len(spawn_pos) != 3 or not all(math.isfinite(float(v)) for v in spawn_pos):
        errors.append("robot.spawn.position must be 3 finite floats")
        return errors
    sx, sy = float(spawn_pos[0]), float(spawn_pos[1])

    assets = spec.get("assets")
    if not isinstance(assets, list) or not assets:
        errors.append("assets must be a non-empty list")
        return errors

    seen_ids: set[str] = set()
    seen_bodies: set[str] = set()
    for idx, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            errors.append(f"assets[{idx}] must be an object")
            continue
        aid = str(asset.get("id") or "").strip()
        bname = str(asset.get("body_name") or aid).strip()
        if not aid or aid in seen_ids:
            errors.append(f"assets[{idx}] has missing or duplicate id={aid!r}")
        if not bname or bname in seen_bodies or bname.startswith("go2_"):
            errors.append(f"assets[{idx}] has invalid or reserved body_name={bname!r}")
        seen_ids.add(aid)
        seen_bodies.add(bname)

        geom = asset.get("geometry") or {}
        shape = str(geom.get("shape") or "")
        dims = geom.get("dimensions") or []
        if shape not in ("box", "cylinder", "sphere"):
            errors.append(f"asset {aid!r}: unsupported shape={shape!r}")
            continue
        if len(dims) != 3 or any(not math.isfinite(float(d)) or float(d) <= 0 for d in dims):
            errors.append(f"asset {aid!r}: geometry.dimensions must be 3 positive floats")
            continue
        pose = asset.get("pose") or {}
        pos = pose.get("position") or []
        if len(pos) != 3 or any(not math.isfinite(float(v)) for v in pos):
            errors.append(f"asset {aid!r}: pose.position must be 3 finite floats")
            continue
        try:
            normalize_quat_xyzw_to_wxyz(pose.get("orientation_xyzw") or [0, 0, 0, 1])
        except ValueError as exc:
            errors.append(f"asset {aid!r}: {exc}")
            continue

        # Resting height check: bottom must not penetrate ground plane deeper than 0.2 mm
        bottom_z = float(pos[2]) - 0.5 * float(dims[2])
        if bottom_z < -2e-4:
            errors.append(
                f"asset {aid!r}: ground interpenetration bottom_z={bottom_z:.4f}m "
                f"(set z >= {0.5 * float(dims[2]):.4f})"
            )

        # Spawn clearance check
        x0, y0, x1, y1 = _asset_xy_aabb(asset)
        nearest_x = min(max(sx, x0), x1)
        nearest_y = min(max(sy, y0), y1)
        if math.hypot(sx - nearest_x, sy - nearest_y) < GO2_CLEARANCE_RADIUS_M:
            errors.append(
                f"asset {aid!r} overlaps Go2 spawn ({sx:.2f}, {sy:.2f}) within {GO2_CLEARANCE_RADIUS_M}m"
            )

    task = spec.get("task") or {}
    goal_pos = task.get("goal_position")
    if not isinstance(goal_pos, list) or len(goal_pos) < 2:
        errors.append("task.goal_position must be [x, y, z]")
    elif not errors:
        success_radius = float(task.get("success_radius_m") or 0.9)
        geo, _ = compute_grid_geodesic(
            spec,
            (sx, sy),
            (float(goal_pos[0]), float(goal_pos[1])),
            goal_stop_radius_m=max(0.55, success_radius * 0.85),
        )
        if geo is None:
            errors.append(
                f"No navigable Go2 path from spawn ({sx:.2f}, {sy:.2f}) to goal "
                f"({float(goal_pos[0]):.2f}, {float(goal_pos[1]):.2f})"
            )
    return errors


def _wall(aid: str, x: float, y: float, dx: float, dy: float, h: float = 1.85, rgba=None) -> dict[str, Any]:
    return {
        "id": aid,
        "kind": "primitive",
        "body_name": aid,
        "geometry": {
            "shape": "box",
            "dimensions": [dx, dy, h],
            "geom_name": f"{aid}_geom",
            "rgba": list(rgba or [0.80, 0.76, 0.69, 1.0]),
        },
        "pose": {"position": [x, y, 0.5 * h], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "physics": {"dynamic": False, "contype": 1, "conaffinity": 7, "friction": [0.8, 0.01, 0.001]},
        "semantic": "wall",
    }


def _box_asset(
    aid: str, x: float, y: float, dx: float, dy: float, dz: float, rgba: list[float], semantic: str, z_base: float = 0.0
) -> dict[str, Any]:
    return {
        "id": aid,
        "kind": "primitive",
        "body_name": aid,
        "geometry": {
            "shape": "box",
            "dimensions": [dx, dy, dz],
            "geom_name": f"{aid}_geom",
            "rgba": rgba,
        },
        "pose": {"position": [x, y, z_base + 0.5 * dz], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "physics": {"dynamic": False, "contype": 1, "conaffinity": 7, "friction": [0.8, 0.01, 0.001]},
        "semantic": semantic,
    }


def _cyl_asset(
    aid: str, x: float, y: float, diameter: float, height: float, rgba: list[float], semantic: str, z_base: float = 0.0
) -> dict[str, Any]:
    return {
        "id": aid,
        "kind": "primitive",
        "body_name": aid,
        "geometry": {
            "shape": "cylinder",
            "dimensions": [diameter, diameter, height],
            "geom_name": f"{aid}_geom",
            "rgba": rgba,
        },
        "pose": {"position": [x, y, z_base + 0.5 * height], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "physics": {"dynamic": False, "contype": 1, "conaffinity": 7, "friction": [0.8, 0.01, 0.001]},
        "semantic": semantic,
    }


def synthesize_scene_spec_offline(prompt: str = "", preset: str = "office_pantry") -> dict[str, Any]:
    """Generate a validated Text2MuJoCo Go2 navigation scene deterministically."""
    clean_prompt = redact_prompt_text(prompt)
    lower = clean_prompt.lower()
    if preset == "auto":
        if any(k in lower for k in ("仓库", "货架", "warehouse", "shelf", "pallet")):
            preset = "warehouse_bypass"
        else:
            preset = "office_pantry"

    if preset == "warehouse_bypass":
        target_label = "绿色物资箱 (green supply crate)"
        target_rgba = [0.12, 0.78, 0.30, 1.0]
        if any(k in lower for k in ("红", "消防", "red", "fire")):
            target_label = "红色消防物资箱 (red fire-safety crate)"
            target_rgba = [0.88, 0.16, 0.16, 1.0]
        instruction = clean_prompt or (
            f"驾驶 Go2 机器狗从仓库西侧起点出发，绕过中央蓝色货架，找到东侧区域的{target_label}并在其近旁停下。"
        )
        assets = [
            _wall("north_wall", 0.0, 3.1, 7.6, 0.16, 1.8),
            _wall("south_wall", 0.0, -3.1, 7.6, 0.16, 1.8),
            _wall("west_wall", -3.8, 0.0, 0.16, 6.36, 1.8),
            _wall("east_wall", 3.8, 0.0, 0.16, 6.36, 1.8),
            _box_asset("central_rack", -0.1, 0.55, 0.65, 2.7, 1.45, [0.22, 0.42, 0.72, 1.0], "warehouse_shelf"),
            _box_asset("north_crate", -1.6, 2.1, 0.9, 0.65, 0.85, [0.76, 0.58, 0.28, 1.0], "wooden_crate"),
            _cyl_asset("yellow_pillar", 1.25, -1.75, 0.48, 1.2, [0.92, 0.78, 0.15, 1.0], "safety_bollard"),
            _box_asset("target_supply_crate", 2.25, 1.35, 0.65, 0.65, 0.75, target_rgba, "target_object"),
        ]
        spawn = {"position": [-2.55, -1.25, GO2_SPAWN_Z], "yaw": 0.15}
        goal_pos = [2.25, 1.35, 0.375]
        target_id = "target_supply_crate"
        scene_name = "mujoco4_go2_warehouse_bypass"
    else:
        target_label = "红色立柱信标与咖啡台 (red espresso station)"
        target_rgba = [0.90, 0.14, 0.14, 1.0]
        if any(k in lower for k in ("蓝", "blue")):
            target_label = "蓝色饮水台 (blue water dispenser)"
            target_rgba = [0.14, 0.45, 0.92, 1.0]
        elif any(k in lower for k in ("绿", "green")):
            target_label = "绿色回收站 (green recycling station)"
            target_rgba = [0.14, 0.80, 0.32, 1.0]
        instruction = clean_prompt or (
            f"控制 Unitree Go2 从走廊起点出发，绕开前方隔断墙进入茶水间区域，找到{target_label}并靠近确认。"
        )
        assets = [
            _wall("north_wall", 0.0, 2.8, 7.2, 0.16, 1.7),
            _wall("south_wall", 0.0, -2.8, 7.2, 0.16, 1.7),
            _wall("west_wall", -3.6, 0.0, 0.16, 5.76, 1.7),
            _wall("east_wall", 3.6, 0.0, 0.16, 5.76, 1.7),
            # Partition wall anchored to south_wall, leaving a 2.4m wide doorway on the north side
            _box_asset("corridor_partition", -0.25, -1.25, 0.24, 2.95, 1.55, [0.74, 0.69, 0.62, 1.0], "partition_wall"),
            _box_asset("side_cabinet", -1.75, 1.95, 1.2, 0.5, 0.95, [0.46, 0.34, 0.24, 1.0], "office_cabinet"),
            _cyl_asset("blue_bin", 1.45, -1.75, 0.42, 0.65, [0.18, 0.48, 0.88, 1.0], "recycling_bin"),
            _box_asset("pantry_counter", 2.15, 1.35, 0.75, 0.90, 0.65, [0.85, 0.82, 0.76, 1.0], "pantry_counter"),
            _cyl_asset("target_beacon", 2.05, 1.35, 0.38, 0.55, target_rgba, "target_object", z_base=0.65),
        ]
        spawn = {"position": [-2.35, -1.35, GO2_SPAWN_Z], "yaw": 0.35}
        goal_pos = [2.05, 1.35, 0.92]
        target_id = "target_beacon"
        scene_name = "mujoco4_go2_office_pantry"

    spec: dict[str, Any] = {
        "schema_version": "1.0",
        "backend": "mujoco",
        "source_prompt": instruction,
        "assumptions": [
            "使用 SI 米制与 Z 轴向上，地面附带棋盘格纹理以便视觉选点估计距离。",
            "搭载 mujoco_menagerie 中的 Unitree Go2 (12 自由度四足模型) 并启用 MuJoCo 4.0 speculative contacts。",
            "Go2 机身安装前/右/后/左四路环视 RGB-D 相机与 360° 激光雷达，对接 SuperNav PixelGoal 地图导航机制。",
        ],
        "open_questions": [],
        "scene": {
            "name": scene_name,
            "description": instruction,
            "runtime": {
                "mujoco_version": "4.0",
                "speculative_contacts": True,
                "mode": "python",
                "headless": True,
                "gl_backend": "egl",
            },
            "world": {
                "units": "m",
                "up_axis": "Z",
                "gravity": [0.0, 0.0, -9.81],
                "ground": True,
                "floor_size": [9.0, 7.5],
                "seed": 42,
            },
        },
        "robot": {
            "model": "unitree_go2",
            "spawn": spawn,
        },
        "assets": assets,
        "interaction_points": [
            {
                "id": "reach_navigation_target",
                "affordance": "navigate",
                "target": {"type": "body", "name": target_id},
                "pose": {"position": goal_pos, "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
                "success_conditions": ["xy_distance(go2_base, goal_position) <= 1.15 m"],
            }
        ],
        "task": {
            "task_id": f"{scene_name}_task01",
            "slug": scene_name,
            "instruction": instruction,
            "target_asset_id": target_id,
            "goal_position": goal_pos,
            "success_radius_m": 1.15,
        },
    }
    geo, _ = compute_grid_geodesic(
        spec,
        (float(spawn["position"][0]), float(spawn["position"][1])),
        (float(goal_pos[0]), float(goal_pos[1])),
        goal_stop_radius_m=0.95,
    )
    spec["task"]["geodesic_distance_m"] = round(float(geo or 5.0), 4)
    errors = validate_scene_spec(spec)
    if errors:
        raise ValueError(f"Offline scene validation failed: {errors}")
    return spec


_LLM_SYSTEM_PROMPT = """You are a Text2MuJoCo scene architect for Unitree Go2 indoor navigation in MuJoCo 4.0.
Convert the user's natural-language scene & navigation request into a single valid JSON object (no markdown commentary outside JSON) following schema_version "1.0":
{
  "schema_version": "1.0",
  "backend": "mujoco",
  "source_prompt": "<original prompt>",
  "assumptions": ["..."],
  "open_questions": [],
  "scene": {
    "name": "<snake_case_name>",
    "description": "<concise summary>",
    "runtime": {"mujoco_version": "4.0", "speculative_contacts": true, "mode": "python", "headless": true, "gl_backend": "egl"},
    "world": {"units": "m", "up_axis": "Z", "gravity": [0, 0, -9.81], "ground": true, "floor_size": [9.0, 7.5], "seed": 42}
  },
  "robot": {
    "model": "unitree_go2",
    "spawn": {"position": [-2.4, -1.4, 0.285], "yaw": 0.2}
  },
  "assets": [
    {
      "id": "<unique_id>",
      "kind": "primitive",
      "body_name": "<unique_id>",
      "geometry": {"shape": "box|cylinder|sphere", "dimensions": [dx, dy, dz], "geom_name": "<unique_id>_geom", "rgba": [r, g, b, 1.0]},
      "pose": {"position": [x, y, z], "orientation_xyzw": [0, 0, 0, 1]},
      "physics": {"dynamic": false, "contype": 1, "conaffinity": 7, "friction": [0.8, 0.01, 0.001]},
      "semantic": "wall|obstacle|furniture|target_object"
    }
  ],
  "interaction_points": [
    {
      "id": "reach_navigation_target",
      "affordance": "navigate",
      "target": {"type": "body", "name": "<target_asset_id>"},
      "pose": {"position": [gx, gy, gz], "orientation_xyzw": [0, 0, 0, 1]},
      "success_conditions": ["xy_distance(go2_base, goal_position) <= 1.15 m"]
    }
  ],
  "task": {
    "task_id": "<snake_case_name>_task01",
    "slug": "<snake_case_name>",
    "instruction": "<clear navigation instruction for the Go2 agent>",
    "target_asset_id": "<target_asset_id>",
    "goal_position": [gx, gy, gz],
    "success_radius_m": 1.15
  }
}
Rules:
1. Full metric dimensions [dx, dy, dz] in meters; resting floor objects MUST satisfy pose.position[2] >= 0.5 * dz (or seated atop another object).
2. Enclose the room with 4 boundary walls leaving >= 1.2m wide navigable doorways/corridors around interior partitions/shelves.
3. Keep robot.spawn.position at least 0.5m away from all walls/obstacles, and ensure a collision-free floor path of width >= 0.9m exists from spawn to within 0.9m of task.goal_position.
4. Give the target object vivid contrasting RGBA color and height >= 0.5m so Go2's cameras (at z=0.5m) can clearly spot it once around obstacles.
"""


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    candidate = fence.group(1) if fence else cleaned
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON root must be an object")
    return parsed


def generate_scene_spec_from_text(
    prompt: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_repairs: int = 3,
    fallback_offline: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate and validate a Text2MuJoCo Go2 scene spec via an OpenAI-compatible LLM."""
    from openai import OpenAI

    resolved_base_url = (
        base_url
        or os.environ.get("SUPERNAV_LLM_BASE_URL")
        or "http://127.0.0.1:8009/v1"
    )
    resolved_model = (
        model
        or os.environ.get("SUPERNAV_LLM_MODEL")
        or "Qwen3.8-Flash-Next"
    )
    resolved_key = (
        api_key
        or os.environ.get("SUPERNAV_MAAS_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if not resolved_key:
        if fallback_offline:
            spec = synthesize_scene_spec_offline(prompt, preset="auto")
            return spec, {"source": "offline_fallback", "reason": "missing_api_key"}
        raise ValueError("Missing API key (set SUPERNAV_MAAS_API_KEY or pass --api-key)")

    clean_prompt = redact_prompt_text(prompt)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
        {"role": "user", "content": clean_prompt},
    ]
    client = OpenAI(base_url=resolved_base_url, api_key=resolved_key, timeout=60.0)
    last_error = ""
    try:
        for attempt in range(1, max_repairs + 1):
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
            try:
                spec = _extract_json_object(content)
                spec["source_prompt"] = clean_prompt
                errors = validate_scene_spec(spec)
            except Exception as exc:
                errors = [f"JSON parse/schema error: {exc}"]
                spec = {}
            if not errors:
                spawn = spec["robot"]["spawn"]["position"]
                goal = spec["task"]["goal_position"]
                geo, _ = compute_grid_geodesic(
                    spec,
                    (float(spawn[0]), float(spawn[1])),
                    (float(goal[0]), float(goal[1])),
                    goal_stop_radius_m=max(0.55, float(spec["task"].get("success_radius_m", 1.15)) * 0.85),
                )
                spec["task"]["geodesic_distance_m"] = round(float(geo or 0.0), 4)
                return spec, {"source": "llm", "model": resolved_model, "attempts": attempt}
            last_error = "; ".join(errors)
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"Validation failed with {len(errors)} issue(s):\n- "
                    + "\n- ".join(errors)
                    + "\nFix the coordinates/dimensions and return ONLY the corrected JSON."
                ),
            })
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
        if fallback_offline:
            spec = synthesize_scene_spec_offline(prompt, preset="auto")
            return spec, {"source": "offline_fallback", "reason": last_error}
        raise

    if fallback_offline:
        spec = synthesize_scene_spec_offline(prompt, preset="auto")
        return spec, {"source": "offline_fallback", "reason": last_error}
    raise ValueError(f"LLM failed to produce a valid scene_spec after {max_repairs} attempts: {last_error}")


def compile_go2_scene(spec: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    """Compile a validated ``scene_spec.json`` + ``unitree_go2`` into ``(MjSpec, MjModel, audit)``."""
    import mujoco
    import mujoco_menagerie

    errors = validate_scene_spec(spec)
    if errors:
        raise ValueError(f"Invalid scene_spec: {errors}")

    world = mujoco.MjSpec()
    scene_info = spec.get("scene") or {}
    world.modelname = str(scene_info.get("name") or "text2mujoco_go2_scene")

    go2_robot = mujoco_menagerie.get("unitree_go2")
    go2_spec = go2_robot.spec("go2_mjx")

    # Harmonize MuJoCo 4.0 solver options with Go2 MJX and enable speculative contacts
    world.option.timestep = 0.004
    world.option.impratio = 100.0
    world.option.cone = go2_spec.option.cone
    world.option.iterations = 20
    world.option.ls_iterations = 10
    go2_spec.option.iterations = 20
    go2_spec.option.ls_iterations = 10
    go2_spec.option.impratio = 100.0
    if hasattr(mujoco.mjtEnableBit, "mjENBL_SPECULATIVE"):
        world.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SPECULATIVE)
        go2_spec.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SPECULATIVE)

    # Visual quality & balanced lighting (no white clipping) for VLM floor/obstacle perception
    world.visual.headlight.diffuse = [0.28, 0.28, 0.30]
    world.visual.headlight.ambient = [0.22, 0.22, 0.25]
    world.worldbody.add_light(
        name="sun_key",
        pos=[1.5, -1.5, 6.0],
        dir=[-0.2, 0.2, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.52, 0.52, 0.50],
        ambient=[0.14, 0.14, 0.16],
    )
    world.worldbody.add_light(
        name="fill_light",
        pos=[-2.0, 2.0, 5.0],
        dir=[0.2, -0.2, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.22, 0.24, 0.28],
        castshadow=0,
    )

    # Soft indoor skybox gradient so views above partitions never clip to pitch black
    world.add_texture(
        name="sky_grad",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.86, 0.89, 0.93],
        rgb2=[0.68, 0.73, 0.80],
        width=256,
        height=256,
    )

    # Checkerboard ground texture + material so VLM sees floor perspective clearly
    tex = world.add_texture(
        name="floor_grid_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.52, 0.58, 0.66],
        rgb2=[0.36, 0.42, 0.50],
        width=512,
        height=512,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        markrgb=[0.28, 0.33, 0.40],
    )
    mat = world.add_material(
        name="floor_grid_mat",
        texrepeat=[4.0, 4.0],
        texuniform=True,
        reflectance=0.04,
    )
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name

    w_cfg = scene_info.get("world") or {}
    fsize = w_cfg.get("floor_size") or [9.0, 7.5]
    world.worldbody.add_geom(
        name="ground_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.5 * float(fsize[0]), 0.5 * float(fsize[1]), 0.1],
        material="floor_grid_mat",
        friction=[0.85, 0.02, 0.01],
        contype=1,
        conaffinity=7,
    )

    # Top-down overview camera
    span = max(float(fsize[0]), float(fsize[1]))
    world.worldbody.add_camera(
        name="topdown_cam",
        pos=[0.0, 0.0, 1.15 * span],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        fovy=52.0,
    )

    # Compile static & dynamic scene assets
    shape_map = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    }
    for asset in spec.get("assets") or []:
        bname = str(asset.get("body_name") or asset["id"])
        pose = asset.get("pose") or {}
        pos = [float(v) for v in pose["position"]]
        quat_wxyz = normalize_quat_xyzw_to_wxyz(pose.get("orientation_xyzw") or [0, 0, 0, 1])
        body = world.worldbody.add_body(name=bname, pos=pos, quat=quat_wxyz)

        geom_cfg = asset["geometry"]
        shape = str(geom_cfg["shape"])
        dims = [float(d) for d in geom_cfg["dimensions"]]
        if shape == "box":
            mj_size = [0.5 * dims[0], 0.5 * dims[1], 0.5 * dims[2]]
        elif shape == "cylinder":
            mj_size = [0.5 * dims[0], 0.5 * dims[2], 0.0]
        else:
            mj_size = [0.5 * dims[0], 0.0, 0.0]

        phys = asset.get("physics") or {}
        is_dyn = bool(phys.get("dynamic", False))
        if is_dyn:
            body.add_freejoint(name=f"{bname}_freejoint")
        geom_kwargs: dict[str, Any] = {
            "name": str(geom_cfg.get("geom_name") or f"{bname}_geom"),
            "type": shape_map[shape],
            "size": mj_size,
            "rgba": [float(c) for c in (geom_cfg.get("rgba") or [0.7, 0.7, 0.7, 1.0])],
            "contype": int(phys.get("contype", 1)),
            "conaffinity": int(phys.get("conaffinity", 7)),
            "friction": [float(f) for f in (phys.get("friction") or [0.8, 0.01, 0.001])],
            "group": 0,
        }
        if is_dyn and phys.get("mass_kg") is not None:
            geom_kwargs["mass"] = float(phys["mass_kg"])
        body.add_geom(**geom_kwargs)

    # Mount 4-way surround cameras + 360 LiDAR site onto Go2's base body
    go2_base = go2_spec.body("base")
    cam_offsets = {
        "front": [0.25, 0.0, 0.24],
        "right": [0.0, -0.15, 0.24],
        "back": [-0.25, 0.0, 0.24],
        "left": [0.0, 0.15, 0.24],
    }
    for direction in CAMERA_DIRECTIONS:
        go2_base.add_camera(
            name=f"{direction}_cam",
            pos=cam_offsets[direction],
            xyaxes=camera_xyaxes(CAMERA_YAW_DEG[direction], DEFAULT_CAM_PITCH_DEG),
            fovy=DEFAULT_CAM_FOVY_DEG,
        )
    go2_base.add_site(
        name="lidar_site",
        pos=[0.08, 0.0, 0.22],
        size=[0.01, 0.01, 0.01],
        group=4,
    )

    spawn_cfg = (spec.get("robot") or {}).get("spawn") or {}
    spawn_pos = [float(v) for v in (spawn_cfg.get("position") or [0.0, 0.0, GO2_SPAWN_Z])]
    spawn_yaw = float(spawn_cfg.get("yaw", 0.0))
    spawn_quat = yaw_to_quat_wxyz(spawn_yaw)

    frame = world.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
    world.attach(go2_spec, prefix="go2_", frame=frame)

    if len(world.keys) > 0:
        qpos0 = list(world.keys[0].qpos)
        if len(qpos0) >= 7:
            qpos0[0:3] = [spawn_pos[0], spawn_pos[1], max(GO2_SPAWN_Z, float(qpos0[2]), spawn_pos[2])]
            qpos0[3:7] = spawn_quat
            world.keys[0].qpos = qpos0

    model = world.compile()
    if model.nkey > 0:
        default_home_z = float(model.key_qpos[0, 2])
        model.key_qpos[0, 0:3] = [
            spawn_pos[0],
            spawn_pos[1],
            max(GO2_SPAWN_Z, default_home_z, spawn_pos[2]),
        ]
        model.key_qpos[0, 3:7] = spawn_quat

    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Static contact penetration audit (Text2MuJoCo standard)
    deepest_penetration = 0.0
    deepest_pair: tuple[str, str] | None = None
    for ci in range(data.ncon):
        con = data.contact[ci]
        if con.dist < deepest_penetration:
            deepest_penetration = float(con.dist)
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom_{con.geom1}"
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom_{con.geom2}"
            deepest_pair = (g1, g2)

    audit = {
        "mujoco_version": mujoco.__version__,
        "speculative_contacts_enabled": bool(
            hasattr(mujoco.mjtEnableBit, "mjENBL_SPECULATIVE")
            and (model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SPECULATIVE))
        ),
        "nbody": int(model.nbody),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "ncam": int(model.ncam),
        "initial_contact_count": int(data.ncon),
        "deepest_initial_contact_m": round(deepest_penetration, 6),
        "deepest_pair": list(deepest_pair) if deepest_pair else None,
        "spawn_pose": {
            "position": [round(float(v), 4) for v in data.qpos[:3]],
            "yaw": round(spawn_yaw, 4),
        },
    }
    if deepest_penetration < -5e-3:
        raise ValueError(f"Initial scene compilation failed contact audit: {audit}")
    return world, model, audit


def export_scene_package(
    spec: Mapping[str, Any],
    output_dir: Path,
    *,
    generation_meta: Mapping[str, Any] | None = None,
    render_preview: bool = True,
) -> dict[str, Any]:
    """Export ``scene_spec.json``, ``model.xml``, ``tasks.json``, ``audit.json``, and previews."""
    import mujoco_menagerie

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_path = output_dir / "scene_spec.json"
    write_json(spec_path, dict(spec))

    world_spec, model, audit = compile_go2_scene(spec)
    if generation_meta:
        audit["generation"] = dict(generation_meta)

    assets_dst = output_dir / "assets"
    assets_src = mujoco_menagerie.get("unitree_go2").path() / "assets"
    if assets_src.is_dir() and not assets_dst.exists():
        assets_dst.symlink_to(assets_src, target_is_directory=True)

    xml_text = world_spec.to_xml()
    xml_text = xml_text.replace('<compiler angle="radian"/>', '<compiler angle="radian" meshdir="assets"/>', 1)
    if model.nkey > 0:
        qpos_str = " ".join(f"{float(v):.6g}" for v in model.key_qpos[0])
        xml_text = re.sub(r'(<key\s+name="go2_home"\s+qpos=")[^"]*(")', rf"\g<1>{qpos_str}\2", xml_text, count=1)
    mjcf_path = output_dir / "model.xml"
    mjcf_path.write_text(xml_text, encoding="utf-8")
    write_json(output_dir / "audit.json", audit)

    task_cfg = spec["task"]
    spawn_cfg = spec["robot"]["spawn"]
    tasks_manifest = {
        "instructions": [
            {
                "task_id": str(task_cfg["task_id"]),
                "slug": str(task_cfg["slug"]),
                "text": str(task_cfg["instruction"]),
                "scene": str(spec_path),
                "spawn": {
                    "start_position": list(spawn_cfg["position"]),
                    "yaw": float(spawn_cfg.get("yaw", 0.0)),
                },
                "ground_truth": {
                    "target_asset_id": str(task_cfg["target_asset_id"]),
                    "goal_position": list(task_cfg["goal_position"]),
                    "success_radius_m": float(task_cfg.get("success_radius_m", 1.15)),
                    "geodesic_distance_m": float(task_cfg.get("geodesic_distance_m", 0.0)),
                },
            }
        ]
    }
    tasks_path = output_dir / "tasks.json"
    write_json(tasks_path, tasks_manifest)

    previews: dict[str, str] = {}
    if render_preview and os.environ.get("MUJOCO_GL", "egl") != "disable":
        try:
            os.environ.setdefault("MUJOCO_GL", "egl")
            import mujoco
            from PIL import Image

            data = mujoco.MjData(model)
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
            renderer = mujoco.Renderer(model, height=DEFAULT_CAM_HEIGHT, width=DEFAULT_CAM_WIDTH)
            preview_dir = output_dir / "previews"
            preview_dir.mkdir(exist_ok=True)
            for cam_name, key in (("topdown_cam", "topdown"), ("go2_front_cam", "go2_front")):
                renderer.update_scene(data, camera=cam_name)
                img = Image.fromarray(renderer.render())
                img_path = preview_dir / f"{key}.png"
                img.save(img_path)
                previews[key] = str(img_path)
            renderer.close()
        except Exception as exc:
            audit["preview_warning"] = f"{type(exc).__name__}: {exc}"
            write_json(output_dir / "audit.json", audit)

    return {
        "scene_spec": str(spec_path),
        "mjcf": str(mjcf_path),
        "tasks_manifest": str(tasks_path),
        "audit": audit,
        "previews": previews,
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv_from_project()
    parser = argparse.ArgumentParser(
        description="Generate a MuJoCo 4.0 + Unitree Go2 navigation scene package from text."
    )
    parser.add_argument("--prompt", type=str, default="", help="Natural-language scene and task description.")
    parser.add_argument(
        "--preset",
        choices=("auto", "office_pantry", "warehouse_bypass"),
        default="auto",
        help="Scene preset when --offline or --fallback-offline is used.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated scene package.")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible endpoint base URL.")
    parser.add_argument("--model", type=str, default=None, help="LLM model name (e.g. Qwen3.8-Flash-Next).")
    parser.add_argument("--api-key-env", type=str, default="SUPERNAV_MAAS_API_KEY", help="Env var holding API key.")
    parser.add_argument("--offline", action="store_true", help="Synthesize deterministically without LLM network call.")
    parser.add_argument("--fallback-offline", action="store_true", help="Fall back to offline synthesis if LLM is unreachable.")
    parser.add_argument("--no-preview", action="store_true", help="Skip initial offscreen PNG preview rendering.")
    args = parser.parse_args(argv)

    if args.offline:
        spec = synthesize_scene_spec_offline(args.prompt, preset=args.preset)
        meta: dict[str, Any] = {"source": "offline", "preset": args.preset}
    else:
        api_key = os.environ.get(args.api_key_env, "").strip() or None
        spec, meta = generate_scene_spec_from_text(
            args.prompt or "生成一个带隔断墙的办公室茶水间，让 Go2 绕过隔断找到红色咖啡台。",
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            fallback_offline=args.fallback_offline,
        )
    summary = export_scene_package(
        spec,
        args.output_dir,
        generation_meta=meta,
        render_preview=not args.no_preview,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
