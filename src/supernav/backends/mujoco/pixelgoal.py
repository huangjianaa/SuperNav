"""360° LiDAR occupancy mapping + RGB-D PixelGoal resolver & A* planner for Go2 in MuJoCo 4.0.

Mirrors the real-robot Go2 PixelGoal pipeline (`robots/go2/.../pixel_goal_map.py`):
1. Accumulates a 2D occupancy grid (`unknown=-1`, `free=0`, `occupied=100`) from
   `mj_multiRay` 3D LiDAR scans mounted on `go2_lidar_site`.
2. Unprojects normalized `[x, y] in [0, 1]^2` pixel coordinates from any of the
   4 surround cameras (`front`, `right`, `back`, `left`) using metric depth.
3. Distinguishes floor hits (`z <= 0.14m`), vertical obstacle hits (stands off `0.38m`
   in front of the wall/object), and far/open rays (caps single exploration hop to
   `1.65m` and up to `0.55m` into unknown frontier).
4. Plans an 8-connected obstacle-inflated A* path with clearance penalty so Go2
   comfortably rounds doorways and corners.
"""

from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Any

import numpy as np

CELL_UNKNOWN = -1
CELL_FREE = 0
CELL_OCCUPIED = 100


class Go2PixelGoalMap:
    """Persistent 2D LiDAR occupancy map and RGB-D PixelGoal A* planner."""

    def __init__(
        self,
        *,
        half_extent_x_m: float = 5.0,
        half_extent_y_m: float = 4.5,
        resolution_m: float = 0.05,
        robot_radius_m: float = 0.28,
        lidar_max_range_m: float = 6.5,
        max_hop_m: float = 1.65,
        unknown_extension_m: float = 0.55,
        vertical_standoff_m: float = 0.55,
    ) -> None:
        self.half_w = float(half_extent_x_m)
        self.half_h = float(half_extent_y_m)
        self.res = float(resolution_m)
        self.nx = int(math.ceil((2.0 * self.half_w) / self.res))
        self.ny = int(math.ceil((2.0 * self.half_h) / self.res))
        self.robot_radius_m = float(robot_radius_m)
        self.lidar_max_range_m = float(lidar_max_range_m)
        self.max_hop_m = float(max_hop_m)
        self.unknown_extension_m = float(unknown_extension_m)
        self.vertical_standoff_m = float(vertical_standoff_m)

        self.grid = np.full((self.ny, self.nx), CELL_UNKNOWN, dtype=np.int8)
        self.last_goal_xy: tuple[float, float] | None = None
        self.last_path_xy: list[tuple[float, float]] = []
        self.trajectory_xy: list[tuple[float, float]] = []

        # Precompute 3-ring 360° ray directions in sensor local frame (180 az x 3 el = 540 rays)
        az = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
        dirs = []
        for pitch_deg in (-5.0, 0.0, 5.0):
            cp = math.cos(math.radians(pitch_deg))
            sp = math.sin(math.radians(pitch_deg))
            for a in az:
                dirs.append([cp * math.cos(a), cp * math.sin(a), sp])
        self._local_rays = np.asarray(dirs, dtype=np.float64)
        self._geomgroup = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        ix = int((x + self.half_w) / self.res)
        iy = int((y + self.half_h) / self.res)
        return max(0, min(self.nx - 1, ix)), max(0, min(self.ny - 1, iy))

    def cell_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        return (-self.half_w + (ix + 0.5) * self.res, -self.half_h + (iy + 0.5) * self.res)

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def mark_robot_footprint_free(self, x: float, y: float, radius_m: float = 0.30) -> None:
        cx, cy = self.world_to_cell(x, y)
        rad_cells = int(math.ceil(radius_m / self.res))
        for dy in range(-rad_cells, rad_cells + 1):
            for dx in range(-rad_cells, rad_cells + 1):
                if dx * dx + dy * dy <= rad_cells * rad_cells:
                    nx_i, ny_i = cx + dx, cy + dy
                    if self.in_bounds(nx_i, ny_i) and self.grid[ny_i, nx_i] != CELL_OCCUPIED:
                        self.grid[ny_i, nx_i] = CELL_FREE

    def update_from_lidar(self, model: Any, data: Any, lidar_site_id: int) -> dict[str, float]:
        """Cast 3D LiDAR rays via ``mj_multiRay`` and update free/occupied cells."""
        import mujoco

        origin = np.asarray(data.site_xpos[lidar_site_id], dtype=np.float64).copy()
        rot = np.asarray(data.site_xmat[lidar_site_id], dtype=np.float64).reshape(3, 3)
        world_rays = (self._local_rays @ rot.T).reshape(-1).copy()

        nray = self._local_rays.shape[0]
        geomid = np.full(nray, -1, dtype=np.int32)
        dist = np.full(nray, -1.0, dtype=np.float64)
        mujoco.mj_multiRay(
            model,
            data,
            origin,
            world_rays,
            self._geomgroup,
            True,
            -1,
            geomid,
            dist,
            None,
            nray,
            self.lidar_max_range_m,
        )

        self.mark_robot_footprint_free(float(origin[0]), float(origin[1]))
        if not self.trajectory_xy or math.hypot(
            origin[0] - self.trajectory_xy[-1][0], origin[1] - self.trajectory_xy[-1][1]
        ) >= 0.06:
            self.trajectory_xy.append((round(float(origin[0]), 4), round(float(origin[1]), 4)))

        rays_3d = world_rays.reshape(nray, 3)
        ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
        step_m = 0.85 * self.res

        for i in range(nray):
            d = float(dist[i])
            rx, ry, rz = float(rays_3d[i, 0]), float(rays_3d[i, 1]), float(rays_3d[i, 2])
            horiz = math.hypot(rx, ry)
            if horiz < 1e-6:
                continue

            hit_obstacle = False
            if d > 0.0:
                hz = oz + d * rz
                # Hits with 0.08 m < z < 1.35 m are genuine obstacles; z <= 0.08 m is ground plane
                if 0.08 < hz < 1.35:
                    hit_obstacle = True
                    trace_len = max(0.0, d - 0.5 * self.res)
                else:
                    trace_len = min(d, self.lidar_max_range_m)
            else:
                trace_len = self.lidar_max_range_m

            n_samples = int(trace_len / step_m)
            for s in range(1, n_samples + 1):
                 sd = s * step_m
                 ix, iy = self.world_to_cell(ox + rx * sd, oy + ry * sd)
                 if self.grid[iy, ix] == CELL_UNKNOWN:
                     self.grid[iy, ix] = CELL_FREE

            if hit_obstacle:
                hx, hy = ox + rx * d, oy + ry * d
                ix, iy = self.world_to_cell(hx, hy)
                self.grid[iy, ix] = CELL_OCCUPIED

        free_cells = int(np.sum(self.grid == CELL_FREE))
        occ_cells = int(np.sum(self.grid == CELL_OCCUPIED))
        return {
            "explored_area_m2": round(free_cells * (self.res ** 2), 2),
            "occupied_cells": occ_cells,
        }

    def _inflated_obstacle_mask(self, radius_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(inflated_blocked_bool, clearance_penalty_grid)``."""
        rad = self.robot_radius_m if radius_m is None else float(radius_m)
        rad_cells = max(1, int(math.ceil(rad / self.res)))
        soft_cells = rad_cells + 3

        occ = self.grid == CELL_OCCUPIED
        dist_cells = np.full((self.ny, self.nx), 999, dtype=np.int32)
        q: deque[tuple[int, int]] = deque()
        ys, xs = np.nonzero(occ)
        for iy, ix in zip(ys.tolist(), xs.tolist()):
            dist_cells[iy, ix] = 0
            q.append((ix, iy))

        while q:
            cx, cy = q.popleft()
            cd = int(dist_cells[cy, cx])
            if cd >= soft_cells:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx_i, ny_i = cx + dx, cy + dy
                if 0 <= nx_i < self.nx and 0 <= ny_i < self.ny and dist_cells[ny_i, nx_i] > cd + 1:
                    dist_cells[ny_i, nx_i] = cd + 1
                    q.append((nx_i, ny_i))

        blocked = dist_cells <= rad_cells
        penalty = np.where(
            dist_cells <= soft_cells,
            (soft_cells + 1 - dist_cells).astype(np.float32) * 0.45,
            0.0,
        )
        return blocked, penalty

    def _nearest_navigable_cell(
        self,
        target_cell: tuple[int, int],
        start_cell: tuple[int, int],
        blocked: np.ndarray,
        *,
        allow_unknown: bool = False,
        max_search_m: float = 0.90,
    ) -> tuple[int, int] | None:
        tx, ty = target_cell
        max_r = max(1, int(math.ceil(max_search_m / self.res)))
        best_cell: tuple[int, int] | None = None
        best_score = float("inf")
        sx, sy = start_cell

        for dy in range(-max_r, max_r + 1):
            for dx in range(-max_r, max_r + 1):
                d2 = dx * dx + dy * dy
                if d2 > max_r * max_r:
                    continue
                ix, iy = tx + dx, ty + dy
                if not self.in_bounds(ix, iy) or blocked[iy, ix]:
                    continue
                cell_val = int(self.grid[iy, ix])
                if cell_val == CELL_OCCUPIED:
                    continue
                if not allow_unknown and cell_val == CELL_UNKNOWN:
                    continue
                # Tie-break slightly toward robot start so standoff lands on the robot's side of a wall!
                score = math.sqrt(d2) + 0.08 * math.hypot(ix - sx, iy - sy)
                if score < best_score:
                    best_score = score
                    best_cell = (ix, iy)
        return best_cell

    def plan_astar(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        allow_unknown: bool = True,
    ) -> list[tuple[float, float]]:
        """Plan an 8-connected obstacle-inflated A* path on the occupancy map."""
        blocked, penalty = self._inflated_obstacle_mask()
        sx, sy = self.world_to_cell(*start_xy)
        # Never block the robot's immediate footprint
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if self.in_bounds(sx + dx, sy + dy):
                    blocked[sy + dy, sx + dx] = False

        gx, gy = self.world_to_cell(*goal_xy)
        snapped = self._nearest_navigable_cell(
            (gx, gy), (sx, sy), blocked, allow_unknown=allow_unknown, max_search_m=0.95
        )
        if snapped is None:
            return []
        gx, gy = snapped

        pq: list[tuple[float, float, int, int]] = [(0.0, 0.0, sx, sy)]
        best: dict[tuple[int, int], float] = {(sx, sy): 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        neighbors = (
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        )

        while pq:
            _, g_cost, cx, cy = heapq.heappop(pq)
            if g_cost > best.get((cx, cy), float("inf")) + 1e-9:
                continue
            if (cx, cy) == (gx, gy):
                cells = [(cx, cy)]
                while cells[-1] in parent:
                    cells.append(parent[cells[-1]])
                cells.reverse()
                raw_pts = [self.cell_to_world(ix, iy) for ix, iy in cells]
                # Subsample every ~0.20 m while preserving start and exact snapped goal
                if len(raw_pts) <= 3:
                    return raw_pts
                thinned = [raw_pts[0]]
                for pt in raw_pts[1:-1]:
                    if math.hypot(pt[0] - thinned[-1][0], pt[1] - thinned[-1][1]) >= 0.20:
                        thinned.append(pt)
                thinned.append(raw_pts[-1])
                return thinned

            for dx, dy, base_step in neighbors:
                nx_i, ny_i = cx + dx, cy + dy
                if not self.in_bounds(nx_i, ny_i) or blocked[ny_i, nx_i]:
                    continue
                cell_state = int(self.grid[ny_i, nx_i])
                if cell_state == CELL_OCCUPIED:
                    continue
                if not allow_unknown and cell_state == CELL_UNKNOWN:
                    continue
                unk_cost = 0.65 if cell_state == CELL_UNKNOWN else 0.0
                ng = g_cost + base_step + float(penalty[ny_i, nx_i]) + unk_cost
                if ng + 1e-9 < best.get((nx_i, ny_i), float("inf")):
                    best[(nx_i, ny_i)] = ng
                    parent[(nx_i, ny_i)] = (cx, cy)
                    h = math.hypot(gx - nx_i, gy - ny_i)
                    heapq.heappush(pq, (ng + h, ng, nx_i, ny_i))
        return []

    def unproject_pixel(
        self,
        point_xy: tuple[float, float],
        depth_m: np.ndarray,
        cam_pos: np.ndarray,
        cam_xmat: np.ndarray,
        fovy_deg: float,
    ) -> dict[str, Any]:
        """Unproject normalized ``(u_norm, v_norm)`` in ``[0, 1]^2`` to 3D world coordinates."""
        h, w = depth_m.shape
        u_norm = float(np.clip(point_xy[0], 0.0, 1.0))
        v_norm = float(np.clip(point_xy[1], 0.0, 1.0))
        px = int(round(u_norm * (w - 1)))
        py = int(round(v_norm * (h - 1)))

        # Small 5x5 median patch around pixel for noise-free depth
        x0, x1 = max(0, px - 2), min(w, px + 3)
        y0, y1 = max(0, py - 2), min(h, py + 3)
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 25.0)]

        fy = 0.5 * h / math.tan(0.5 * math.radians(float(fovy_deg)))
        fx = fy
        cx = 0.5 * (w - 1)
        cy = 0.5 * (h - 1)

        dir_cam = np.array([(px - cx) / fx, -(py - cy) / fy, -1.0], dtype=np.float64)
        ray_world = cam_xmat.reshape(3, 3) @ dir_cam
        ray_norm = np.linalg.norm(ray_world) or 1.0
        ray_unit = ray_world / ray_norm

        if valid.size == 0:
            return {
                "valid_depth": False,
                "z_cam_m": None,
                "hit_world": None,
                "ray_unit": ray_unit,
                "pixel_rc": (py, px),
            }

        z_cam = float(np.median(valid))
        pt_cam = dir_cam * z_cam  # because dir_cam[2] == -1.0 and z_cam is optical Z depth
        hit_world = np.asarray(cam_pos, dtype=np.float64) + cam_xmat.reshape(3, 3) @ pt_cam
        return {
            "valid_depth": True,
            "z_cam_m": round(z_cam, 4),
            "hit_world": hit_world,
            "ray_unit": ray_unit,
            "pixel_rc": (py, px),
        }

    def resolve_and_plan_pixel_goal(
        self,
        *,
        point_xy: tuple[float, float],
        direction: str,
        depth_m: np.ndarray,
        cam_pos: np.ndarray,
        cam_xmat: np.ndarray,
        fovy_deg: float,
        robot_xy: tuple[float, float],
    ) -> dict[str, Any]:
        """Resolve a normalized camera pixel into a safe 2D goal + A* waypoint trajectory."""
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        unproj = self.unproject_pixel(point_xy, depth_m, cam_pos, cam_xmat, fovy_deg)
        ray_unit = unproj["ray_unit"]
        horiz_dir = np.array([ray_unit[0], ray_unit[1]], dtype=np.float64)
        horiz_norm = float(np.linalg.norm(horiz_dir))
        if horiz_norm < 1e-6:
            return {
                "ok": False,
                "mode": "blocked_pixel_target",
                "reason": "degenerate_vertical_ray",
                "path_xy": [],
            }
        horiz_unit = horiz_dir / horiz_norm

        mode = "resolved"
        if unproj["valid_depth"] and unproj["hit_world"] is not None:
            hit = unproj["hit_world"]
            hx, hy, hz = float(hit[0]), float(hit[1]), float(hit[2])
            raw_dist = math.hypot(hx - rx, hy - ry)
            if hz > 0.16:
                # Vertical wall / furniture / target beacon: stand off cleanly in front of surface
                standoff_d = max(0.18, min(self.max_hop_m, raw_dist - self.vertical_standoff_m))
                tx = rx + horiz_unit[0] * standoff_d
                ty = ry + horiz_unit[1] * standoff_d
                mode = "vertical_surface_standoff"
            elif raw_dist > self.max_hop_m:
                tx = rx + horiz_unit[0] * self.max_hop_m
                ty = ry + horiz_unit[1] * self.max_hop_m
                mode = "exploration"
            else:
                tx, ty = hx, hy
                mode = "resolved"
        else:
            tx = rx + horiz_unit[0] * self.max_hop_m
            ty = ry + horiz_unit[1] * self.max_hop_m
            mode = "exploration"

        # Clamp exploration into unknown cells so Go2 doesn't blindly charge through unseen regions
        dist_desired = math.hypot(tx - rx, ty - ry)
        step_check = 0.5 * self.res
        unknown_run = 0.0
        capped_tx, capped_ty = tx, ty
        for s in range(1, int(dist_desired / step_check) + 1):
            sd = s * step_check
            px_w = rx + horiz_unit[0] * sd
            py_w = ry + horiz_unit[1] * sd
            ix, iy = self.world_to_cell(px_w, py_w)
            cell_v = int(self.grid[iy, ix])
            if cell_v == CELL_OCCUPIED:
                safe_d = max(0.15, sd - self.vertical_standoff_m)
                capped_tx = rx + horiz_unit[0] * safe_d
                capped_ty = ry + horiz_unit[1] * safe_d
                break
            if cell_v == CELL_UNKNOWN:
                unknown_run += step_check
                if unknown_run >= self.unknown_extension_m:
                    capped_tx, capped_ty = px_w, py_w
                    mode = "exploration"
                    break

        path = self.plan_astar((rx, ry), (capped_tx, capped_ty), allow_unknown=True)
        if not path:
            self.last_goal_xy = None
            self.last_path_xy = []
            return {
                "ok": False,
                "mode": "blocked_pixel_target",
                "direction": direction,
                "point": [round(float(point_xy[0]), 4), round(float(point_xy[1]), 4)],
                "reason": "no_collision_free_path",
                "path_xy": [],
            }

        final_goal = path[-1]
        target_yaw = math.atan2(horiz_unit[1], horiz_unit[0])
        self.last_goal_xy = final_goal
        self.last_path_xy = path
        return {
            "ok": True,
            "mode": mode,
            "direction": direction,
            "point": [round(float(point_xy[0]), 4), round(float(point_xy[1]), 4)],
            "goal_xy": [round(final_goal[0], 4), round(final_goal[1], 4)],
            "target_yaw": round(target_yaw, 4),
            "planned_distance_m": round(
                sum(
                    math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                    for i in range(len(path) - 1)
                ),
                4,
            ),
            "path_xy": [(round(px, 4), round(py, 4)) for px, py in path],
        }

    def render_topdown_rgb(
        self,
        robot_pose: tuple[float, float, float],
        *,
        scale: int = 3,
    ) -> np.ndarray:
        """Render a crisp RGB visualization of the 2D LiDAR occupancy map, A* path, and Go2 pose."""
        from PIL import Image, ImageDraw

        img = np.zeros((self.ny, self.nx, 3), dtype=np.uint8)
        img[self.grid == CELL_UNKNOWN] = (34, 38, 46)
        img[self.grid == CELL_FREE] = (226, 232, 240)
        img[self.grid == CELL_OCCUPIED] = (20, 24, 31)

        # Flip vertically so +Y (north) is at the top of the image
        img = np.flipud(img)
        pil_img = Image.fromarray(img).resize(
            (self.nx * scale, self.ny * scale), resample=Image.Resampling.NEAREST
        )
        draw = ImageDraw.Draw(pil_img)

        def to_px(wx: float, wy: float) -> tuple[float, float]:
            ix = (wx + self.half_w) / self.res
            iy = (self.ny - 1) - ((wy + self.half_h) / self.res)
            return ix * scale, iy * scale

        # Historical walked trajectory (blue)
        if len(self.trajectory_xy) >= 2:
            pts = [to_px(x, y) for x, y in self.trajectory_xy]
            draw.line(pts, fill=(59, 130, 246), width=max(2, scale - 1))

        # Active A* planned path (emerald green)
        if len(self.last_path_xy) >= 2:
            pts = [to_px(x, y) for x, y in self.last_path_xy]
            draw.line(pts, fill=(16, 185, 129), width=max(2, scale))

        # Active PixelGoal target (amber ring)
        if self.last_goal_xy is not None:
            gx, gy = to_px(*self.last_goal_xy)
            r = 5
            draw.ellipse((gx - r, gy - r, gx + r, gy + r), outline=(245, 158, 11), width=2)

        # Go2 robot pose & heading arrow (crimson)
        rx, ry, ryaw = robot_pose
        rpx, rpy = to_px(rx, ry)
        rr = max(5, int(round(self.robot_radius_m / self.res * scale * 0.65)))
        draw.ellipse((rpx - rr, rpy - rr, rpx + rr, rpy + rr), fill=(220, 38, 38), outline=(255, 255, 255))
        hx = rpx + math.cos(ryaw) * (rr * 1.8)
        hy = rpy - math.sin(ryaw) * (rr * 1.8)
        draw.line([(rpx, rpy), (hx, hy)], fill=(255, 255, 255), width=2)

        return np.asarray(pil_img, dtype=np.uint8)
