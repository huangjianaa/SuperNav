"""Unitree Go2 locomotion controller for MuJoCo 4.0 navigation.

Supports three execution modes:
- ``rl_torque`` (default): Closed-loop 50 Hz neural quadruped policy (ONNX) driving
  Go2's 12 position-servo joint actuators with full MuJoCo 4.0 contact physics.
- ``cpg_torque``: Pure-NumPy diagonal trot gait generator driving the 12 actuators
  when ONNX Runtime or the cached policy file is unavailable.
- ``kinematic``: Fast collision-checked SE(2) base stepping for rapid smoke tests.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any
import urllib.request

import numpy as np

from supernav.runtime.config import repo_root_from_here

ONNX_POLICY_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_playground/"
    "main/rss2025/mujoco_playground/experimental/sim2sim/onnx/go1_policy.onnx"
)
ONNX_POLICY_SHA256 = "386cde6a1eac679e2f2a313ade2b395d9f26905b151ee3b019c2c4163d49b6f2"
DEFAULT_CACHE_REL = Path(".cache/mujoco_go2/go2_locomotion.onnx")

# Involution mapping between Go2 [FL, FR, RL, RR] and policy [FR, FL, RR, RL]
_LEG_PERM = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)

_GO2_JOINT_NAMES = (
    "go2_FL_hip_joint",
    "go2_FL_thigh_joint",
    "go2_FL_calf_joint",
    "go2_FR_hip_joint",
    "go2_FR_thigh_joint",
    "go2_FR_calf_joint",
    "go2_RL_hip_joint",
    "go2_RL_thigh_joint",
    "go2_RL_calf_joint",
    "go2_RR_hip_joint",
    "go2_RR_thigh_joint",
    "go2_RR_calf_joint",
)

_GO2_ACTUATOR_NAMES = (
    "go2_FL_hip",
    "go2_FL_thigh",
    "go2_FL_calf",
    "go2_FR_hip",
    "go2_FR_thigh",
    "go2_FR_calf",
    "go2_RL_hip",
    "go2_RL_thigh",
    "go2_RL_calf",
    "go2_RR_hip",
    "go2_RR_thigh",
    "go2_RR_calf",
)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def ensure_onnx_policy(cache_path: Path | None = None, *, allow_download: bool = True) -> Path | None:
    """Locate or download the Apache-2.0 quadruped locomotion ONNX policy."""
    repo_root = repo_root_from_here()
    candidates = []
    if cache_path is not None:
        candidates.append(Path(cache_path))
    candidates.append(repo_root / DEFAULT_CACHE_REL)
    candidates.append(Path.home() / ".cache" / "supernav" / "go2_locomotion.onnx")

    for cand in candidates:
        if cand.is_file():
            digest = hashlib.sha256(cand.read_bytes()).hexdigest()
            if digest == ONNX_POLICY_SHA256:
                return cand

    if not allow_download:
        return None

    target = Path(cache_path) if cache_path is not None else (repo_root / DEFAULT_CACHE_REL)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ONNX_POLICY_URL, timeout=15.0) as resp:
            payload = resp.read()
        if hashlib.sha256(payload).hexdigest() != ONNX_POLICY_SHA256:
            return None
        target.write_bytes(payload)
        return target
    except Exception:
        return None


class Go2LocomotionController:
    """Closed-loop locomotion & waypoint follower for ``unitree_go2`` in MuJoCo 4.0."""

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        mode: str = "auto",
        policy_path: Path | None = None,
        action_scale: float = 0.5,
        ctrl_dt: float = 0.02,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.action_scale = float(action_scale)
        self.sim_dt = float(model.opt.timestep)
        self.n_substeps = max(1, int(round(float(ctrl_dt) / self.sim_dt)))
        self.ctrl_dt = self.n_substeps * self.sim_dt

        # Resolve Go2 body, joints, actuators, and IMU sensors by name
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "go2_base")
        if self.base_body_id < 0:
            raise ValueError("Compiled model is missing body 'go2_base'")
        base_jnt = int(model.body_jntadr[self.base_body_id])
        self.base_qpos_adr = int(model.jnt_qposadr[base_jnt])
        self.base_dof_adr = int(model.jnt_dofadr[base_jnt])

        self.qpos_idx = np.array(
            [
                int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                for n in _GO2_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self.qvel_idx = np.array(
            [
                int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                for n in _GO2_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self.act_idx = np.array(
            [
                int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n))
                for n in _GO2_ACTUATOR_NAMES
            ],
            dtype=np.int64,
        )
        self.imu_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "go2_imu"))

        if model.nkey > 0:
            self.default_angles = np.array(model.key_qpos[0, self.qpos_idx], dtype=np.float32)
            self.standing_z = float(model.key_qpos[0, self.base_qpos_adr + 2])
        else:
            self.default_angles = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float32)
            self.standing_z = 0.285

        self._default_angles_policy = self.default_angles[_LEG_PERM]
        self._last_action_policy = np.zeros(12, dtype=np.float32)
        self._cpg_phase = 0.0
        self._ort_session: Any | None = None

        requested = str(mode or "auto").lower()
        if requested in ("auto", "rl_torque", "onnx"):
            resolved_policy = ensure_onnx_policy(policy_path, allow_download=(requested != "cpg_torque"))
            if resolved_policy is not None:
                try:
                    import onnxruntime as ort

                    self._ort_session = ort.InferenceSession(
                        str(resolved_policy),
                        providers=["CPUExecutionProvider"],
                    )
                    self.mode = "rl_torque"
                except Exception:
                    if requested in ("rl_torque", "onnx"):
                        raise
                    self.mode = "cpg_torque"
            elif requested in ("rl_torque", "onnx"):
                raise RuntimeError("Requested rl_torque mode, but Go2 ONNX policy could not be found.")
            else:
                self.mode = "cpg_torque"
        elif requested in ("cpg_torque", "kinematic"):
            self.mode = requested
        else:
            raise ValueError(f"Unsupported locomotion mode: {mode!r}")

    def base_pose(self) -> tuple[float, float, float, float, float]:
        """Return ``(x, y, z, yaw_rad, up_z)`` of ``go2_base`` in world frame."""
        pos = self.data.xpos[self.base_body_id]
        xmat = self.data.xmat[self.base_body_id].reshape(3, 3)
        yaw = math.atan2(float(xmat[1, 0]), float(xmat[0, 0]))
        up_z = float(xmat[2, 2])
        return float(pos[0]), float(pos[1]), float(pos[2]), yaw, up_z

    def reset_gait_state(self) -> None:
        self._last_action_policy.fill(0.0)
        self._cpg_phase = 0.0
        self.data.ctrl[self.act_idx] = self.default_angles

    def settle_standing(self, duration_s: float = 0.4) -> None:
        """Hold standing posture briefly so feet settle cleanly onto ground contacts."""
        self.reset_gait_state()
        if self.mode == "kinematic":
            self._mujoco.mj_forward(self.model, self.data)
            return
        steps = max(1, int(round(duration_s / self.ctrl_dt)))
        for _ in range(steps):
            self.step_velocity(0.0, 0.0, 0.0)

    def _build_rl_obs(self, cmd_vx: float, cmd_vy: float, cmd_wz: float) -> np.ndarray:
        imu_xmat = self.data.site_xmat[self.imu_site_id].reshape(3, 3)
        global_linvel = np.asarray(self.data.sensor("go2_global_linvel").data, dtype=np.float32)
        local_linvel = (imu_xmat.T @ global_linvel).astype(np.float32)
        gyro = np.asarray(self.data.sensor("go2_gyro").data, dtype=np.float32)
        gravity = (imu_xmat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)

        qpos_policy = self.data.qpos[self.qpos_idx][_LEG_PERM].astype(np.float32)
        qvel_policy = self.data.qvel[self.qvel_idx][_LEG_PERM].astype(np.float32)
        joint_offsets = qpos_policy - self._default_angles_policy
        cmd = np.array([cmd_vx, cmd_vy, cmd_wz], dtype=np.float32)

        return np.concatenate(
            [
                local_linvel,
                gyro,
                gravity,
                joint_offsets,
                qvel_policy,
                self._last_action_policy,
                cmd,
            ],
            axis=0,
        ).astype(np.float32)

    def _step_cpg(self, cmd_vx: float, cmd_vy: float, cmd_wz: float) -> None:
        speed = math.hypot(cmd_vx, cmd_vy) + 0.35 * abs(cmd_wz)
        targets = self.default_angles.copy()
        if speed > 0.03:
            freq_hz = 2.4
            self._cpg_phase = (self._cpg_phase + 2.0 * math.pi * freq_hz * self.ctrl_dt) % (2.0 * math.pi)
            # Diagonal trot pairs: (FL, RR) at phase, (FR, RL) at phase + pi
            leg_phases = (
                self._cpg_phase,
                self._cpg_phase + math.pi,
                self._cpg_phase + math.pi,
                self._cpg_phase,
            )
            stride_pitch = float(np.clip(cmd_vx * 0.28, -0.22, 0.24))
            yaw_diff = float(np.clip(cmd_wz * 0.10, -0.14, 0.14))
            roll_step = float(np.clip(cmd_vy * 0.14, -0.12, 0.12))
            leg_signs = (1.0, -1.0, 1.0, -1.0)  # left=+1, right=-1
            for leg_i, (ph, side) in enumerate(zip(leg_phases, leg_signs)):
                swing = max(0.0, math.sin(ph))
                stance = math.cos(ph)
                hip_delta = roll_step * swing + (-yaw_diff * side * stance)
                thigh_delta = -(stride_pitch + side * yaw_diff * 0.5) * stance + 0.18 * swing
                calf_delta = -0.32 * swing
                base = 3 * leg_i
                targets[base + 0] += hip_delta
                targets[base + 1] += thigh_delta
                targets[base + 2] += calf_delta
        self.data.ctrl[self.act_idx] = targets
        for _ in range(self.n_substeps):
            self._mujoco.mj_step(self.model, self.data)

    def _step_kinematic(self, cmd_vx: float, cmd_vy: float, cmd_wz: float) -> None:
        x, y, _, yaw, _ = self.base_pose()
        nyaw = _wrap_angle(yaw + cmd_wz * self.ctrl_dt)
        nx = x + (cmd_vx * math.cos(yaw) - cmd_vy * math.sin(yaw)) * self.ctrl_dt
        ny = y + (cmd_vx * math.sin(yaw) + cmd_vy * math.cos(yaw)) * self.ctrl_dt

        adr = self.base_qpos_adr
        prev_qpos = self.data.qpos[adr : adr + 7].copy()
        half = 0.5 * nyaw
        self.data.qpos[adr : adr + 3] = [nx, ny, self.standing_z]
        self.data.qpos[adr + 3 : adr + 7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
        self.data.qpos[self.qpos_idx] = self.default_angles
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

        # Reject kinematic move if Go2 base/legs collide with any world obstacle (group 0)
        for ci in range(self.data.ncon):
            con = self.data.contact[ci]
            if con.dist < -0.01:
                g1_grp = int(self.model.geom_group[con.geom1])
                g2_grp = int(self.model.geom_group[con.geom2])
                g1_name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
                g2_name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
                if "ground_floor" in (g1_name, g2_name):
                    continue
                if 0 in (g1_grp, g2_grp):
                    self.data.qpos[adr : adr + 7] = prev_qpos
                    self._mujoco.mj_forward(self.model, self.data)
                    break

    def step_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Advance one control period (`ctrl_dt`, default 20 ms) under body velocity command."""
        cmd_vx = float(np.clip(vx, -0.65, 0.85))
        cmd_vy = float(np.clip(vy, -0.40, 0.40))
        cmd_wz = float(np.clip(wz, -1.40, 1.40))

        if self.mode == "kinematic":
            self._step_kinematic(cmd_vx, cmd_vy, cmd_wz)
            return
        if self.mode == "cpg_torque" or self._ort_session is None:
            self._step_cpg(cmd_vx, cmd_vy, cmd_wz)
            return

        obs = self._build_rl_obs(cmd_vx, cmd_vy, cmd_wz)
        pred = self._ort_session.run(["continuous_actions"], {"obs": obs.reshape(1, -1)})[0][0]
        pred = np.asarray(pred, dtype=np.float32)
        self._last_action_policy = pred.copy()
        ctrl_targets = self.default_angles + self.action_scale * pred[_LEG_PERM]
        self.data.ctrl[self.act_idx] = ctrl_targets
        for _ in range(self.n_substeps):
            self._mujoco.mj_step(self.model, self.data)

    def follow_waypoints(
        self,
        waypoints_xy: list[tuple[float, float]],
        *,
        target_yaw: float | None = None,
        max_duration_s: float = 18.0,
        arrival_tol_m: float = 0.22,
        lookahead_m: float = 0.48,
        max_vx: float = 0.65,
        step_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Drive Go2 along a 2D collision-free path using closed-loop pure pursuit."""
        if not waypoints_xy:
            x, y, z, yaw, up_z = self.base_pose()
            return {
                "state": "succeeded",
                "reason": "empty_path",
                "steps": 0,
                "sim_time_s": 0.0,
                "distance_walked_m": 0.0,
                "min_up_z": round(up_z, 4),
                "final_pose": {"x": round(x, 4), "y": round(y, 4), "z": round(z, 4), "yaw": round(yaw, 4)},
                "trajectory_xy": [(round(x, 4), round(y, 4))],
            }

        goal_x, goal_y = float(waypoints_xy[-1][0]), float(waypoints_xy[-1][1])
        x0, y0, z0, yaw0, up0 = self.base_pose()
        traj: list[tuple[float, float]] = [(round(x0, 4), round(y0, 4))]
        walked_m = 0.0
        min_up_z = up0
        prev_x, prev_y = x0, y0

        max_steps = max(1, int(round(max_duration_s / self.ctrl_dt)))
        wp_idx = 0
        best_goal_dist = math.hypot(goal_x - x0, goal_y - y0)
        stall_steps = 0
        status = "succeeded"
        reason = "arrived"

        for step_i in range(1, max_steps + 1):
            x, y, z, yaw, up_z = self.base_pose()
            min_up_z = min(min_up_z, up_z)
            step_d = math.hypot(x - prev_x, y - prev_y)
            walked_m += step_d
            prev_x, prev_y = x, y
            if step_i % 5 == 0:
                traj.append((round(x, 4), round(y, 4)))
                if step_callback is not None:
                    step_callback(step_i, (x, y, z, yaw, up_z))

            if up_z < 0.42 or z < 0.11:
                status = "failed"
                reason = "robot_fallen"
                break

            goal_dist = math.hypot(goal_x - x, goal_y - y)
            if goal_dist + 0.03 < best_goal_dist:
                best_goal_dist = goal_dist
                stall_steps = 0
            else:
                stall_steps += 1

            if goal_dist <= arrival_tol_m:
                if target_yaw is None:
                    status = "succeeded"
                    reason = "arrived"
                    break
                yaw_err_final = _wrap_angle(float(target_yaw) - yaw)
                if abs(yaw_err_final) <= 0.20:
                    status = "succeeded"
                    reason = "arrived"
                    break
                wz_cmd = float(np.clip(1.8 * yaw_err_final, -1.0, 1.0))
                self.step_velocity(0.0, 0.0, wz_cmd)
                continue

            # Advance lookahead waypoint along polyline
            while wp_idx + 1 < len(waypoints_xy):
                wx, wy = waypoints_xy[wp_idx]
                if math.hypot(wx - x, wy - y) < lookahead_m:
                    wp_idx += 1
                else:
                    break

            tx, ty = waypoints_xy[wp_idx]
            desired_yaw = math.atan2(ty - y, tx - x)
            yaw_err = _wrap_angle(desired_yaw - yaw)

            # Rotate in place when heading error is large so Go2 doesn't swing wide into walls
            if abs(yaw_err) > 0.70:
                vx_cmd = 0.05
                vy_cmd = 0.0
                wz_cmd = float(np.clip(1.9 * yaw_err, -1.25, 1.25))
            else:
                cos_scale = max(0.2, math.cos(yaw_err))
                speed_ramp = min(max_vx, max(0.20, 1.15 * goal_dist))
                vx_cmd = speed_ramp * cos_scale
                vy_cmd = float(np.clip(0.18 * math.sin(yaw_err), -0.15, 0.15))
                wz_cmd = float(np.clip(1.8 * yaw_err, -1.20, 1.20))

            self.step_velocity(vx_cmd, vy_cmd, wz_cmd)

            # Stall guard (e.g. grazed an unseen obstacle for > 3.2 s)
            if stall_steps > int(round(3.2 / self.ctrl_dt)):
                status = "partial"
                reason = "stalled_near_obstacle"
                break
        else:
            x_end, y_end, _, _, _ = self.base_pose()
            if math.hypot(goal_x - x_end, goal_y - y_end) <= arrival_tol_m * 1.45:
                status = "succeeded"
                reason = "arrived_near_tolerance"
            else:
                status = "partial"
                reason = "timeout"

        # Brief deceleration settle so cameras capture crisp unblurred images
        for _ in range(8):
            self.step_velocity(0.0, 0.0, 0.0)

        xf, yf, zf, yawf, upf = self.base_pose()
        min_up_z = min(min_up_z, upf)
        traj.append((round(xf, 4), round(yf, 4)))
        return {
            "state": status,
            "reason": reason,
            "locomotion_mode": self.mode,
            "steps": step_i if waypoints_xy else 0,
            "sim_time_s": round((step_i if waypoints_xy else 0) * self.ctrl_dt, 3),
            "distance_walked_m": round(walked_m, 4),
            "remaining_distance_m": round(math.hypot(goal_x - xf, goal_y - yf), 4),
            "min_up_z": round(min_up_z, 4),
            "final_pose": {
                "x": round(xf, 4),
                "y": round(yf, 4),
                "z": round(zf, 4),
                "yaw": round(yawf, 4),
            },
            "trajectory_xy": traj,
        }
