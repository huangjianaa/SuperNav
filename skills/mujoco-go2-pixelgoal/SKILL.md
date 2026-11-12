---
name: mujoco-go2-pixelgoal
description: Navigate Unitree Go2 in MuJoCo 4.0 indoor scenes using 4-way surround RGB-D cameras, 360 LiDAR occupancy mapping, and PixelGoal A* path execution.
---

# MuJoCo 4.0 Unitree Go2 PixelGoal Navigation Skill

## Sensor & Action Contract
- **Embodiment**: 12-DoF physical Unitree Go2 quadruped (`mujoco_menagerie` `unitree_go2`) walking with closed-loop RL torque control in MuJoCo 4.0 (`speculative="enable"`).
- **Observation (`observe_surround`)**: Returns JSON telemetry + 4 RGB images in strict order:
  1. `front` (`0°`)
  2. `right` (`-90°`)
  3. `back` (`180°`)
  4. `left` (`+90°`)
- **Navigation (`pixelgoal_navigate(point=[x, y], direction=...)`)**:
  - `point` is normalized `[x, y]` in `[0.0, 1.0]^2` (`[0, 0]` = top-left, `[1, 1]` = bottom-right).
  - Unprojects metric depth onto the 360° LiDAR occupancy grid, inflates obstacles by Go2's body radius (`0.28 m`), plans an 8-connected A* path, and drives Go2 along the waypoints.
  - Returns updated 4-way surround images upon arrival.
- **Completion (`claim_goal(description=..., direction=..., point=[x, y])`)**:
  - Call ONLY when Go2 has physically approached within `~1.0 m` of the target object and confirmed it visually.

## Best Practices
1. **Doorways & Partitions**: Click open floor (`y` between `0.62` and `0.84`) past the corner or doorway rather than clicking directly on an occluding partition wall.
2. **Approaching Targets**: When the target object is visible across the room, click the open floor just in front of its base (or the lower portion of the target; vertical surface hits automatically stand off `0.40 m` in front of the object).
3. **Avoid Oscillation**: Check `pose` and the 4 views before reversing direction; prefer unexplored open corridors when searching.
