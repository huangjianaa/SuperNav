"""Native OpenAI-compatible multimodal MCP agent for SuperNav (e.g. Qwen3.8-Flash-Next).

Connects directly to any SuperNav stdio MCP server (such as ``mujoco-go2``) and drives
an OpenAI-compatible VLM endpoint with 4-view surround camera images, sliding-window
image pruning, ``<think>`` block extraction, and structured ``raw.jsonl`` / ``canonical.jsonl``
evidence logging. Also provides an image-only ``vision_heuristic`` controller for
offline verification when external LLM network access is unavailable.
"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Mapping

import numpy as np
from PIL import Image

from supernav.runtime.agents import AgentResult, model_name
from supernav.runtime.config import write_json
from supernav.runtime.mcp import resolve_mcp_spec
from supernav.runtime.providers import configured_provider, validate_provider_environment
from supernav.runtime.skill_runtime import copy_snapshot_skills
from supernav.runtime.streams import _clean_tool_name, iter_json_lines, write_canonical_events

_THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
_DIRECTIONS = ("front", "right", "back", "left")
_DIR_YAW_RAD = {
    "front": 0.0,
    "right": -0.5 * math.pi,
    "back": math.pi,
    "left": 0.5 * math.pi,
}


def _strip_think(text: str) -> tuple[str, str]:
    thoughts = [m.strip() for m in _THINK_RE.findall(text or "") if m.strip()]
    cleaned = _THINK_RE.sub("", text or "").strip()
    return cleaned, "\n".join(thoughts)


def _expand_env_url(raw_url: str) -> str:
    expanded = os.path.expandvars(str(raw_url or "")).strip()
    return expanded


def _detect_target_color_mode(prompt_text: str) -> str:
    low = prompt_text.lower()
    if any(k in low for k in ("绿", "green")):
        return "green"
    if any(k in low for k in ("蓝", "blue")):
        return "blue"
    return "red"


def _analyze_views_visually(
    views_b64: list[str],
    *,
    color_mode: str,
    robot_pose: Mapping[str, Any],
    visited_xy: list[tuple[float, float]],
    blocked_headings: list[float],
    action_count: int,
    last_step_stalled: bool = False,
) -> dict[str, Any]:
    """Pure image-space perception over 4 RGB camera views + robot odometry (zero GT access)."""
    best_target: dict[str, Any] | None = None
    best_open: dict[str, Any] | None = None

    rx = float(robot_pose.get("x", 0.0))
    ry = float(robot_pose.get("y", 0.0))
    ryaw = float(robot_pose.get("yaw_rad", 0.0))

    for idx, b64_data in enumerate(views_b64[:4]):
        direction = _DIRECTIONS[idx]
        arr = np.asarray(Image.open(BytesIO(base64.b64decode(b64_data))).convert("RGB"), dtype=np.float32)
        h, w, _ = arr.shape
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # 1) Target color segmentation in image space (robust to shaded sides of 3D cylinders/boxes)
        if color_mode == "green":
            mask = (g > 88.0) & ((g - np.maximum(r, b)) > 34.0) & (g > 1.55 * np.maximum(r, b))
        elif color_mode == "blue":
            mask = (b > 95.0) & ((b - np.maximum(r, g)) > 40.0) & (b > 1.60 * np.maximum(r, g))
        else:
            mask = (r > 88.0) & ((r - np.maximum(g, b)) > 40.0) & (r > 1.75 * np.maximum(g, b))

        mask[: int(0.02 * h), :] = False
        count = int(np.sum(mask))
        ratio = count / float(h * w)
        if ratio >= 0.0012:
            ys, xs = np.nonzero(mask)
            u_cent = float(np.mean(xs)) / max(1, w - 1)
            v_cent = float(np.mean(ys)) / max(1, h - 1)
            v_bot = float(np.percentile(ys, 92)) / max(1, h - 1)
            if best_target is None or ratio > best_target["ratio"]:
                best_target = {
                    "direction": direction,
                    "ratio": ratio,
                    "centroid": [round(u_cent, 4), round(v_cent, 4)],
                    "nav_point": [round(u_cent, 4), round(min(0.82, max(0.55, v_bot + 0.05)), 4)],
                }

        # 2) Cool-slate checkerboard floor detection (b > r + 5 on floor; walls/cabinets have warm r >= b)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        is_floor = ((b - r) > 5.0) & (chroma < 52.0) & (luma > 40.0) & (luma < 235.0)
        is_floor[: int(0.50 * h), :] = False

        for u_cand in (0.24, 0.37, 0.50, 0.63, 0.76):
            c0 = max(0, int((u_cand - 0.06) * w))
            c1 = min(w, int((u_cand + 0.06) * w))
            col_ratio = np.mean(is_floor[:, c0:c1], axis=1)

            horizon_row = h - 1
            miss = 0
            for row_i in range(h - 2, int(0.50 * h), -1):
                if col_ratio[row_i] >= 0.38:
                    horizon_row = row_i
                    miss = 0
                else:
                    miss += 1
                    if miss >= 4:
                        break

            openness = (h - 1 - horizon_row) / float(0.50 * h)
            if openness < 0.20:
                continue

            ray_yaw = ryaw + _DIR_YAW_RAD[direction] - (u_cand - 0.5) * math.radians(54.0)
            est_dist = min(1.60, 0.45 + 1.55 * openness)
            cand_wx = rx + est_dist * math.cos(ray_yaw)
            cand_wy = ry + est_dist * math.sin(ray_yaw)

            novelty_penalty = 0.0
            for vx, vy in visited_xy:
                d_vis = math.hypot(cand_wx - vx, cand_wy - vy)
                if d_vis < 1.35:
                    novelty_penalty += (1.35 - d_vis) * 0.55

            heading_penalty = 0.0
            for bh in blocked_headings:
                ang = abs(math.atan2(math.sin(ray_yaw - bh), math.cos(ray_yaw - bh)))
                if ang < 0.55:
                    heading_penalty += (0.55 - ang) * 1.8

            score = openness - novelty_penalty - heading_penalty + (0.05 if direction == "front" else 0.0)
            v_click = max(0.58, min(0.84, (horizon_row / float(h - 1)) + 0.08))
            if best_open is None or score > best_open["score"]:
                best_open = {
                    "direction": direction,
                    "score": score,
                    "openness": round(openness, 4),
                    "ray_yaw": ray_yaw,
                    "nav_point": [round(u_cand, 4), round(v_click, 4)],
                }

    if best_target is not None and not last_step_stalled:
        if best_target["ratio"] >= 0.052 and action_count >= 2:
            return {
                "tool": "claim_goal",
                "arguments": {
                    "description": f"Arrived in front of the {color_mode} target object",
                    "direction": best_target["direction"],
                    "point": best_target["centroid"],
                },
                "thought": (
                    f"Target ({color_mode}) fills {best_target['ratio']*100:.2f}% of `{best_target['direction']}` view; "
                    "Go2 is within arrival range. Claiming goal."
                ),
            }
        target_yaw = ryaw + _DIR_YAW_RAD[best_target["direction"]] - (best_target["centroid"][0] - 0.5) * math.radians(54.0)
        return {
            "tool": "pixelgoal_navigate",
            "arguments": {
                "direction": best_target["direction"],
                "point": best_target["nav_point"],
            },
            "ray_yaw": target_yaw,
            "thought": (
                f"Spotted {color_mode} target in `{best_target['direction']}` camera "
                f"(area={best_target['ratio']*100:.2f}%). Navigating toward {best_target['nav_point']}."
            ),
        }

    if best_open is not None:
        return {
            "tool": "pixelgoal_navigate",
            "arguments": {
                "direction": best_open["direction"],
                "point": best_open["nav_point"],
            },
            "ray_yaw": best_open["ray_yaw"],
            "thought": (
                f"Target not yet visible; exploring open floor in `{best_open['direction']}` "
                f"at {best_open['nav_point']} (openness={best_open['openness']})."
            ),
        }

    return {
        "tool": "pixelgoal_navigate",
        "arguments": {"direction": "left", "point": [0.5, 0.76]},
        "thought": "Limited floor visibility; rotating toward left view.",
    }


def canonicalize_openai_mcp(raw_path: str | Path, out_path: str | Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for idx, row in enumerate(iter_json_lines(raw_path)):
        etype = str(row.get("type") or "")
        ts = row.get("timestamp")
        if etype == "assistant":
            text = str(row.get("text") or "").strip()
            if text:
                events.append(
                    {
                        "type": "assistant_text",
                        "text": text,
                        "timestamp": ts,
                        "raw_index": idx,
                    }
                )
            for tc in row.get("tool_calls") or []:
                if not isinstance(tc, Mapping):
                    continue
                events.append(
                    {
                        "type": "tool_call",
                        "id": tc.get("id"),
                        "name": _clean_tool_name(str(tc.get("name") or "")),
                        "arguments": tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {},
                        "timestamp": ts,
                        "raw_index": idx,
                    }
                )
        elif etype == "tool_result":
            events.append(
                {
                    "type": "tool_result",
                    "id": row.get("tool_call_id"),
                    "name": _clean_tool_name(str(row.get("name") or "")),
                    "content": str(row.get("content") or ""),
                    "image_count": int(row.get("image_count") or 0),
                    "is_error": bool(row.get("is_error", False)),
                    "timestamp": ts,
                    "raw_index": idx,
                }
            )
    write_canonical_events(out_path, events)
    return events


class OpenAIMCPAgent:
    """In-process OpenAI-compatible VLM + stdio MCP agent (`agent = 'openai_mcp'`)."""

    name = "openai_mcp"

    def validate_environment(self, *, agent_cfg: Mapping[str, Any]) -> None:
        provider = configured_provider(agent_cfg)
        if provider and str(provider.get("type") or "openai") == "vision_heuristic":
            return
        validate_provider_environment(agent_cfg)

    def prepare_project(
        self,
        *,
        run_dir: Path,
        workspace_root: Path,
        config: Mapping[str, Any],
        arm_cfg: Mapping[str, Any],
        agent_cfg: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
    ) -> Path:
        project_dir = run_dir / "openai_mcp_project"
        project_dir.mkdir(parents=True, exist_ok=True)

        skills_dir = project_dir / "skills"
        snapshot_root = arm_cfg.get("_skill_snapshot_root")
        if snapshot_root:
            copy_snapshot_skills(Path(str(snapshot_root)), skills_dir)

        spec = resolve_mcp_spec(workspace_root=workspace_root, config=config, arm_cfg=arm_cfg)
        provider = configured_provider(agent_cfg)
        resolved_provider = None
        if provider:
            resolved_provider = {
                "id": provider["id"],
                "base_url": _expand_env_url(provider["base_url"]),
                "env_key": provider["env_key"],
                "type": str(provider.get("type") or "openai"),
                "mode": "explicit_api",
            }

        system_parts = [str(config.get("agent_instructions") or "").strip()]
        policy = str(config.get("agent_policy") or "").strip()
        if policy:
            system_parts.append(policy)
        skill_name = str(arm_cfg.get("skill_name") or "").strip()
        if skill_name:
            skill_md = skills_dir / skill_name / "SKILL.md"
            if skill_md.is_file():
                system_parts.append(skill_md.read_text(encoding="utf-8").strip())

        meta = {
            "agent": self.name,
            "model": {"name": model_name(model_cfg) or "Qwen3.8-Flash-Next"},
            "provider": resolved_provider,
            "system_prompt": "\n\n".join(p for p in system_parts if p),
            "tool_whitelist": list(arm_cfg.get("tool_whitelist") or []),
            "max_turns": int(agent_cfg.get("max_turns") or 18),
            "keep_image_turns": int(agent_cfg.get("keep_image_turns") or 2),
            "fallback_vision_heuristic": bool(agent_cfg.get("fallback_vision_heuristic", False)),
            "mcp": {
                "name": spec.name,
                "command": spec.command,
                "args": list(spec.args),
                "environment": dict(spec.environment),
            },
        }
        write_json(project_dir / "openai_mcp_project.json", meta)
        return project_dir

    def run(
        self,
        *,
        prompt: str,
        run_dir: Path,
        project_dir: Path,
        agent_cfg: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
        timeout_s: int,
    ) -> AgentResult:
        raw_path = run_dir / "raw.jsonl"
        stderr_path = run_dir / "stderr.log"
        raw_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")

        meta = json.loads((project_dir / "openai_mcp_project.json").read_text(encoding="utf-8"))
        started = time.monotonic()
        session_id = f"openai_mcp_{int(time.time())}"

        try:
            asyncio.run(
                asyncio.wait_for(
                    self._run_loop(
                        prompt=prompt,
                        meta=meta,
                        agent_cfg=agent_cfg,
                        model_cfg=model_cfg,
                        raw_path=raw_path,
                        stderr_path=stderr_path,
                        session_id=session_id,
                    ),
                    timeout=float(timeout_s),
                )
            )
            return AgentResult(
                returncode=0,
                duration_s=time.monotonic() - started,
                command=["python", "-m", "supernav.runtime.openai_agent"],
                raw_path=raw_path,
                stderr_path=stderr_path,
                session_id=session_id,
            )
        except asyncio.TimeoutError as exc:
            with stderr_path.open("a", encoding="utf-8") as err_f:
                err_f.write(f"TimeoutError after {timeout_s}s: {exc}\n")
            return AgentResult(
                returncode=124,
                duration_s=time.monotonic() - started,
                command=["python", "-m", "supernav.runtime.openai_agent"],
                raw_path=raw_path,
                stderr_path=stderr_path,
                session_id=session_id,
                timed_out=True,
                error=f"TimeoutError: exceeded {timeout_s}s",
            )
        except Exception as exc:
            with stderr_path.open("a", encoding="utf-8") as err_f:
                err_f.write(f"{type(exc).__name__}: {exc}\n")
            return AgentResult(
                returncode=1,
                duration_s=time.monotonic() - started,
                command=["python", "-m", "supernav.runtime.openai_agent"],
                raw_path=raw_path,
                stderr_path=stderr_path,
                session_id=session_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _run_loop(
        self,
        *,
        prompt: str,
        meta: Mapping[str, Any],
        agent_cfg: Mapping[str, Any],
        model_cfg: Mapping[str, Any],
        raw_path: Path,
        stderr_path: Path,
        session_id: str,
    ) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        mcp_cfg = meta["mcp"]
        merged_env = dict(os.environ)
        merged_env.update({str(k): str(v) for k, v in (mcp_cfg.get("environment") or {}).items()})
        server_params = StdioServerParameters(
            command=str(mcp_cfg["command"]),
            args=[str(a) for a in (mcp_cfg.get("args") or [])],
            env=merged_env,
        )

        provider = meta.get("provider") or configured_provider(agent_cfg) or {}
        provider_type = str(provider.get("type") or "openai")
        resolved_model = model_name(model_cfg) or str((meta.get("model") or {}).get("name") or "Qwen3.8-Flash-Next")
        base_url = _expand_env_url(str(provider.get("base_url") or os.environ.get("SUPERNAV_LLM_BASE_URL") or ""))
        env_key_name = str(provider.get("env_key") or "SUPERNAV_MAAS_API_KEY")
        api_key = os.environ.get(env_key_name, "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        fallback_heuristic = bool(meta.get("fallback_vision_heuristic", False))
        use_heuristic = provider_type == "vision_heuristic"

        client = None
        if not use_heuristic:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url or None,
                api_key=api_key or "EMPTY",
                timeout=float(agent_cfg.get("request_timeout_s") or 90.0),
            )

        def _log_raw(record: dict[str, Any]) -> None:
            record.setdefault("sessionID", session_id)
            record.setdefault("timestamp", time.time())
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        whitelist = set(meta.get("tool_whitelist") or [])
        max_turns = int(meta.get("max_turns") or 18)
        keep_image_turns = max(1, int(meta.get("keep_image_turns") or 2))
        color_mode = _detect_target_color_mode(prompt)

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as mcp_session:
                await mcp_session.initialize()
                listed = await mcp_session.list_tools()
                openai_tools: list[dict[str, Any]] = []
                available_names: set[str] = set()
                for t in listed.tools:
                    if whitelist and t.name not in whitelist:
                        continue
                    available_names.add(t.name)
                    openai_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description or "",
                                "parameters": t.inputSchema or {"type": "object", "properties": {}},
                            },
                        }
                    )

                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": str(meta.get("system_prompt") or "")},
                    {"role": "user", "content": prompt},
                ]
                _log_raw({"type": "session_start", "model": resolved_model, "provider_type": provider_type})

                last_views_b64: list[str] = []
                last_pose: dict[str, Any] = {}
                visited_xy: list[tuple[float, float]] = []
                blocked_headings: list[float] = []
                last_ray_yaw: float | None = None
                last_step_stalled = False
                last_action_count = 0
                image_group_seq = 0

                for turn_idx in range(1, max_turns + 1):
                    planned_calls: list[dict[str, Any]] = []
                    assistant_text = ""

                    if not use_heuristic and client is not None:
                        # Prune older image groups to keep VLM context compact
                        min_keep_group = max(1, image_group_seq - keep_image_turns + 1)
                        wire_messages = [
                            {k: v for k, v in m.items() if not k.startswith("_")}
                            for m in messages
                            if m.get("_image_group", min_keep_group) >= min_keep_group
                        ]
                        try:
                            resp = await asyncio.to_thread(
                                client.chat.completions.create,
                                model=resolved_model,
                                messages=wire_messages,
                                tools=openai_tools if openai_tools else None,
                                tool_choice="auto" if openai_tools else None,
                                temperature=float(agent_cfg.get("temperature", 0.2)),
                            )
                            choice_msg = resp.choices[0].message
                            raw_text = choice_msg.content or ""
                            clean_text, thought_text = _strip_think(raw_text)
                            assistant_text = clean_text or thought_text

                            assistant_msg: dict[str, Any] = {"role": "assistant", "content": raw_text or ""}
                            if choice_msg.tool_calls:
                                assistant_msg["tool_calls"] = [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments or "{}",
                                        },
                                    }
                                    for tc in choice_msg.tool_calls
                                ]
                                for tc in choice_msg.tool_calls:
                                    try:
                                        parsed_args = json.loads(tc.function.arguments or "{}")
                                    except json.JSONDecodeError:
                                        parsed_args = {}
                                    planned_calls.append(
                                        {
                                            "id": tc.id,
                                            "name": tc.function.name,
                                            "arguments": parsed_args if isinstance(parsed_args, dict) else {},
                                        }
                                    )
                            messages.append(assistant_msg)
                        except Exception as exc:
                            if not fallback_heuristic:
                                raise
                            with stderr_path.open("a", encoding="utf-8") as err_f:
                                err_f.write(
                                    f"LLM endpoint unreachable ({type(exc).__name__}: {exc}); "
                                    "activating fallback_vision_heuristic.\n"
                                )
                            use_heuristic = True

                    if use_heuristic:
                        if not last_views_b64:
                            planned_calls = [
                                {"id": f"call_heur_{turn_idx}", "name": "observe_surround", "arguments": {}}
                            ]
                            assistant_text = "Capturing initial 4-way surround camera observation."
                        else:
                            decision = _analyze_views_visually(
                                last_views_b64,
                                color_mode=color_mode,
                                robot_pose=last_pose,
                                visited_xy=visited_xy,
                                blocked_headings=blocked_headings,
                                action_count=last_action_count,
                                last_step_stalled=last_step_stalled,
                            )
                            last_ray_yaw = decision.get("ray_yaw")
                            planned_calls = [
                                {
                                    "id": f"call_heur_{turn_idx}",
                                    "name": decision["tool"],
                                    "arguments": decision["arguments"],
                                }
                            ]
                            assistant_text = decision["thought"]

                    _log_raw(
                        {
                            "type": "assistant",
                            "turn": turn_idx,
                            "text": assistant_text,
                            "tool_calls": planned_calls,
                        }
                    )

                    if not planned_calls:
                        # Agent finished speaking without requesting further tools
                        break

                    episode_terminal = False
                    for tc in planned_calls:
                        tname = tc["name"]
                        targs = tc["arguments"]
                        if tname not in available_names:
                            err_txt = json.dumps({"ok": False, "error": f"Unknown tool {tname}"})
                            _log_raw(
                                {
                                    "type": "tool_result",
                                    "tool_call_id": tc["id"],
                                    "name": tname,
                                    "content": err_txt,
                                    "image_count": 0,
                                    "is_error": True,
                                }
                            )
                            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": err_txt})
                            continue

                        mcp_res = await mcp_session.call_tool(tname, targs)
                        text_chunks: list[str] = []
                        img_b64_list: list[str] = []
                        img_mimes: list[str] = []
                        for item in mcp_res.content:
                            itype = getattr(item, "type", "")
                            if itype == "text" or hasattr(item, "text"):
                                text_chunks.append(str(item.text))
                            elif itype == "image" or hasattr(item, "data"):
                                raw_data = item.data
                                if isinstance(raw_data, bytes):
                                    b64_str = base64.b64encode(raw_data).decode("ascii")
                                else:
                                    b64_str = str(raw_data)
                                img_b64_list.append(b64_str)
                                img_mimes.append(str(getattr(item, "mimeType", None) or "image/jpeg"))

                        joined_text = "\n".join(text_chunks).strip() or "{}"
                        _log_raw(
                            {
                                "type": "tool_result",
                                "tool_call_id": tc["id"],
                                "name": tname,
                                "content": joined_text,
                                "image_count": len(img_b64_list),
                                "is_error": bool(getattr(mcp_res, "isError", False)),
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": joined_text,
                            }
                        )

                        # Track latest public observation for visual controller & terminal status
                        try:
                            parsed_payload = json.loads(text_chunks[0]) if text_chunks else {}
                        except Exception:
                            parsed_payload = {}

                        if isinstance(parsed_payload, dict):
                            nav_info = parsed_payload.get("navigation")
                            if isinstance(nav_info, dict):
                                walked = float(nav_info.get("distance_walked_m") or 0.0)
                                last_step_stalled = walked < 0.25 or nav_info.get("state") == "blocked_pixel_target"
                                if last_step_stalled and last_ray_yaw is not None:
                                    blocked_headings.append(float(last_ray_yaw))
                            else:
                                last_step_stalled = False
                            obs_obj = (
                                parsed_payload.get("observation")
                                if isinstance(parsed_payload.get("observation"), dict)
                                else parsed_payload
                            )
                            if isinstance(obs_obj.get("pose"), dict):
                                last_pose = dict(obs_obj["pose"])
                                px, py = float(last_pose.get("x", 0.0)), float(last_pose.get("y", 0.0))
                                if not visited_xy or math.hypot(px - visited_xy[-1][0], py - visited_xy[-1][1]) > 0.25:
                                    visited_xy.append((px, py))
                            if obs_obj.get("action_count") is not None:
                                last_action_count = int(obs_obj["action_count"])
                            if parsed_payload.get("terminal") is True or tname == "claim_goal":
                                episode_terminal = True

                        if img_b64_list:
                            last_views_b64 = img_b64_list
                            image_group_seq += 1
                            content_parts: list[dict[str, Any]] = [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Updated Unitree Go2 surround views after `{tname}` "
                                        "(Image 1=front, Image 2=right, Image 3=back, Image 4=left):"
                                    ),
                                }
                            ]
                            for b64_s, mime in zip(img_b64_list, img_mimes):
                                content_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{mime};base64,{b64_s}"},
                                    }
                                )
                            messages.append(
                                {
                                    "role": "user",
                                    "_image_group": image_group_seq,
                                    "content": content_parts,
                                }
                            )

                    if episode_terminal:
                        break

    def parse(self, raw_path: str | Path, out_path: str | Path) -> List[Dict[str, Any]]:
        return canonicalize_openai_mcp(raw_path, out_path)
