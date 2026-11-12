"""MCP launch specification builder for the MuJoCo 4.0 Go2 backend (no simulator SDK imports)."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Mapping

from supernav.backends.mujoco import DEFAULT_BRIDGE_PORT
from supernav.paths import python_paths
from supernav.runtime.mcp import MCPServerSpec


def build_mcp_spec(
    *,
    workspace_root: Path,
    config: Mapping[str, Any],
    arm_cfg: Mapping[str, Any],
) -> MCPServerSpec:
    del workspace_root
    mcp_cfg = config.get("mcp") if isinstance(config.get("mcp"), Mapping) else {}
    bridge_cfg = config.get("bridge") if isinstance(config.get("bridge"), Mapping) else {}
    bridge_port = int(bridge_cfg.get("port") or DEFAULT_BRIDGE_PORT)
    transport = str(mcp_cfg.get("transport") or "stdio")

    env: dict[str, str] = {"PYTHONDONTWRITEBYTECODE": "1"}
    if python_paths():
        env["PYTHONPATH"] = os.pathsep.join(python_paths())
    for extra in (mcp_cfg.get("environment") or {}, arm_cfg.get("environment") or {}):
        if isinstance(extra, Mapping):
            env.update({str(k): str(v) for k, v in extra.items()})

    default_args = (
        "-m",
        "supernav",
        "mcp",
        "--backend",
        "mujoco",
        "--bridge-port",
        str(bridge_port),
        "--transport",
        transport,
    )
    raw_args = mcp_cfg.get("args")
    args = tuple(str(a) for a in raw_args) if isinstance(raw_args, list) else default_args
    http_args = (
        "-m",
        "supernav",
        "mcp",
        "--backend",
        "mujoco",
        "--bridge-port",
        str(bridge_port),
        "--transport",
        "streamable-http",
    )
    return MCPServerSpec(
        name=str(mcp_cfg.get("name") or "mujoco-go2"),
        command=str(mcp_cfg.get("command") or sys.executable),
        args=args,
        environment=env,
        transport=transport,
        http_args=http_args,
    )
