"""Agent-facing MCP server for Unitree Go2 PixelGoal navigation in MuJoCo 4.0."""

import argparse
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Optional
import urllib.error
import urllib.request


def _encode_view_jpeg(path_str: str, max_side: int = 512) -> bytes:
    from PIL import Image as PILImage

    img = PILImage.open(path_str).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=84)
    return buf.getvalue()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18941, help="HTTP MCP port when transport=streamable-http.")
    parser.add_argument("--bridge-port", type=int, required=True, help="Loopback Go2 MuJoCo bridge HTTP port.")
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    args = parser.parse_args(argv)

    from mcp.server.fastmcp import FastMCP, Image

    mcp = FastMCP("mujoco-go2-pixelgoal", host="127.0.0.1", port=args.port)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    bridge_base = f"http://127.0.0.1:{args.bridge_port}"

    def _call_bridge(route: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else b"{}"
        req = urllib.request.Request(
            f"{bridge_base}{route}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener.open(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"MuJoCo Go2 bridge error ({exc.code}): {detail}") from None

    def _format_with_views(result: dict[str, Any]):
        obs = result.get("observation") if isinstance(result.get("observation"), dict) else result
        raw_views = obs.get("views") if isinstance(obs, dict) else None
        images = []
        public_views: list[dict[str, Any]] = []
        if isinstance(raw_views, list):
            for v in raw_views:
                if not isinstance(v, dict):
                    continue
                vpath = v.get("path")
                if isinstance(vpath, str) and Path(vpath).is_file():
                    images.append(Image(data=_encode_view_jpeg(vpath), format="jpeg"))
                public_views.append(
                    {
                        "direction": v.get("direction"),
                        "image_ref": v.get("image_ref"),
                        "heading_deg": v.get("heading_deg"),
                    }
                )
        sanitized = json.loads(json.dumps(result))
        target_obs = sanitized.get("observation") if isinstance(sanitized.get("observation"), dict) else sanitized
        if isinstance(target_obs, dict):
            target_obs.pop("map_path", None)
            if public_views:
                target_obs["views"] = public_views
                target_obs["view_order"] = ["front", "right", "back", "left"]
        return [json.dumps(sanitized, ensure_ascii=False), *images]

    @mcp.tool()
    def observe_surround():
        """Capture 4-way surround RGB views (front, right, back, left) and update 360 LiDAR occupancy."""
        return _format_with_views(_call_bridge("/observe", {}))

    @mcp.tool()
    def pixelgoal_navigate(point: list[float], direction: str = "front"):
        """Navigate Unitree Go2 toward normalized pixel [x, y] in [0, 1]^2 on the chosen camera ('front', 'right', 'back', 'left')."""
        return _format_with_views(
            _call_bridge("/navigate", {"point": list(point), "direction": str(direction)})
        )

    @mcp.tool()
    def pixelgoal_status():
        """Query current Go2 pose, action budget, and last PixelGoal navigation state."""
        return json.dumps(_call_bridge("/status", {}), ensure_ascii=False)

    @mcp.tool()
    def pixelgoal_stop():
        """Immediately halt Go2 movement and hold standing posture without ending the episode."""
        return json.dumps(_call_bridge("/stop", {}), ensure_ascii=False)

    @mcp.tool()
    def claim_goal(
        description: str,
        direction: str = "front",
        point: Optional[list[float]] = None,
    ):
        """Declare that Go2 has arrived near the target object and terminate the episode."""
        return _format_with_views(
            _call_bridge(
                "/claim",
                {"description": description, "direction": direction, "point": point},
            )
        )

    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
