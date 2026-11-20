"""openai_mcp forwards private Habitat task bindings to the MCP child.

The Codex, OpenCode and Kimi runners merge ``_benchmark_process_environment``
into the MCP process environment; without the same merge the OpenAI-compatible
runner's ``init_scene`` fell back to the dataset ``default`` because the per-episode
scene/dataset/spawn bindings never reached the child.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from supernav.backends.habitat.experiment import HabitatBackend
from supernav.runtime.openai_agent import OpenAIMCPAgent

PRIVATE_KEYS = ("HAB_MCP_INIT_DEFAULTS_JSON", "HAB_MCP_NAV_GOALS_JSON")


def _prepare(agent, root):
    config = {
        "environment": {"backend": "habitat"},
        "benchmark_profile": "global_task",
        "scene": "private-scene",
        "scene_dataset_config_file": str(root / "private-dataset.json"),
        "spawn": {"start_position": [13.5, 0.25, -27.0], "sensor_height": 1.2},
        "bridge": {"host": "127.0.0.1", "port": 18911},
        "mcp": {},
    }
    arm = {"movement": "visual_point", "skill_mode": "none"}
    metadata = {
        "ordered": True,
        "targets": [{"index": 1, "description": "chair"}, {"index": 2, "description": "table"}],
    }
    HabitatBackend().prepare_task(
        config, SimpleNamespace(instruction="Find the furniture.", metadata=metadata), arm, root,
    )
    project = agent.prepare_project(
        run_dir=root, workspace_root=root, config=config, arm_cfg=arm, agent_cfg={}, model_cfg={},
    )
    return project, arm.get("_benchmark_process_environment", {})


@pytest.fixture
def clean_private_keys(monkeypatch):
    for key in PRIVATE_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_private_task_bindings_reach_openai_mcp_child(tmp_path, clean_private_keys):
    project, private = _prepare(OpenAIMCPAgent(), tmp_path / "episode")
    assert private, "backend must compile a private benchmark environment block"

    meta = json.loads((project / "openai_mcp_project.json").read_text())
    environment = meta["mcp"]["environment"]
    assert {key: environment[key] for key in PRIVATE_KEYS} == private

    defaults = json.loads(environment["HAB_MCP_INIT_DEFAULTS_JSON"])
    assert defaults["scene"] == "private-scene"
    assert defaults["scene_dataset_config_file"] == str(tmp_path / "episode" / "private-dataset.json")
    assert defaults["start_position"] == [13.5, 0.25, -27.0]
    assert json.loads(environment["HAB_MCP_NAV_GOALS_JSON"])["targets"][1]["description"] == "table"


def test_private_task_bindings_are_snapshot_not_alias(tmp_path, clean_private_keys):
    agent = OpenAIMCPAgent()
    project, private = _prepare(agent, tmp_path / "episode")
    private["HAB_MCP_INIT_DEFAULTS_JSON"] = "changed-after-preparation"

    meta = json.loads((project / "openai_mcp_project.json").read_text())
    assert "changed-after-preparation" not in meta["mcp"]["environment"]["HAB_MCP_INIT_DEFAULTS_JSON"]
    assert json.loads(meta["mcp"]["environment"]["HAB_MCP_INIT_DEFAULTS_JSON"])["scene"] == "private-scene"


def test_no_private_bindings_keeps_spec_environment_untouched(tmp_path, clean_private_keys):
    agent = OpenAIMCPAgent()
    config = {
        "environment": {"backend": "mujoco"},
        "mcp": {"name": "mujoco-go2", "transport": "stdio", "environment": {"KEEP": "1"}},
        "bridge": {"host": "127.0.0.1", "port": 0},
    }
    arm = {"movement": "pixelgoal", "skill_mode": "native"}
    project = agent.prepare_project(
        run_dir=tmp_path, workspace_root=Path(tmp_path), config=config, arm_cfg=arm,
        agent_cfg={}, model_cfg={},
    )
    meta = json.loads((project / "openai_mcp_project.json").read_text())
    environment = meta["mcp"]["environment"]
    assert environment["KEEP"] == "1"
    assert all(key not in environment for key in PRIVATE_KEYS)
