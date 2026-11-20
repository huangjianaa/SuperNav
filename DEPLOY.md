# SuperNav 本机启动指南

适用于本机（Ubuntu 22.04 / RTX 5070 Ti / CUDA 12.8）已经完成一次完整部署的环境。
从零安装步骤见 [`docs/local-reproduction.zh-CN.md`](docs/local-reproduction.zh-CN.md)。

## 一句话

```bash
cd /home/hj/hj/SuperNav && source .cache/env.sh
```

这一行会设好仿真器 Python、显示、代理、视觉模型端点。**每个新终端都要执行一次**。
然后按下面三种方式之一启动。

---

## 方式一：MuJoCo + Go2 四足机器人（最快，1 分钟）

不需要显示服务，不需要下载数据，EGL 离屏渲染。

```bash
source .cache/env.sh
MUJOCO_GL=egl .venv/bin/supernav run \
  --config configs/local/mujoco-go2-ark.json \
  --task-ids mujoco4_go2_office_pantry_task01 \
  --output-dir data/runs/mujoco-live
```

产物在 `data/runs/mujoco-live/<episode>/metrics.json`，里面 `success`、`spl`、`tool_calls` 都有。
单次约 1–3 分钟，跑 4–6 次导航动作。

---

## 方式二：Habitat-GS 单目标 ObjectNav（真实论文仿真器）

需要 `DISPLAY=:1`（env.sh 已设），因为 3DGS 渲染走 CUDA-GL 互操作，无显示会白屏。
下面这条已实测跑通：`success=True`，9 次导航 + 1 次转向后找到目标。

```bash
source .cache/env.sh
.venv/bin/supernav run \
  --config configs/local/habitat-gs-ark.json \
  --task-ids interior_0039_839888__task_001_find_black_geometric_bookshelf \
  --output-dir .cache/supernav-gs/objectnav/runs-ark \
  --run-tag mytest
```

可选的 4 个任务（`--task-ids` 换掉即可，或用逗号分隔跑多个）：

| task-id | 场景 |
|---|---|
| `interior_0039_839888__task_001_find_black_geometric_bookshelf` | 0039 |
| `interior_0042_839881__task_001_find_marble_reception_desk` | 0042 |
| `interior_0042_839881__task_002_find_microwave_oven` | 0042 |
| `interior_0042_839881__hardtask_find_silver_refrigerator` | 0042 |

> **必须换 `--run-tag`**：同一 episode 重复跑进同一目录后，导出评分会拒绝（防止无意重试）。

---

## 方式三：Habitat-GS 多目标任务（测 SR1–SR5）

```bash
source .cache/env.sh
.venv/bin/supernav run \
  --config configs/local/habitat-multi-ark.json \
  --task-ids multi_objectnav_17_000009 \
  --output-dir .cache/supernav-gs/multi-objectnav/runs-ark \
  --run-tag mytest
```

可选 `multi_objectnav_17_000008` … `_000013`（000008/9/10 在 0039 场景，11/12/13 在 0042）。
这个单次约 10–15 分钟（要依次找多个目标）。

---

## 跑完怎么评分（论文口径 SR/SPL，不是 metrics.json 里的 success）

```bash
source .cache/env.sh
TASK=objectnav   # 多目标改成 multi-objectnav
BASE=.cache/supernav-gs/$TASK

# 1) 导出轨迹证据（--visuals-root 固定是这个目录）
.venv/bin/supernav export-habitatgs \
  --dataset-root .cache/habitatgs-benchmark --task $TASK \
  --runs-root $BASE/runs-ark \
  --visuals-root $BASE/visuals \
  --output $BASE/submissions.json

# 2) 用仿真器环境 + 官方 GT 评分
.cache/venv-habitat/bin/python -m supernav score-habitatgs \
  --dataset-root .cache/habitatgs-benchmark --task $TASK \
  --scene-root .cache/gs-scenes-runtime \
  --submissions $BASE/submissions.json \
  --reference global_shortest --output $BASE/scores.json
```

看 `scores.json` 的 `summary`。注意：**只跑了少数任务时，其余会按 missing 计失败**，
所以成功率看起来很低，这是正常现象，不是模型表现。

---

## 实时观看仿真过程

两个终端，都先 `source .cache/env.sh`。

终端 A（查看器）：

```bash
.venv/bin/supernav web --live-dir "$SUPERNAV_LIVE_DIR" \
  --runs-root .cache/supernav-gs/objectnav/runs-ark --port 8766
```

浏览器打开 <http://127.0.0.1:8766> ，能看到四路 RGB 观测、轨迹和工具调用。

终端 B：跑上面任意一条 `supernav run`。

---

## 重启仿真

`supernav run` 默认**跳过已存在的运行目录**，所以同一个任务第二次跑只会打印
`[skip] ...: already exists`。这不是卡住，上次那次通常已经正常结束。

三种重启方式：

```bash
source .cache/env.sh

# ① 覆盖重跑（推荐）：同一目录，证据目录内新增 simulator-N/
.venv/bin/supernav run --config configs/local/mujoco-go2-ark.json \
  --task-ids mujoco4_go2_office_pantry_task01 --overwrite \
  --output-dir data/runs/mujoco-live

# ② 新标签：保留历史，生成 ..._run2 新目录
.venv/bin/supernav run --config configs/local/mujoco-go2-ark.json \
  --task-ids mujoco4_go2_office_pantry_task01 --run-tag run2 \
  --output-dir data/runs/mujoco-live

# ③ 彻底清空重来
rm -rf data/runs/mujoco-live
.venv/bin/supernav run --config configs/local/mujoco-go2-ark.json \
  --task-ids mujoco4_go2_office_pantry_task01 \
  --output-dir data/runs/mujoco-live
```

| 方式 | 目录 | 适用 |
|---|---|---|
| `--overwrite` | 同一个，新增 `simulator-N/` | 调试迭代 |
| `--run-tag xxx` | 新建 `..._xxx` | 保留对比；**Habitat 评分必须用这个** |
| `rm -rf` 后重跑 | 全新 | 丢弃所有历史 |

如果命令真的挂住（而不是 `[skip]`），先查残留进程和端口：

```bash
ps -eo pid,cmd | grep -E "mujoco.http_server|habitat-bridge|supernav mcp" | grep -v grep
ss -ltn | grep -E ":(18911|18941)\b"
pkill -f "supernav habitat-bridge"     # Habitat 固定端口时才需要
```

MuJoCo 每次运行自动挑空闲端口，一般不会冲突；Habitat 的 `bridge.port` 固定，
残留进程会占住它。

---

## 常见的坑（都是本机踩过的）

| 现象 | 原因 / 解决 |
|---|---|
| `[skip] ...: already exists` | 运行目录已存在，用 `--overwrite` 或 `--run-tag`，见上一节 |
| `No Stage Attributes exists ... dataset 'default'` | 已修复（`src/supernav/runtime/openai_agent.py` 合并 `_benchmark_process_environment`）；若换回旧代码会复现 |
| Habitat 画面全白 | 没设 `DISPLAY=:1`。检查 `pgrep -a Xorg` 有进程，`DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority` |
| `Bridge startup timed out` | `bridge.port` 必须是具体端口，不能是 0；或端口被占（`ss -ltn \| grep 18911`） |
| MCP/bridge 连不上 | 回环被代理劫持了，确认 `NO_PROXY=127.0.0.1,localhost` |
| 查看器里 MuJoCo 轨迹断成直线段 | 已修复（见下节）；旧录像仍是旧轨迹，需重跑 |
| 想换模型 | 改 `configs/local/*.json` 里的 `model.name` 和 `provider.base_url`；ARK 上 `doubao-seed-2-1-pro-260915` 已验证视觉+工具调用 |

### MuJoCo 查看器轨迹不连续（已修复）

原因：MuJoCo 后端原先只在一次导航跳（hop）**结束后**推送一次身体位姿，而一次 hop
会走 1–1.5m 的 A\* 曲线路径，于是地图上的轨迹变成一串 1.4m 的直线跳跃。Habitat
每步都推，所以看不到这个问题。

修复：

- `src/supernav/backends/mujoco/simulator.py` 在 `follow_waypoints` 的
  `step_callback` 里推送中间位姿；
- `src/supernav/runtime/live.py` 的 `LivePublisher.sample` 支持 `capture=None`，
  只更新轨迹、跳过离屏渲染（否则 5Hz 渲染会让仿真变慢）。

效果（同一任务实测）：单段最大跳变 **1.45 m → 0.067 m**，轨迹点从 4–11 个增至
约 90–124 个；单次运行耗时仍在 1–2 分钟。

回归测试：`tests/test_web_viewer.py::test_pose_only_sample_records_trail_without_capture`
与 `tests/mujoco/test_mujoco_go2_pipeline.py` 的 `step_callback` 断言。

> 看旧录像仍会断线——那些 JSON 是修复前写的。用 `--overwrite`/`--run-tag`
> 重跑一次即可。

---

## 已就绪的资源

- `.cache/env.sh` — 环境脚本（被 gitignore）
- `configs/local/{mujoco-go2-ark,habitat-gs-ark,habitat-multi-ark}.json` — 三个本机配置
- 2 个 Habitat 场景 + 4 单目标 / 6 多目标任务 + 1 个 MuJoCo 场景
- 完整流程与已知限制见 [`docs/local-reproduction.zh-CN.md`](docs/local-reproduction.zh-CN.md)
