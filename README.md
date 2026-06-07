# inter-baseline

本项目当前主要入口是 `mappo.py` 和 `evaluate_checkpoint.py`。命令建议在仓库根目录执行：

```powershell
cd C:\Users\admin\Project\inter-baseline
```

## 训练

实际训练命令：

```powershell
uv run python mappo.py --env-type highway --env-name intersection-multi-agent-v1 --device cpu --total-timesteps 1000000 --batch-size 3 --eval-steps 10 --num-eval-ep 10
```

快速冒烟训练：

```powershell
uv run python mappo.py --env-type highway --env-name intersection-multi-agent-v1 --device cpu --total-timesteps 5000 --batch-size 3 --eval-steps 5 --num-eval-ep 2 --checkpoint-dir runs\smoke
```

如果要传入 highway-env 配置覆盖，使用 UTF-8 JSON 文件：

```powershell
uv run python mappo.py --env-type highway --env-name intersection-multi-agent-v1 --env-config-path path\to\env_config.json --device cpu
```

训练输出目录形如：

```text
runs\MAPPO-highway__intersection-multi-agent-v1__YYYY-MM-DD_HH-MM-SS\
```

常用训练参数：

- `--env-type`: 默认 `highway`。
- `--env-name`: 默认 `intersection-multi-agent-v1`。
- `--env-config-path`: 可选，读取 UTF-8 JSON 环境配置覆盖。
- `--agent-ids` / `--no-agent-ids`: 是否在观测中加入 agent one-hot id，默认开启。
- `--total-timesteps`: 总环境步数，默认 `1000000`。
- `--batch-size`: 每次 rollout 收集的 episode 数，默认 `3`。
- `--eval-steps`: 每多少个训练迭代做一次评估，默认 `10`。
- `--num-eval-ep`: 每次评估 episode 数，默认 `10`。
- `--save-checkpoints` / `--no-save-checkpoints`: 是否保存 checkpoint，默认开启。
- `--checkpoint-dir`: TensorBoard 和 checkpoint 根目录，默认 `runs`。
- `--checkpoint-interval`: 每 N 次评估另存一次 checkpoint，`0` 表示关闭。
- `--checkpoint-best-metric`: 最优 checkpoint 指标，默认 `eval/ep_reward`。
- `--checkpoint-best-mode`: `max` 或 `min`，默认 `max`。
- `--device`: `cpu`、`cuda` 或 `mps`，默认 `cpu`。
- `--seed`: 随机种子，默认 `1`。

## 验证 checkpoint

验证指定 checkpoint 的实际效果：

```powershell
uv run python evaluate_checkpoint.py --checkpoint-path runs\MAPPO-highway__intersection-multi-agent-v1__2026-06-07_22-53-39\checkpoint_best.pt --num-episodes 5 --device cpu --deterministic --record-video --fps 15 --overlay
```

验证最新的 `checkpoint_best.pt`：

```powershell
$ckpt = Get-ChildItem runs -Recurse -Filter checkpoint_best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1
uv run python evaluate_checkpoint.py --checkpoint-path $ckpt.FullName --num-episodes 5 --device cpu --deterministic --record-video --fps 15 --overlay
```

只看终端指标、不录视频：

```powershell
uv run python evaluate_checkpoint.py --checkpoint-path runs\MAPPO-highway__intersection-multi-agent-v1__2026-06-07_22-53-39\checkpoint_best.pt --num-episodes 10 --device cpu --deterministic --no-record-video
```

验证脚本会从 checkpoint 里恢复训练时保存的 `env_type`、`env_name`、`agent_ids`、网络结构和 `env_config`。当前脚本会把 trace 固定写到 checkpoint 同目录：

```text
checkpoint_best_trace.jsonl
```

如果开启 `--record-video`，每个 episode 的视频也会写到 checkpoint 同目录：

```text
checkpoint_best_ep000.mp4
checkpoint_best_ep001.mp4
...
```

常用验证参数：

- `--checkpoint-path`: 必填，MAPPO actor checkpoint 路径。
- `--num-episodes`: 验证 episode 数，默认 `3`。
- `--seed`: 可选；不传时使用 checkpoint 中保存的训练 seed。
- `--device`: `cpu`、`cuda` 或 `mps`，默认 `cpu`。
- `--deterministic` / `--no-deterministic`: 贪心动作或采样动作，默认贪心。
- `--display` / `--no-display`: 是否弹出 pygame 窗口显示标注帧，默认关闭。
- `--record-video` / `--no-record-video`: 是否录制 MP4，默认开启。
- `--fps`: 视频和显示帧率，默认 `15`；传 `0` 时使用环境渲染帧率。
- `--overlay` / `--no-overlay`: 是否在视频帧上绘制策略和环境诊断信息，默认开启。

## 环境 wrapper 快速检查

```powershell
uv run python demo_highway_wrapper.py
```

这个命令用于检查 `HighwayWrapper` 的观测维度、状态维度、动作维度、奖励和终止语义。
