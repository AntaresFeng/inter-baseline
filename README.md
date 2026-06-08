# inter-baseline

本项目当前主要入口是 `mappo.py` 和 `evaluate_checkpoint.py`。命令建议在仓库根目录执行：

```powershell
cd C:\Users\admin\Project\inter-baseline
```

## 训练

MLP 基线训练命令：

```powershell
uv run python mappo.py --env-type highway --env-name intersection-multi-agent-v1 --device cpu --total-timesteps 1000000
```

快速冒烟训练：

```powershell
uv run python mappo.py --env-type highway --env-name intersection-multi-agent-v1 --device cpu --total-timesteps 5000 --batch-size 3 --eval-steps 5 --num-eval-ep 2 --checkpoint-dir runs\smoke
```

如果要传入 highway-env 配置覆盖，使用 UTF-8 JSON 文件：

```powershell
uv run python mappo.py --env-type highway --env-name intersection-multi-agent-v1 --device cpu --env-config-path path\to\env_config.json --normalize-advantage --clip-gradients 1.0 --entropy-coef 0.005 --learning-rate-actor 0.0003 --learning-rate-critic 0.0005
```

带 attention 的推荐训练命令。建议先跑 `200000` 步做 sanity training，看 TensorBoard 中 `eval/ep_reward`、`eval/std_ep_reward`、`train/entropy` 和 `train/critic_loss` 是否正常；曲线稳定后再把 `--total-timesteps` 改成 `1000000` 做正式训练：

```powershell
uv run python .\mappo.py `
  --env-type highway `
  --env-name intersection-multi-agent-v1 `
  --env-config-path env_config.json `
  --device cpu `
  --actor-model attention `
  --critic-model attention `
  --attention-embed-dim 64 `
  --attention-heads 2 `
  --total-timesteps 200000 `
  --batch-size 8 `
  --normalize-advantage `
  --clip-gradients 1.0 `
  --entropy-coef 0.005 `
  --learning-rate-actor 0.0003 `
  --learning-rate-critic 0.0005
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
- `--actor-model`: `mlp` 或 `attention`，默认 `mlp`。
- `--critic-model`: `mlp` 或 `attention`，默认 `mlp`。
- `--attention-embed-dim`: attention entity embedding 维度，默认 `64`。
- `--attention-heads`: attention head 数，默认 `2`，对应 rl-agents 的 self-attention 2-head 配置。
- `--total-timesteps`: 总环境步数，默认 `1000000`。
- `--batch-size`: 每次 rollout 收集的 episode 数，默认 `8`。
- `--normalize-advantage`: 是否标准化 advantage，建议 attention 训练开启。
- `--clip-gradients`: 梯度裁剪阈值；`<= 0` 表示关闭，attention 训练建议 `1.0`。
- `--entropy-coef`: entropy bonus 系数，attention 训练建议从 `0.005` 开始。
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
