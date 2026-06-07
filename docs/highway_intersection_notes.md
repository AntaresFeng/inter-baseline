# Highway Intersection Notes

## MAPPO action semantics

当前 `HighwayWrapper` 使用 `intersection-multi-agent-v1`。每个 agent 的动作空间是
`Discrete(3)`，底层来自 highway-env 的 `DiscreteMetaAction`：

- `0`: `SLOWER`
- `1`: `IDLE`
- `2`: `FASTER`

`[0, 4.5, 9]` 不是动作表，而是 `MDPVehicle.target_speeds` 目标速度挡位表。
`SLOWER` / `FASTER` 会先把当前实际速度映射到最近的速度挡位，再降一档或升一档并裁剪到合法范围。
`IDLE` 保持当前目标速度。

## Reset-time speeds

只看 `intersection-multi-agent-v1`：

- 背景 IDM 车辆由 `_spawn_vehicle()` 生成，初始速度是
  `8.0 + normal() * speed_deviation`。普通背景车使用 `speed_deviation=1.0`。
- 普通背景车生成后，环境会先 warm up 3 秒，因此 reset 返回时它们的速度已经可能偏离初始采样值。
- warm up 之后会生成一辆 guaranteed challenger，`speed_deviation=0.0`，所以生成速度是
  `8.0 m/s`，但如果离受控车过近，后续防碰撞逻辑可能移除它。
- 受控车最后生成在入口车道，初始实际速度使用 `ego_lane.speed_limit`。当前 intersection 入口车道限速是
  `10.0 m/s`。
- 受控车是 `MDPVehicle`，目标速度挡位为 `[0, 4.5, 9]`。因此 reset 后通常是
  `speed=10.0`、`speed_index=2`、`target_speed=9.0`。

运行时检查也符合这个结论：受控车 reset 在最高目标速度挡位，背景车速度随 seed、生成位置和 warm up 仿真变化。
