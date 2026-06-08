# AGENTS.md

## 提示

- TODO 文件是给人类用户的，不属于 agent 上下文。
- 在中文 Windows 上使用 PowerShell 5.1 读写 UTF-8 文件时，必须显式指定 `-Encoding UTF8` 参数。
- 需要写脚本，遇到转义、here-string、here-doc、编码问题，或可复用脚本，都推荐在 scripts/ 下写入文本并运行。

## 仓库概览

- `mappo.py`：主要训练/算法实现。
- `env/`：环境封装与配置。
- `demo_highway_wrapper.py`：highway 环境封装示例或快速验证入口。
- `docs/highway_env_faq.md`：highway-env 环境疑难杂症参考。
- `docs\highway_intersection_notes.md`：highway-env intersection 环境细节，目前包括动作空间说明和重置时速度

## 环境与工具

- 使用 `uv` 管理虚拟环境和运行 Python 命令。
- 可用命令：`jq`、`rg`、`fd`、`ruff`。
- 优先使用 `rg` / `fd` 搜索文件和文本
- Python 命令优先用：
  - `uv run python ...`

## 编码与编辑约定

- 保持改动小而聚焦，遵循现有代码风格。
- 不做无关重构、格式化或文件移动。
- 读写文本文件时保持 UTF-8 编码。
- 新增注释要解释不明显的设计原因，不要重复代码表面含义。
- 对配置、环境参数、训练参数的变更要谨慎，尽量说明行为影响。

## Debug 相关

- highway-env 源码的准确位置：
  - `uv run python -c "import highway_env;print(highway_env.__file__)"`
- highway-env 环境疑难杂症参考：
  - `docs\highway_env_faq.md`
