# 复现说明

## 1. 可直接检查的内容

```bash
uv sync --frozen
uv run python -m unittest discover -s tests_local -v
uv run python scripts/verify_research_results.py
```

离线测试使用脚本化响应和临时文件，不会发起模型请求。结果验证根据 `results/validation_outcomes.jsonl` 重算每轮的通过数、通过率、TDS、nDCG 和过度提问率，并核对冻结提示词。

`frozen_prompts/` 包含实际评估过的原始 CoT 和蒸馏 v1 提示词。`manifests/contrast_questions_train_v1.json` 保存最新训练实验的固定题目、种子与设置，作为记录；续跑使用原工作区未改变的清单。

## 2. 在新环境运行算法

两个明确的 SDK 单类入口：

- `clarify/algorithms/original_cot_clarifier.py`：原始 CoT。
- `clarify/algorithms/distilled_clarifier.py`：蒸馏 v1。

两者继承上游初稿、针对性提问、一次澄清与代码清理逻辑。蒸馏版只替换最终生成提示词。`contrast_question_clarifier.py` 含两个公开类，应使用专用实验入口，而不是 SDK 单类加载器。

先按 `.env.example` 配置接口。官方 SDK 直接使用 `OPENAI_API_KEY` 等环境变量；它不会自动读取 `.env`，也不会自动使用历史实验的 ZenMux 费用账本。以下示例会调用模型，仅运行 SDK 自带的小型演示数据：

```bash
mkdir -p runs/demo
docker pull ganler/evalplus
uv run python generate_responses.py clarify/algorithms/original_cot_clarifier.py --output_path=runs/demo/responses.jsonl --max_workers=1
uv run python evaluate_responses.py --generation_path=runs/demo/responses.jsonl --output_path=runs/demo/results.jsonl
```

替换算法路径可运行蒸馏版；完整验证时生成和评测两个命令都添加 `--split=val`。实际重新调用的服务、路由、参数和随机性可能导致结果不同。

本项目历史运行通过 `experiments/zenmux.py` 将模型请求路由到 ZenMux。`ZENMUX_API_KEY` 只对该本地路由入口有效。SDK 的单题预算与历史共享累计预算是两种不同限制，不能把上述 SDK 示例当成原工作区的受账本约束续跑。

## 3. 历史实验的精确重放限制

公开包不含旧的模型问答、完整结果、成功调用缓存与私有费用账本。因此依赖 `runs/` 的历史入口不能仅凭精简结果精确重放；缺少文件时应停止，不应构造假问答或清空账本。

蒸馏的四轮完整验证都是“固定历史澄清问答后的最终生成”实验。重新运行 SDK 会产生新的提问与回答，属于新的完整流程实验，与历史结果的评估范围不同。

原始 CoT 历史结果使用当时的四步提示词；当前 `cot_clarifier.py` 是 few-shot 变体。为保持已有清单与本地续跑，新整理没有重命名或覆盖这个历史文件，另提供了清晰的算法适配入口。

若在原工作区恢复新提问策略实验，先核实服务访问和费用记录。不要在余额不足时进行批量重试。随后使用原入口及原账本：

```bash
.venv/bin/python -m experiments.compare_contrast_questions run
```

评测使用官方 Docker 评测器。最新训练样本中的测试包含超过 4300 位的整数，宿主与容器 Python 需设置 `PYTHONINTMAXSTRDIGITS=0`；原工作区的 `runs/contrast_questions_train_v1/eval_bin/docker` 只为容器传入这个变量，未更改测试或评分器。

```bash
sudo -n -g docker env PATH="$PWD/runs/contrast_questions_train_v1/eval_bin:/usr/bin:/bin" PYTHONINTMAXSTRDIGITS=0 .venv/bin/python -m experiments.compare_contrast_questions evaluate_watch
```

这些是原工作区恢复说明，不是新克隆工作区的一键复现承诺。
