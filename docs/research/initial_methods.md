# 方法 3 与方法 4：首轮实验

两套独立算法：`ExampleConfirmation` 用小例子确认一个行为规则；
`AssumptionAudit` 从代码草稿中检查一个没有需求依据的行为假设。
两者均只通过官方 `env.llm()`、`env.can_ask()`、`env.ask_human()` 交互。
没有访问参赛者方法，也没有在算法中读取标准答案或隐藏测试。

本地入口在官方环境之外，将 LiteLLM 请求转发到 ZenMux。
ZenMux 模型 ID 是 `openai/gpt-4.1-mini`，LiteLLM 的兼容接口路由字符串是
`openai/openai/gpt-4.1-mini`：最前面的 `openai/` 仅用于选择协议。
解题与模拟用户使用同一模型，温度分别为 0.7 和 0。

## 固定设置

- 从官方训练集按 `(task_id, prompt)` 去除完全重复项，然后每个数据集随机抽取 20 题。
- 随机种子 42；两个数据集交错排列，前 5 题也覆盖两个数据集。
- 两种方法使用完全相同的 40 条题目，每题每方法运行一次；不训练模型。
- 最多 1 轮用户澄清；问题必须只针对一个事实。方法 3 最多 3 次解题模型调用，方法 4 最多 3 次。
- 官方单题费用预算保持各 1 美元；本次所有调用合计上限为用户确认的 2 美元。
- 每次调用最多输出 8192 token，超时 120 秒；自动重试关闭，避免不明确的重复扣费。
- 运行顺序在题目间交替；已完成结果可续跑复用。
- 执行结果由官方 Docker 评测器判定；TDS/nDCG 使用官方计算函数。

Key 通过 `ZENMUX_API_KEY` 或 Git 忽略的 `.env` 文件读取，不进入结果日志。
`runs/` 中的题目清单、输出和费用记录均为本地产物，不随代码提交。

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s tests_local -v
uv run --frozen python -m experiments.compare_clarifiers prepare
uv run --frozen python -m experiments.compare_clarifiers run --budget 2 --limit 5
uv run --frozen python -m experiments.compare_clarifiers evaluate
uv run --frozen python -m experiments.compare_clarifiers run --budget 2 --limit 40
uv run --frozen python -m experiments.compare_clarifiers evaluate
```

费用采用 ZenMux 官方公开价格和返回的 token 用量估计，不扣除缓存优惠，
不是最终账单。每次请求发出前写入保守费用预留；出错或缺少用量时保留预留并停止，
不会将未知费用记为零。续跑会读取同一费用账本，继续受累计 2 美元上限约束。

结果是训练子集上的探索性比较，不能作为官方验证集成绩，也不能据此确定最终优胜方法。
参赛署名与许可证尚未填写；提交前由用户确定，只提交最终选定的单文件算法。
