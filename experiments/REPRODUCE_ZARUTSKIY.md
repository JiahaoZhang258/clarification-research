# 第一名方法复现

> 历史实验记录：下文的“第一名”、预算与榜单目标对应当时的复现设置，不代表当前排名或新运行预算。最新项目状态见根目录 README 和 results/；原账本及问答仅保存在本地 runs/。

复现对象为官方确认的 ZarutskiyClarifier，源提交
`ae882179a9b33e9439be6fe3524a84d6c701606f`。算法文件逐字复制上游，保留作者和 MIT 许可证。

- 源码：https://github.com/zarutskiysy/clarification-competition/blob/ae882179a9b33e9439be6fe3524a84d6c701606f/clarify/algorithms/zarutskiy_clarifier.py
- 官方确认：https://github.com/serval-uni-lu/clarification-competition/pull/24#issuecomment-6165020976
- SHA256：`8295ce0d907664506b9e7795aadbc71b78586516f139de75f9b9aef2db065612`

```text
公开题目 --> 初稿 --> 分析一个关键歧义 --> 一个澄清问题
                                              |
                                       官方模拟用户回答
                                              |
公开题目 -------------------------------------+
                                              v
                                    重新生成代码（无初稿）
                                              |
                                    代码清理 --> 官方评测
```

使用算法全部默认参数：总是提问、针对一个行为事实、以假设确认的形式提问、
最终只生成一份代码，并明确澄清回答优先于冲突的原文或示例。
多候选比较、探测输入、执行筛选和需求重写功能保持默认关闭。

## 固定实验设置

- 使用当前官方 SDK（`main@1450e08`）的完整 `val` 划分：770 条，HumanEval 350 条、MBPP 420 条。
- 保留重复记录，共 768 个不同题目/描述对；每条记录都有独立标识，每条只生成一次。
- 按题号、描述和完整内容哈希排序，再交错两个数据集，以稳定续跑顺序。
- 官方 `data/clarifications.jsonl` 以官方加载方式补充到模拟用户环境。
- 算法只收到公开描述、函数名，以及通过环境获得的澄清回答。
- 解题和模拟用户模型均为 `openai/gpt-4.1-mini`；温度分别为 0.7 和 0。
- 每题最多澄清一次，官方解题/澄清单题预算均为 1 美元。
- 本地通过 LiteLLM 调用 ZenMux；每次最多输出 8192 token，120 秒超时，网络自动重试关闭。
- 正式批量运行使用 4 个任务线程（前 5 条检查使用单线程），各自隔离上下文；费用预留通过锁串行写入账本。无多代理参与。
- 官方 Docker 评测器与官方指标计算函数负责通过率、TDS、nDCG 和过度提问率。

## 累计预算与续跑

累计上限为 **2 美元**，包含此前方法 3/4 实验和本次全部解题、模拟用户调用。
首次运行把旧 `runs/methods34/usage.jsonl` 原样复制为
`runs/zenmux_cumulative_usage.jsonl`，保留此前保守估计费用 0.097556 美元。
此后所有新调用追加到这个共享账本，并且续跑要求旧账本内容未变。
请求发出前预留费用；未知失败保留预留；余额不足时停止新请求，保留已经完成的结果。

```bash
.venv/bin/python -m experiments.reproduce_zarutskiy prepare
.venv/bin/python -m experiments.reproduce_zarutskiy run --limit 5 --workers 1
.venv/bin/python -m experiments.reproduce_zarutskiy evaluate
.venv/bin/python -m experiments.reproduce_zarutskiy run --limit 770 --workers 4
.venv/bin/python -m experiments.reproduce_zarutskiy evaluate
.venv/bin/python -m experiments.reproduce_zarutskiy report
```

同一个 `runs/zarutskiy_val_ae88217/` 目录保存清单、原始源码、调用过程、生成代码、
评测结果和汇总。续跑跳过已完成的实例，保留重复样本各自的结果。
中断任务从头重试时，其之前调用的费用仍保留在累计账本；失败尝试单独存档。

## 结果解释

官方目标：通过率 69.7403%、TDS 0.6696826、nDCG 0.9259740、提问率 100%。
TDS 为轮数折扣成功率，正确代码才计分，额外提问会折扣；nDCG 为官方提问质量指标。
代码生成的随机性、ZenMux 路由和本地输出上限可能使本次分数不同。

榜单的平均费用只取解题端 `prompt_cost`；本次总预算还包括模拟用户调用。
报告分别列出榜单口径的平均解题费用、全部调用实际报告费用和保守预算估计。
本复现不修改或发布官方榜单。
