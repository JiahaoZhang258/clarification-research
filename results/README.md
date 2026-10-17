# 精简实验结果

所有结果为本地实验，不是主办方确认成绩。结果快照日期：2026-10-17。

| 文件 | 内容 |
|---|---|
| `validation.json` | 8 个完整验证实验的通过率、TDS、nDCG、提问率、过度提问率和对应范围的平均费用 |
| `validation_outcomes.jsonl` | 770 条记录的匿名标识、各版本通过与否及提问质量标记，可重算指标；不含题目、代码或测试 |
| `distillation_stability.json` | 蒸馏四轮波动、净增益、按记录计算的探索性 McNemar 检验 |
| `training_distillation.json` | 训练集 97 条的提示词对照：65 对 65 |
| `training_variants.json` | 原训练开发集 100 条的策略比较 |
| `capacity_probe.json` | 50 条既有失败题各 5 次生成的能力探测 |
| `contrast_questions_partial.json` | 新提问策略的未完成实验；两组各三轮，总计仅完成 203/600 |
| `provenance.json` | 原始文件相对路径和 SHA256 校验值；原始文件保留在本地 |

## 指标与范围

- `pass_at_1`：一次最终解答通过测试的比例。
- `tds`：官方公式的轮数折扣成功率；本组完整验证记录均澄清一次。
- `ndcg`：官方提问质量分数；这里来源于模拟用户评定，不能替代代码测试。
- `clarification_rate`：至少问过一次的题目比例。
- `overasking_rate`：在官方标记不需要澄清的题目上仍提问的比例；训练集缺乏同样可靠的标签，因此训练报告为 null。
- `average_solver_cost_usd`：完整解题端调用费用，不含模拟用户端。
- `average_final_generation_cost_usd`：仅最后生成代码的费用，与完整解题端费用不可直接比较。

蒸馏的四轮共用原始 CoT 的历史问答与单次基线，不是四轮新提问流程。四轮平均通过率为 71.49%，范围 70.52%–72.08%；平均净增 3.5 条，但最后一轮比基线少 4 条，尚不能确认稳定提升。

McNemar 检验比较同一记录上两方法是否通过。多个记录可能来自相同原始题目，相关性会限制其统计解释；不要把四轮共 3080 条观测当作 3080 道独立题目。

最新提问实验按原始题目选择 100 个训练题目，每题保留一个变体，排除此前开发题目。其输出不完整且两组样本不齐，不能用当前通过率作有效策略排名。

```bash
python scripts/verify_research_results.py
```

该检查不调用模型。若需要从本地完整记录重新导出：

```bash
python scripts/export_research_results.py
```

导出只读取既有文件，不生成代码或调用 API；缺少本地原始记录时不能执行导出。
