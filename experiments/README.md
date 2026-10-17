# 实验索引

本目录保留实际开展过的实验代码。历史入口有冻结的输出目录、预算、源码哈希和旧问答依赖；它们是实验记录，不是可以在任意新工作区直接续跑的统一命令行工具。

## 入口对应关系

| 入口 | 实验 | 本地依赖 / 状态 |
|---|---|---|
| `compare_clarifiers.py` | 最早的例子确认与草稿假设审计，训练 40 条 | 支持独立准备新输出，详见 `--help`；[原始说明](../docs/research/initial_methods.md)保留当时设置 |
| `reproduce_zarutskiy.py` | 上游方法 770 条验证复现 | 依赖历史 `runs/methods34/` 账本和计费摘要 |
| `compare_question_selection.py` | 固定训练开发集 100 条，多种策略 | 依赖 `runs/devset100/manifest.json`，多个历史方法 |
| `reproduce_cot.py` | 当前文件实际运行 few-shot + CoT | 输出 `runs/cot_fewshot_val/`；名称沿用历史，不用于重建原始 CoT |
| `reproduce_patch.py` | 保留草稿再修改 | 验证集负结果，依赖共享账本 |
| `diagnose_cot_capacity.py` | 50 条失败题各采样 5 次 | 依赖原始 CoT 的完整结果与问答；6 条至少被救回一次 |
| `compare_prompt_distillation.py` | 蒸馏与原始 CoT，训练 97 条 | 固定旧问答，排除 3 个示范题目；两组各通过 65 条 |
| `compare_prompt_distillation_val.py` | 蒸馏完整验证，复用原始 CoT 基线 | 需要 `runs/cot_val/` 原始问答和结果 |
| `repeat_prompt_distillation_val.py` | 蒸馏追加 3 次验证复测 | 需要初测及冻结提示词，不重新生成原始 CoT |
| `compare_contrast_questions.py` | 新提问策略，两组各 3 轮完整流程 | 100 个原始训练题目；已完成 203/600，等待服务恢复 |
| `zenmux.py` | 本地接口路由和费用账本 | 保守预留费用；不属于参赛算法本身 |

`prompt_distillation/` 保存提示词与构造来源，`contrast_question_v1/` 保存从既有训练案例整理的提问规则。来源 JSON 中的初始状态字段是构造当时的记录，最新评估状态以 `results/` 为准。

## 冻结与复现

清单记录源码与数据哈希，续跑必须沿用相同输入。不要为了让旧入口在新克隆里通过而创建空账本、伪造旧结果或删除费用记录。

个人项目的两个明确算法入口是 `clarify/algorithms/original_cot_clarifier.py` 和 `clarify/algorithms/distilled_clarifier.py`。它们可通过 SDK 在新环境运行；历史精确重放另见 [复现说明](../reproducibility/README.md)。

公开汇总在 [results/](../results/README.md)，完整模型输出和服务账本只在原工作区 `runs/`。所有历史预算属于当时实验设置，不代表对读者的付费运行授权。
