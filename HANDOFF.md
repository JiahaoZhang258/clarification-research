# 项目交接

更新日期：2026-10-17。

## 当前结论

- 模型：`openai/gpt-4.1-mini`；本地通过 ZenMux 路由。
- 原始 CoT：完整验证集 547/770，通过率 71.04%，TDS 0.682153。
- 蒸馏 v1 四轮：553、555、551、543；最高 72.08%，平均 71.49%。尚未确认稳定提升。
- 新提问策略：训练集 100 个独立原始题目、两组各三轮，目标 600 次；已生成并评测 203 次，服务访问受限而停止。样本不齐，不能判断收益。
- 没有正在运行的实验。没有向比赛官方提交 PR。

## 版本与文件

`clarify/algorithms/cot_clarifier.py` 是历史 few-shot + CoT 版本，不是 547/770 的原始版本。
明确入口分别是 `original_cot_clarifier.py` 和 `distilled_clarifier.py`；冻结提示词位于 `reproducibility/frozen_prompts/`。

公开结果和指标口径见 `results/README.md`；复现限制见 `reproducibility/README.md`；实验入口索引见 `experiments/README.md`。

## 本地续跑

完整输出、成功调用缓存和累计账本保留在被忽略的 `runs/`，不进入上传包。
最近实验的状态是 `runs/contrast_questions_train_v1/blocked_status.json`。
当前批准的累计预算上限为 $14，保守账本约 $12.36，含失败请求的未知费用预留；这不是实际账单。

继续前核实模型账户访问状态，并通过受账本约束的最小调用检查连通性。不得把余额不足当成临时故障批量重试。
成功调用缓存在 `call_cache.jsonl`；应复用缓存续跑，不重新生成已有结果，也不清空费用账本。

历史实验的源码和清单已冻结。不要随意修改实验算法或入口，否则清单哈希校验会拒绝续跑。
评测需要 Docker 权限；原工作区使用 `sudo -n -g docker` 运行评测。最近训练实验的宿主和 Docker Python 均设置 `PYTHONINTMAXSTRDIGITS=0`，以处理测试中的超大整数字面量；细节在本地 `evaluation_environment.json`。

## 工作约定

使用中文；解释专业术语。影响代码、数据选择、预算或评估口径的改动，先说明方案并取得用户确认。
不打印密钥。比较要报告重复运行的波动，不把相同题目的多轮结果当成独立样本，不用未完成实验下结论。

个人 GitHub 整理范围为源码、复现说明和精简结果；实际推送目标需使用用户自己的仓库地址。当前 `origin` 仍为上游比赛仓库。
