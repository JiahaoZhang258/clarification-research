# Clarification Research：基于 GPT-4.1-mini 的澄清与代码生成实验

本仓库是个人研究工作区，基于 [Clarification Competition SDK](https://github.com/serval-uni-lu/clarification-competition) 开发。研究目标是在需求不完整、存在歧义或矛盾时，通过一次澄清提高 Python 函数生成的成功率。

这是个人项目归档，不是向比赛官方提交的参赛 PR。以下成绩来自本地实验，未经主办方确认。上游 SDK 的原始说明保存在 [UPSTREAM_README.md](UPSTREAM_README.md)，上游许可证与作者署名保留。

## 当前结果

模型为 `openai/gpt-4.1-mini`，完整验证集包含 770 条记录。

| 方法 / 轮次 | 通过数 | Pass@1 | TDS |
|---|---:|---:|---:|
| 上游 Zarutskiy 方法的本地复现 | 534/770 | 69.35% | 0.66594 |
| 原始 CoT | 547/770 | 71.04% | 0.68215 |
| few-shot + CoT | 543/770 | 70.52% | 0.67717 |
| 保留草稿后修正 | 513/770 | 66.62% | 0.63975 |
| 蒸馏提示词：初测 | 553/770 | 71.82% | 0.68964 |
| 蒸馏提示词：复测 1 | **555/770** | **72.08%** | **0.69213** |
| 蒸馏提示词：复测 2 | 551/770 | 71.56% | 0.68714 |
| 蒸馏提示词：复测 3 | 543/770 | 70.52% | 0.67717 |

- **Pass@1**：每条记录一次最终解答通过测试的比例。
- **TDS**：按澄清轮数折扣的成功分数。
- **nDCG**：官方用于衡量提问质量的指标；本地模拟评分不保证代码正确。
- **CoT**：在代码前给出需求分析和推理步骤。本项目的“蒸馏”指提示词与范例整理，没有训练模型权重。

蒸馏版四轮平均通过率为 **71.49%**，平均 TDS 为 **0.68652**。最高单次分数是 72.08%，尚未证明存在稳定提升。四轮都复用原始 CoT 的历史澄清问答，因此检验的是最终生成阶段；原始 CoT 的随机波动没有重复测量。

当前新增的“用具体例子区分两个需求解释”策略，在训练集进行两组、各三轮的完整流程比较。已完成 **203/600** 次生成和评测，因模型服务访问受限未完成；各组样本不齐，暂无效果结论。

完整指标、精简逐条判定和实验限制见 [results/README.md](results/README.md)。

## 项目组织

```text
公开题目
   |
   v
clarify/algorithms/       初稿、一个澄清问题、整合回答、生成代码
   |         ^
   v         |
clarify/env.py           官方模型与模拟用户交互接口
   |
   v
clarify/runtime.py       Docker 执行测试
   |
   v
experiments/             历史实验编排、预算账本、续跑
   |
   +--> runs/            本地完整记录，不上传
   +--> results/         可公开的精简汇总和逐条判定
```

| 目录 / 文件 | 作用 |
|---|---|
| `clarify/`、`data/` | 保留上游 SDK 与公开数据；新增算法位于 `clarify/algorithms/` |
| `experiments/` | 历史实验入口、冻结规则及提示词来源 |
| `reproducibility/` | 可公开的冻结提示词、实验清单和复现说明 |
| `results/` | 精简实验结果，不含完整模型输出或隐藏测试 |
| `tests_local/` | 本地离线测试 |
| `scripts/` | 上游验证脚本，以及结果导出、检查、打包工具 |
| `docs/` | 保留的上游网站快照；其中榜单不是本项目新增成绩 |
| `HANDOFF.md` | 当前项目状态与继续工作的入口 |

## 算法入口

| 文件 | 含义 |
|---|---|
| `original_cot_clarifier.py` | 新增明确入口：冻结的原始 CoT，对应历史 547/770 |
| `distilled_clarifier.py` | 新增明确入口：冻结的蒸馏 v1 提示词，对应四轮验证实验 |
| `cot_clarifier.py` | 历史文件，实际内容为 **few-shot + CoT**，对应 543/770；保留原名以免破坏历史引用 |
| `zarutskiy_clarifier.py` | MIT 许可的上游方法原样复现 |
| `contrast_question_clarifier.py` | 新提问策略及其冻结对照类；包含两个类，使用专用实验入口，不直接交给 SDK 单类加载器 |
| 其他算法文件 | 探索性方法与负结果，见 [实验索引](experiments/README.md) |

两个新增算法入口只提供明确的加载路径，并嵌入或复用冻结提示词；没有为这些新文件重新开展付费评测。表中分数属于历史运行，不能保证重新运行得到相同结果。

## 安装与离线检查

需要 Python 3.13+ 和 uv；执行代码测试需要 Docker 及 `ganler/evalplus` 镜像。

```bash
uv sync --frozen
uv run python -m unittest discover -s tests_local -v
uv run python scripts/verify_research_results.py
```

上述测试和结果检查不调用付费模型接口，也不需要本地 `runs/`。

实际运行算法、选择接口和重放历史实验的区别见 [复现说明](reproducibility/README.md)。API 密钥在本地环境变量或 `.env` 中配置，模板见 [.env.example](.env.example)。

## 上传范围

上传源码、公开数据、依赖锁文件、复现说明和精简结果。`.env`、`.venv/`、`runs/`、缓存、完整调用日志与费用账本留在本地。

```bash
python scripts/package_research.py --output /path/to/clarification-research.zip
```

打包工具仅生成文件归档，不创建 Git 提交、不修改远程仓库、不上传，也不向官方提交 PR。原始数据和第三方代码的归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

个人上传包默认排除上游比赛的 `.github/workflows/pr_validation.yml`；研究检查仍可使用上面的离线命令运行。
