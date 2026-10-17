# 第三方来源与许可

本项目基于 [serval-uni-lu/clarification-competition](https://github.com/serval-uni-lu/clarification-competition)，工作区上游起点为 `1450e080eab7eeaba9874b65a2e3839f97390344`。上游 `LICENSE`、源码版权声明、公开数据与依赖许可保持原状。

`clarify/algorithms/zarutskiy_clarifier.py` 原样来自 Semyon Zarutskiy 的 MIT 许可算法，保留 SPDX 版权与许可头、作者和团队说明。对应来源提交为 `ae882179a9b33e9439be6fe3524a84d6c701606f`，文件 SHA256 为 `8295ce0d907664506b9e7795aadbc71b78586516f139de75f9b9aef2db065612`。

本地 CoT、蒸馏、提问策略及实验基础设施是在该 SDK 和方法之上进行的研究扩展。公开归档不意味着这些第三方算法由本项目原创，也不意味着主办方认可本地实验结果。

数据衍生自 HumanEval、MBPP 及上游发布的需求变体；使用和再分发时遵循各自来源的许可。Python 依赖的版本锁定于 `uv.lock`，其第三方许可由各依赖项目提供。
