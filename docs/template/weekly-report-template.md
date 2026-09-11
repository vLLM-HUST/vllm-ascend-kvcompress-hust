# 项目周报模板

## 输出格式

在docs/dev/reports/weekly目录下生成以年月日命名的周报

填写格式：
```md
# 项目周报
## 模型选择
xxxx
## 优化场景
xxxx、xxxx
## 研究方向
xxxx
## 研究成果
无
## 优化目标
xxxx、xxxx
## 推进情况
xxxx
## 补充说明
- xxx
- xxx
- xxx
## 附件（有则填，没有则不填）
<文件路径>
```

## 填报说明

标了“*”的为必填项。

### 模型选择 *

> 可多选（根据项目任务书交付及现阶段开发要求基于四类模型下开展工作）
>
> 可选项：Qwen2.5-14B、Qwen3.6-27B、DeepSeek V4 Flash、Kimi K2.6、其它工作项

## 优化场景（数据集）*

> 可多选，后续成员的优化应尽量落到明确的业务/测试场景中。现有44个数据集按典型场景归纳，便于快速选定方向。如果优化场景与目标内所列场景不一致，可以填你希望新增的场景（经评审小组讨论审核后方可实施）。44个数据集详细信息参见：vllm-hust-dev-hub/docs/
>
> 可选项：AI2-ARC、AIME-2024、BBH、BFCL-V3、BFCL-V4、BURSTGPT-TRACE、EVOSCIENTIST、GPQA、GSM8K、HOTPOT-QA、HOTPOTQA-REACT-100、HUMANEVAL、INFINITEBENCH、INSTRUCTCODER、JSONSCHEMABENCH、LIVECODEBENCH-CODE-GENERATION、LIVECODEBENCH-EXECUTION、LIVECODEBENCH-TEST-GENERATION、LONGBENCH、LONGBENCH-V2、MATH-500、MBPP、MMLU-PRO、OASST1、PAIO-CHAT-1000、PAIO-CODE-EVAL-1000、PAIO-JOSN-500、PAIO-LONG-PREFILL-5000、PAIO-LONGTEST-4000、PAIO-PREFIX-SHARED-5000、PAIO-REUSE-CONV-5000、SCHEMASTORE、SHAREGPT-V3、SONNET、STRUCTEVAL、SZYN-OPENCODE-SWEBENCH-VERIFIED-500、SYFI-CODING-TRACE、TAU2-BENCH、TOOLBENCH、ULTRACHAT-200k、VISIONARENE、WILDCHAT、WILDCHAT-4.8M、trie-workloads、vllm-hust-bidkv、通用场景、无

## 研究方向 *

> 请简要概括你的研究方向或要解决的具体问题是什么

## 研究成果 *

> 与本课题研究相关的学术成果（论文、期刊、会议等）
>
> 可选项：无、编撰中、已完成待发表、已发表

## 优化目标 *

> 可多选，也可新增自己新的优化点（但每个优化结果不能造成整体性能的负增益）
>
> 可选项：请求吞吐、输入吞吐、输出吞吐、总吞吐、TTFT、TPOT、端到端时延、成功率、正确率/任务得分、OOM、静默截断率、B1/B0 吞吐比、吞吐提升率、时延改善率、跨场景几何平均、极限百万token成本、SLO百万token成本、峰值填平率、报价（加价率）、SLO Goodput、Cache命中率、HBM峰值、NPU平均利用率、MFU、报价（会计毛利率）、三轮变异系数、平均功耗、每百万token能耗、Jain公平指数、无、调度模块、KV Cache传输与恢复生命周期可观测性

## 推进情况 *

> 可多选，本周工作进展梳理
>
> 可选项：构思阶段、调研阶段、复现阶段、验证阶段、交付阶段

## 补充说明 *

> 分点简要总结本周工作内容、成果、难点卡点、求助事项（工作总结将作为月度考评的重要依据）

## 附件

> 解决项目瓶颈、重大难题或取得突出贡献的请提交相应支撑材料供课题组审核并作为额外奖励的证明（需提交文件）
