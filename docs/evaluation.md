# 历史指标与统计口径

已恢复的是持续训练日志，冻结模型的官方评估记录为 0 条已恢复；这不表示比赛期间从未评估。最终提交模型、官方成绩和排名 UNKNOWN。

## 可重算证据

- [episodes.jsonl](../experiments/episodes.jsonl)：208 条本地单个 aisrv 日志的 GAMEOVER 数值记录。
- [learner_metrics.jsonl](../experiments/learner_metrics.jsonl)：79 条 learner loss/KL/clip 等打印记录。
- [window_metrics.json](../experiments/window_metrics.json)：四窗口统计，每个指标保留样本数 n。
- [source_fingerprints.json](../experiments/source_fingerprints.json)：转换前脱敏来源的 SHA-256，不代表官方签名。

公开行保留 `window_id`、`source` 别名和归档源行号，排除原始消息、进程、容器、IP、账号以及非数值调试字段。原始日志本地归档，公开数据足以复算表中描述统计，但不是完整比赛日志分发。

```powershell
python -B scripts/summarize_experiments.py --check
```

工具使用 JSON 解析数值记录，不导入 PPO 或环境。各窗口配置由历史文件/备份交叉核对，部分字段并非每个 episode 都有；公开数值本身不能独立证明 seed、worker 或代码冻结。指标打印精度限制保留，不补造置信区间。

## 训练窗口

| ID / 时间 | 初始化 | battery / charger / max / NPC | 局数 | score mean | WIN / BAT / BAT_LOW | KL / clip_frac |
|---|---|---|---:|---:|---|---|
| W10 / 4-19 10:37–10:57 | 39171 | 200 / 4 / 1000 / 4 | 53 | 755.830 | 16 / 4 / 0 | .001908 / .135895 |
| W11 / 4-19 11:51–12:04 | 39171 | 200 / 4 / 1000 / 4 | 40 | 701.300 | 11 / 4 / 3 | .001605 / .113942 |
| W12 / 4-19 12:10–12:40 | 39171 | 200 / 4 / 1000 / 4 | 82 | 772.659 | 26 / 2 / 2 | .001953 / .136773 |
| W27 / 4-27 15:34–15:56 | 73189 | 150 / 3 / 1000 / 4 | 33 | 354.939 | 0 / 18 / 7 | .002183 / .166944 |

年份为 2026。各窗口 10 图随机，seed/worker UNKNOWN。初始化 ID 只标示开始来源；参数随后持续更新。WIN/BAT/BAT_LOW 是 workflow 自定义结果标签，BAT_LOW 为 `FAIL_BATTERY_LOW_CLEAN`。W12 旧摘要的 BAT4 曾合并两类，现在拆为 2+2。score、训练 reward 和清扫率不能混作一个指标。

W27 共 25/33 电量相关失败、另 8 局低清扫标签。其均分不能与 200 电/4 桩窗口作为公平算法对比。网络、奖励、规则、资源及持续更新同时变化，不作 checkpoint 排名、因果提升或消融结论。

## 入口限制

原配置 eval/exam 指向外部 `tools.eval.workflow.*`，源码保留但所需服务/模块未提供。`Agent.exploit` 返回经 guide/规则处理的贪心动作，不能描述为纯网络 argmax。本地没有执行官方环境评估、checkpoint 推理或新的训练。
