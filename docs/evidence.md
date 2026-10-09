# 公开证据说明

本页只保留公开树中可以直接复核的来源。`VERIFIED` 表示代码、结构化日志或静态检查支持该事实；不表示因果收益、最终提交或法律授权。`USER_CONFIRMED` 表示所有者明确确认；`UNVERIFIED` 表示历史记录缺少闭环；`UNKNOWN` 表示没有证据；`REVIEW_REQUIRED` 表示仍需发布前确认。

## 公开源码证据

| ID | 文件 / 函数 | 支持内容 |
|---|---|---|
| S01 | [model.py](../agent_ppo/model/model.py) / `Model.forward` | 73D base、局部/全局 CNN、scalar、delta heads、phase value gate |
| S02 | [conf.py](../agent_ppo/conf/conf.py) / `Config` | 4277D、8 动作、PPO 参数、规则开关 |
| S03 | [algorithm.py](../agent_ppo/algorithm/algorithm.py) / `learn/_compute_loss` | masked PPO、value clipping、entropy、一次 optimizer step |
| S04 | [definition.py](../agent_ppo/feature/definition.py) / `_calc_gae` | episode 逆序 GAE；不读取 done |
| S05 | [preprocessor.py](../agent_ppo/feature/preprocessor.py) / `reward_process` | reward 门控和环境字段消费 |
| S06 | [agent.py](../agent_ppo/agent.py) / `observation_process/predict/exploit` | 地图记忆、采样、规则、贪心动作和回充状态 |
| S07 | [train_workflow.py](../agent_ppo/workflow/train_workflow.py) / `workflow` | step、终局 shaping、GAE、平台样本发送 |
| S08 | [环境配置](../agent_ppo/conf/train_env_conf.toml)、[平台配置](../conf/configure_app.toml) | 当前保存环境和预加载设置；不证明最终提交 |

## 实验数据

`experiments/episodes.jsonl` 保留 208 条 `[GAMEOVER]` 数值记录，`experiments/learner_metrics.jsonl` 保留 79 条 learner 指标。每行含窗口 ID、来源类别和原脱敏日志行号；`experiments/source_fingerprints.json` 记录归档前脱敏源文件指纹。原始日志、账号、进程、IP 和内部历史不在公开树。

四个窗口都是 continuing-training window，不是冻结模型评估：W10/W11/W12 从 39171 开始，使用 battery 200、4 chargers；W27 从 73189 开始，使用 battery 150、3 chargers。W27 的 25/33 局为电量相关失败。不同环境、规则、奖励和持续更新不允许严格 A/B 或 checkpoint 排名；最终提交模型和官方成绩 UNKNOWN。

## 官方环境资料

[13.0.1环境介绍](https://tencentarena.com/docs/p-competition-robot_vacuum/13.0.1/guidebook/dev-guide/intro)是保存环境说明的官方定位。历史核对使用所有者保存的页面缓存：每步耗电、桩3×3邻域即时补满、NPC邻域碰撞等机制。本次未恢复实际runtime或重新核验线上正文，缓存语义不自动证明所有历史run使用同一环境版本，也不是个人算法贡献。

## 授权与版本边界

所有权声明见 [NOTICE.md](../NOTICE.md)。Tencent 版权头保持原样。原始工程没有 Git 仓库、提交历史或 LFS 元数据可供公开候选继承。2026-10-09在独立候选目录初始化本地Git，用于真实源码的首次本地提交；没有导入原始历史、设置远程或发布GitHub仓库。公开源码清单和 SHA-256 位于 `configs/source_manifest.json`。
