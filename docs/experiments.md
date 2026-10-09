# 实验分析

本页整理已经存在的实验记录，不新增训练或评估。每条结论都区分实现是否存在与收益是否被同条件证实。

## 可复核训练窗口

| 窗口 | 初始化 | 环境 | 局数 | 平均 score | 观察 |
|---|---|---|---:|---:|---|
| W10 | 39171 | battery 200 / chargers 4 / max 1000 / NPC 4 | 53 | 755.830 | 16 个自定义 WIN，4 个电量失败 |
| W11 | 39171 | battery 200 / chargers 4 / max 1000 / NPC 4 | 40 | 701.300 | 11 个自定义 WIN，4 个 BAT、3 个 BAT_LOW |
| W12 | 39171 | battery 200 / chargers 4 / max 1000 / NPC 4 | 82 | 772.659 | 26 个自定义 WIN，BAT/BAT_LOW 各 2 |
| W27 | 73189 | battery 150 / chargers 3 / max 1000 / NPC 4 | 33 | 354.939 | 18 个 BAT、7 个 BAT_LOW，未出现自定义 WIN |

地图均为 1-10 随机；seed 和 worker 数未知。`WIN`、`FAIL_BATTERY`、`FAIL_BATTERY_LOW_CLEAN` 是 workflow 标签，不等于官方 completed/abnormal/timeout 或比赛胜利。W27 的 25/33 电量相关失败应作为失败诊断展示。

## 17 组历史尝试

历史版本过程主要来自本地归档的实验笔记和备份，当前代码可以核对保留的机制；存在机制不证明历史修改产生收益。表中“历史摘要”不作为已验证实验数据。USER_CONFIRMED 只用于笔记中明确记载的用户反馈。归档行号帮助所有者复核，公开读者无法仅靠本仓库验证这些未公开历史原件。

| ID / 问题与假设 | 修改 / 配置 | 方法与实际结果 | 结论 / 证据 |
|---|---|---|---|
| E01 非法概率、更新过大；降低更新幅度可能稳定 | normalize；lr8e-5→3e-5、clip.15→.10、beta.015→.008；早期300电/4桩，地图/步数亦变 | 历史两小时高KL摘要未恢复raw；当前参数见 [Algorithm](../agent_ppo/algorithm/algorithm.py) | 实现VERIFIED；得分收益UNVERIFIED；归档实验笔记段1520 |
| E02 满步但低清扫；survival不能代替效率 | 清扫/score-step shaping、survival0、减少纯探索；35021/300电→39171/200电 | clean/step和repeat为目标，缺同配置前后评估 | 实现VERIFIED；收益UNVERIFIED；段1570/1668/1689；[reward](reward.md) |
| E03 地图轴/误阻塞；坐标与短时障碍可能导致假卡住 | `[z][x]`、corner/TTL/static fallback、回充暂停wall-stuck；35021→39171，200/4/1000 | 用户历史反馈改善；W10-12内部blocked/stuck0 | 反馈USER_CONFIRMED、窗口VERIFIED、因果UNVERIFIED；段1308/1392/1652/1787；[episodes](../experiments/episodes.jsonl) |
| E04 charge_count高且余电高；可能在桩邻域重复 | 桩目标、zone no-gain escape、动态exit；35021/300/4 | 短充/慢充假设与保存的即时补电说明不一致 | 实现VERIFIED；充电时长解释UNVERIFIED；段1595/1612/1736/1763 |
| E05 远路没电；更早回充或可安全 | reserve/cap/route强化后回退；39171/200/4/1000 | 历史摘要142局约471分、95电量失败、charge.216/override.194，缺原日志 | 所有历史数字UNVERIFIED；失败假设有分析价值，不能画已验证性能图；段1860/1888 |
| E06 未知路径99/局部循环；需要可达性fallback | known-map BFS、charger-frontier/open-frontier；39171/200/4 | 历史34局690.5、55局731.9仅摘要 | 当前规则VERIFIED；数字/收益UNVERIFIED；段1937/2011/2033/2055 |
| E07 clean重复、硬接管过强；软偏置可能减少打断 | clean_seen/repeat pressure、known-dirt/frontier soft bias、progress cooldown；39171/200/4 | W10-12均分755.830/701.300/772.659，repeat亦起伏 | 窗口VERIFIED，因果UNVERIFIED；归档段1243；不是成功消融 |
| E08 NPC避让收窄后早退；需兼顾直接路线 | 历史distance1→2→3、恢复direct BFS-first；39171/200/4 | 非满步4/53→15/40→6/82，多项同时改变 | 记录VERIFIED、因果UNVERIFIED；段1197-1241；不把内部stuck0等同无碰撞 |
| E09 规则记忆网络不可见；增加状态表达 | 73D legacy + local/scalar/global CNN delta；39171→45951→47571，200/4 | 当前网络可读；历史不同key/shape的模块检查仅支持结构演进 | 架构VERIFIED，收益UNVERIFIED；段732/770/813；[Model](../agent_ppo/model/model.py) |
| E10 补电后沿旧路；强逃离也可能回归 | post-charge escape/route memory反复clamp/disable/restore；45951/47571、200/4 | 历史repeat.427/.395、route.602仅摘要 | 当前机制VERIFIED；数字/收益UNVERIFIED；段621/653/702；保留回退 |
| E11 reward误罚正常前进；收窄门控 | stale-repeat/charge-progress/terminal-overhead；47571/200/4/1000 | 历史30分钟回归记录，无单因素对照 | 实现VERIFIED，结果UNVERIFIED；段565/594；[reward](reward.md) |
| E12 轨迹/clean-map/phase value表达 | local12/global14/scalar32、three value deltas；47571→49245→54713/65675 | 当前678753参数结构与gate见源码；名称没有监督语义 | 结构VERIFIED，收益UNVERIFIED；段536；[ppo](ppo.md) |
| E13 中距/近桩仍没电；reserve与capture不足 | reserve、final capture、frontier绕行与回退；65675历史，150/3反馈 | 历史12局7BAT等仅摘要 | 实现VERIFIED、数字/效果UNVERIFIED；段31/57/81；低电探索不保证安全 |
| E14 10%guided behavior；改善采样学习一致性？ | True/.10后回退False/0；65675/200/4；当前False/.10+adaptive | 历史KL.149/clip.329仅摘要；当前old_prob仍存raw PPO | 历史数字UNVERIFIED、当前开关VERIFIED；段5；[hybrid](hybrid_decision.md) |
| E15 电量距离日志不足；先提高可观察性 | safe-distance链、charge_count/余电诊断、panel短名；65675→69680/73189 | 修复诊断字段，不等于训练收益 | 实现VERIFIED，效果UNVERIFIED；段2109/2154/2196 |
| E16 后期扁平MLP与CNN恢复；选型依据缺失 | 71815/73801/76279备份为4277→128→64；73189等为CNN家族 | 私有备份/历史模块检查支持结构差异，未公开权重 | 结构历史证据VERIFIED，选型动机/收益UNKNOWN；当前仅保留真实CNN源码 |
| E17 开局未补电；opening guard或可减少早死 | v24 thresholds/reserve；73189/150/3/1000/10图 | W27均354.939、25电量失败、WIN0 | 配置/窗口VERIFIED、因果UNVERIFIED；不能宣称已解决；[W27](../experiments/window_metrics.json) |

公开 JSONL 是四窗口可重算的指标源；运行：

```powershell
python -B scripts/summarize_experiments.py --check
```

## 不能做的比较

39171 和 73189 的窗口使用不同电量、桩数、代码/规则阶段和持续训练状态；不能直接比较 checkpoint 优劣、算法提升百分比或消融效果。最终提交模型、官方成绩和排名 UNKNOWN。
