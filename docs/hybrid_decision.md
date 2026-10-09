# 混合决策：网络、规则、环境

实际实现见 [Agent](../agent_ppo/agent.py) 的 `predict/_anti_repeat_prob_adjust/_progress_score_soft_prob/_rule_adjust_action/_update_force_charge_mode`、[Preprocessor](../agent_ppo/feature/preprocessor.py)、[Config](../agent_ppo/conf/conf.py)。完整Agent需要官方SDK，历史隔离采样检查不覆盖全部规则门控。证据边界见 [evidence.md](evidence.md)。

## 能力归属

| 层 | 实现的作用 | 不能据此宣称 |
|---|---|---|
| PPO网络 | 输出8 logits/value，用reward与GAE学习状态-动作偏好 | PPO独立学会全部BFS/充电/NPC能力 |
| 软引导 | 重访衰减、新格/frontier/known-dirt/open-area加权、score混合、NPC/路线热度 | 原始网络输出就是最终行为分布 |
| 硬规则 | 非法动作修正、回充/离桩、卡住/循环、progress rescue、NPC避让 | 改善来自网络而非规则 |
| 环境 | 路过污渍清扫、耗电、进桩补电、碰撞检测、终止/评分 | 机器人自主实现充电物理过程 |

## 实际顺序

1. Model输出 → legal masked PPO分布。
2. 根据压力计算guide blend；guided分布由反重复等规则生成，与PPO混合。
3. progress score分布可进一步soft blend。
4. 随机采样action及贪心d_action；二者均经rule_adjust_action。workflow训练执行随机action，exploit使用d_action。
5. old_prob根据 `USE_GUIDED_BEHAVIOR_PROB` 选behavior或PPO分布。当前False不等于关闭引导；base guide为0.10、自适应pressure0.22/hard0.30，还有score bias。

## 触发和风险

| 问题 | 当前机制/典型触发 | 风险与证据边界 |
|---|---|---|
| 清扫探索 | 可见/已知dirt和frontier BFS第一步做软偏置；累积已知格、visit、trajectory | 这是手工搜索辅助；缺独立收益消融 |
| 重复访问 | visit/近期循环reward和概率衰减；严重clean-repeat压力触发progress rescue | 强压旧路可能破坏必要返航 |
| progress rescue | stall≥22或loop/clean-escape/revisit压力；cooldown12、loop_min_step80等门控；未forcecharge/未escape并满足评分条件 | 多条件分支，不等于所有stall都强接管 |
| 普通回充 | simple mode；base进入0.25、退出0.36；path≥10 reserve37/cap0.56；未知fallback0.46 | 不同距离阈值可高于退出，滞回/多次状态更新待测 |
| 开局未补电保护 | step≤240且未回血；远path≥42、中path≥28或未知且Manhattan≥24；阈值0.68/0.60/0.66、cap0.70 | 仍受reenter cooldown110、电量≤0.30等分支影响；未证明解决早死 |
| 回充导航 | charger-near/known-map BFS → frontier/open-frontier →局部fallback，附NPC风险 | 未知距离99不代表真实99步；不能盲信Manhattan可达 |
| 离桩 | 补电/邻域阶段状态、escape和post-charge路线记忆 | 充满仍绕桩、激进离桩都可能回归 |
| 防碰撞 | 平台mask、本地墙/斜角/blocked目标、NPC静态/运动预测/最终avoid | 预测有误差，仍可能碰撞 |
| 卡住 | 位移不变、TTL/目标格记忆、local escape、recent-loop rescue；充电期暂停部分wall-stuck | internal blocked=0不证明环境没有无效动作 |

原环境缓存写桩3×3即时充满、NPC3×3碰撞失败（当前线上和实际版本待核验），所以“短充优化”不能按缓慢充电解释。raw charger方向fallback只有上一步确实补电才可能被允许，其他情况强调passable逃离。

## 已做过的修改和结论

历史先尝试强制导航/积极回充，再收窄progress门控、使用soft bias、修地图轴/blocked判断、给未知路径frontier fallback、区分clean-repeat与charge-route-repeat，并多次回退post-charge/guide配置。E03用户历史反馈轴修复改善为USER_CONFIRMED；W10-W12的内部blocked/stuck为0是VERIFIED；其因果没有同条件闭环。来源见 [实验台账](experiments.md)和 [结构化记录](../experiments/episodes.jsonl)。

E14的10%guided behavior尝试后记录高KL/clip并回退，数字仅有归档历史摘要（UNVERIFIED）。当前False/.10+adaptive与历史False/0不同，不能写成问题已解决。W27低资源窗口25/33电量相关失败支持保留失败诊断，不能包装电量管理成功。

rule_override_rate只记硬动作变更，不覆盖soft guidance全部影响。高接管率可能遮蔽网络学习，低接管率也不证明是纯PPO。d_action分支只恢复部分状态的风险没有完整轨迹测试；本次不改策略，不运行Agent。

