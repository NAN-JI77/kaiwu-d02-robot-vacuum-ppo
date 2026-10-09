# Reward设计与边界

实际源码见 [Preprocessor.reward_process](../agent_ppo/feature/preprocessor.py)、[Config reward shaping](../agent_ppo/conf/conf.py)、[EpisodeRunner.run_episodes](../agent_ppo/workflow/train_workflow.py)。下表描述当前保存代码，不套用到39171历史窗口。`env_reward`、Agent shaping reward和平台total_score是三个不同量。证据边界见 [evidence.md](evidence.md)。

## 当前构成

| 类别 | 主要条件和系数 | 意图/限制 |
|---|---|---|
| 清扫 | 0.16×新增清扫格；有清扫+0.045；每步-0.002；survival0 | 避免仅活满步奖励 |
| 新格/污渍 | 首次访问+0.008；接近污渍+0.018；离开-0.015 | 探索只是辅助，不等于有效清扫 |
| 重访 | visit>2且未补电，-0.024×min(visit-1,3) | 路线必要重用需门控 |
| 循环/停滞 | 最近8步≤3唯一格-0.05；无清扫≥35步每级-0.014，至4级 | 区分短循环和清扫停滞 |
| 低电 | ratio<0.25为-0.035；<0.12再-0.12 | 不直接决定回充动作 |
| 电量risk | ratio<0.45且未补电；接近+0.006、后退-0.008；远路/critical按压力加权 | 优先有效BFS距离，未知需fallback |
| 预回充 | 0.30≤ratio<0.55、未补电、桩距≥10；接近+0.003，后退/停滞-0.002 | 有门控的shaping，不证明保持最优策略不变 |
| 短循环细分 | ABAB-0.006、三角-0.005、recent-0.003；低电×2、forcecharge×2.5 | 多项可叠加 |
| 清扫效率 | 非回充/补电且电量≥0.30；按新格、重复密度、stall等加减；密度-0.022/-0.044等 | 参数表不替代完整条件 |
| 回充路线 | 回充未补电上下文；repeat-0.018、no-progress-0.016、progress+0.005 | 不把所有返航旧路都当坏重复 |
| 补电 | prev_ratio≤0.36时 min(0.08,0.01+0.001×delta_battery)；否则unneeded-0.04 | 保存的环境说明描述即时补电；实际runtime版本未恢复，不能宣传缩短物理充电时长 |
| 阻塞/NPC | blocked-0.04、stuck-0.06、wall-near-0.01；NPC距离/接近/清扫风险门控 | 不保证完全安全 |
| 满步终局 | 5×clean_ratio + 9×clip((score/steps-0.95)/0.95,-1,1.5)；另有低清扫/repeat/charge overhead/高余电项 | completed不等于高分 |
| 提前结束 | -2；电量≤0.02另-4；另有低分/回充路线项 | 对应workflow，不是官方总分 |

## 不应夸大的部分

代码中存在旧电量risk的 `if False` 分支和多个feature flag。系数存在不表示项总是启用。reward_process还接收forcecharge、clean-repeat、route-repeat、post-charge等Agent上下文；最小合成预处理只检验基础调用，不能证明全部门控。

“预回充势能”是工程命名，没有证明满足potential-based shaping定理。平台charge_count字段定义仍待确认。历史日志记载的短充/驻留假设与官方缓存即时补满机制不同，需单独核对。

reward源码和参数按字节保留，没有通过新训练证明设计提高成绩。相同条件消融、reward贡献分解和端到端轨迹验证为NOT_RUN；本次整理不训练、不加载模型。

