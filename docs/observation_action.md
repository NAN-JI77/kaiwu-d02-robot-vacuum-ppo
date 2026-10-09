# Observation / Action

实现位于 [Preprocessor](../agent_ppo/feature/preprocessor.py) 的 `pb2struct/build_base_feature` 与 [Agent](../agent_ppo/agent.py) 的 `_augmented_local_memory_feature/_augmented_global_memory_feature/_augmented_scalar_memory_feature/_augment_observation_feature`。完整4277D Agent需要官方SDK；以下描述当前源码的消费端约定。[证据说明](evidence.md)区分代码事实与运行验证。

## 4277维布局

| 平坦区间（左闭右开） | 长度 | 信息 | layout |
|---|---|---|---|
| [0,73) | 73 | 49局部格 + 16全局状态 + 8合法动作 | 基础向量 |
| [73,661) | 588 | 7×7×12局部记忆 | cell-major HWC，Model转NCHW |
| [661,4245) | 3584 | 16×16×14全局记忆 | channel-major CHW |
| [4245,4277) | 32 | phase / battery / repeat / route / NPC标量 | 状态向量 |

环境字典中的21×21视野裁剪为7×7基础输入。16个全局状态按顺序为：step_norm、battery_ratio、cleaning_progress、remaining_dirt、pos_x_norm、pos_z_norm、N/E/S/W污渍射线距离、最近污渍距离、接近污渍标记、低电标记、最近桩距离、最近NPC距离、NPC危险标记。当前最近污渍距离实际读取整个21×21视野；旧注释写7×7，不能用注释替代执行代码。

局部12通道包含通行、未知、污渍、访问热度、frontier、桩热度、NPC热度、NPC预测、疑似NPC已清扫、回充轨迹、最近轨迹、清扫期已见。全局14通道在相似状态上加入已清扫/自身热度等。全局记忆来自累积观测，不是开局获取全图真值。

32D包含充电/清扫阶段、电量、停滞、重复、已知区域、NPC运动、路线等。`ENABLE_SCALAR_RISK_FEATURES=False` 时风险槽位使用旧语义，不能把关闭的风险槽位替换描述为当前生效贡献。

## 8方向动作

坐标使用 `(dx,dz)`：

| ID | 增量 | ID | 增量 |
|---|---|---|---|
| 0 | (1,0) | 4 | (-1,0) |
| 1 | (1,-1) | 5 | (-1,1) |
| 2 | (0,-1) | 6 | (0,1) |
| 3 | (-1,-1) | 7 | (1,1) |

没有独立充电或停止动作。mask合并平台合法动作、本地通行/斜角过滤和短时blocked记忆。NPC额外安全过滤/规则不等于环境保证无碰撞。地图二维规则读取按 `[z][x]`，不能随意转置既有网络输入。

## SDK字段与验证边界

Preprocessor读取 `env_obs.observation` 中的step_no、frame_state.heroes（pos.x/z、battery/battery_max、dirt_cleaned）、organs/NPC、env_info、legal_action/legal_act、map_info。正式protobuf/SDK的完整字段和版本没有恢复；这是调用代码的消费端约定，不是官方完整schema。

历史模块检查曾验证合成字典→73D与基础reward、Model分段reshape/forward；它不等于完整真实4277D轨迹验证。本仓库无权重，本次静态检查不创建Agent或Model。Model自动pad/truncate错误长度有掩盖版本错误的风险，原行为保留。

原 `SampleData` 的旧维度注释仍写69D，实际字段由 `Config.DIM_OF_OBSERVATION=4277` 决定；该注释是历史遗留，不应据此修改当前输入。原源码按字节保留，以文档澄清而未改策略。

