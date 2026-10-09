# PPO 与 Actor-Critic 实现

依据：[Model](../agent_ppo/model/model.py)、[Algorithm](../agent_ppo/algorithm/algorithm.py)、[Config](../agent_ppo/conf/conf.py)、[GAE](../agent_ppo/feature/definition.py)。描述当前保存工程，不套用到所有历史 checkpoint。

## 网络

| 模块 | 层与尺寸 |
|---|---|
| base | 73→Linear128→ReLU→Linear64→ReLU |
| local CNN | 12×7×7→Conv3×3(16,pad1)→ReLU→Conv3×3(32,pad1)→ReLU→Flatten→Linear64→ReLU |
| global CNN | 14×16×16→Conv3×3(16,pad1)→ReLU→Conv3×3(32,pad1)→ReLU→Flatten→Linear64→ReLU |
| scalar | 32→Linear32→ReLU→Linear32→ReLU |
| fusion | concat(base64,local64,scalar32)→Linear64→ReLU |
| phase context | concat(fusion64,global64)→Linear64→ReLU |
| phase gate | scalar hidden32→Linear16→ReLU→Linear3→softmax |

Actor 是 legacy logits + fusion delta + global delta，输出 8 logits。Critic 是 legacy value + fusion/global delta + gate 加权的三个 phase value delta，输出 1 value。三个头命名 clean/charge/post-charge，没有对应监督标签。总参数 678753，46 个 state_dict 键；显式地图/轨迹状态不等于 RNN。

Linear/Conv 使用正交初始化，bias 为零，输出 head gain 为 0.01。local flat 输入按 HWC reshape 后转换到 NCHW，global 按 CHW reshape。保留 legacy 分支是历史结构兼容设计，不能将当前结构认定为 39171 时的完整网络。

## Loss

`Algorithm._compute_loss` 对合法动作计算 softmax，取执行动作的新旧概率比。batch advantage 使用总体标准差标准化（eps1e-8，单元素不标准化）。

```text
ratio = new_prob(action) / max(old_prob(action), 1e-9)
policy_loss = mean(max(-ratio*adv, -clip(ratio,0.9,1.1)*adv))
value_clipped = old_value + clip(value_pred-old_value,-0.10,0.10)
value_loss = 0.5*mean(max((return-value_pred)^2,(return-value_clipped)^2))
entropy = mean(-sum(masked_probability * log(masked_probability)))
loss = policy_loss + 0.7*value_loss - beta*entropy
```

`USE_VALUE_CLIP` 配置存在，但实现无条件裁剪 value。日志 `approx_kl` 实际按旧 masked 分布与新分布的 KL 计算；`clip_frac` 为 ratio 超出 0.10 的样本比例。

Adam lr3e-5、eps1e-8、betas(0.9,0.999)，grad norm0.5。`_update_entropy_beta` 按本地 train_step 将 0.008 线性衰减到 0.0015，150000 调用封顶。一次 `learn` 只做一次 optimizer step；不能由平台 `samples_per_insert=5` 推断本地有五个 epoch，当前 limiter 为 MinSize。

## GAE

`sample_process` 将非尾样本 next_value 设为下一样本 value。workflow 为尾样本提供零 next_value，`_calc_gae` 逆序计算：

```text
delta = reward + 0.995*next_value - value
advantage = delta + 0.995*0.97*next_advantage
reward_sum = advantage + value
```

函数不使用 done。有限时域结束与外部 time-limit 截断是否应区分 bootstrap 尚未核实；没有修改这项原行为。

## 保留的工程限制

`predict` 的执行概率经过 guide/规则，`USE_GUIDED_BEHAVIOR_PROB=False` 使样本仍存原 PPO 分布。因此不能宣称严格 on-policy 已获验证；即使切换 old probability，硬动作映射也仍需单独分析。

Model 会 pad/truncate 错误长度，Agent loader 允许部分 key/shape；可能掩盖版本错配。历史模块数值检查是模块级证据，不是完整策略验证。本仓库无权重，不提供加载/推理命令；不进行新训练或修改 PPO。
