# 复现能力

| 等级 | 当前状态 | 证据/限制 |
|---|---|---|
| Level 1：代码可阅读 | 达到 | 真实 `agent_ppo`、根配置和入口已保留 |
| Level 2：关键模块可独立测试 | 有历史证据；当前不重跑模型测试 | 先前 CPU 模块检查使用 Python 3.8.20、PyTorch 2.4.1+cu118、NumPy 1.24.4 |
| Level 3：独立策略推理 | 未达到 | 缺 `common_python`、`kaiwudrl`、环境和权重 |
| Level 4：重新训练并评估 | NOT_RUN | 官方运行器、服务和环境未恢复；本阶段按所有者决定不训练 |

公开候选的检查命令只做标准库工作：`tools/self_check.py`、`tests/test_candidate.py`、`scripts/summarize_experiments.py --check`。它们不导入 Agent、不加载 checkpoint、不连接平台。

保留的行为和版本风险包括：Model 对错误长度输入会 pad/truncate，loader 允许部分 key/shape，GAE 不读取 done，soft guidance 可能改变执行分布而 old probability 保留原 PPO 分布，checkpoint 只含 state_dict。当前最终提交模型和官方成绩 UNKNOWN。

原工程的历史模块测试曾检验PPO loss、Model前向、基础预处理以及隔离GAE/采样函数；它们未执行完整Agent环境轨迹。涉及模型加载的脚本/测试已移出公开树，本次7项候选测试仅验证发布内容，不增加训练/推理能力的结论。完整4277D环境预处理、多步规则状态、官方sample序列化和分布一致性仍未验证。
