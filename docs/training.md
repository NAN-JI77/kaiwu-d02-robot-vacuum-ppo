# 训练数据流与复现边界

真实入口是腾讯开悟平台工作流，不是独立 Python trainer：`train_test.py` 调用平台运行器，`conf/algo_conf_robot_vacuum.toml` 将 actor/learner/aisrv 指向 `agent_ppo.agent.Agent`，训练工作流位于 `agent_ppo/workflow/train_workflow.py`。

```text
Environment.reset
  -> Agent.observation_process (4277D observation + legal mask)
  -> Model.forward / predict / rule adjustment
  -> Environment.step
  -> agent.last_reward + terminal shaping
  -> sample_process / GAE
  -> send_sample_data / platform buffer
  -> Algorithm.learn (one optimizer step)
  -> model synchronization
```

workflow 保存前一状态、执行动作、old value/prob、reward、done 和 next value；episode 尾部的 `next_value` 为零，再调用 `_calc_gae`。`Algorithm.learn` 消费批次，完成 forward、loss、backward、gradient clip 和 Adam step。平台服务的多 worker、Reverb、模型同步与调度未包含在仓库中。

## 当前边界

缺失 `common_python`、`kaiwudrl`、平台 `tools.*` 与官方环境服务，因此本候选不承诺 Agent、官方评估或重新训练可运行。模型权重和 checkpoint 也不分发。本地公开内容检查只做静态检查和历史结构化数据重算，没有启动训练、推理、环境或平台。

保存的配置仍含 `preload_model=true` 与 `preload_model_id=73189`，这是历史运行入口，不是本仓库可用权重。不要仅通过关闭配置来宣称获得独立可运行性。

## 平台调度与配置

原平台配置容量10000、batch2048、Uniform sampler、FIFO remover、MinSize limiter。`samples_per_insert=5/error_buffer=5`仅为另一limiter模式的配置，不表示本实现有5个PPO epoch。模型同步/worker/seed、样本复用及原Python/CUDA版本UNKNOWN。

workflow按代码根目录读取 `agent_ppo/conf/train_env_conf.toml`，根 `conf/` 参与平台加载。配置中的diy段原样保留以维持原文件指纹，但本仓库不包含agent_diy。eval/exam引用外部官方tools模块。保存间隔500秒，网络文件只保存state_dict，不能恢复optimizer/RNG/replay和全部learner状态。
