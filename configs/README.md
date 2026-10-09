# 配置与源码指纹

[evidence_snapshot.json](evidence_snapshot.json) 是结构化文档快照，供公开信息一致性检查；不是运行配置，不被原 Agent/平台读取。

[source_manifest.json](source_manifest.json) 登记 22 个按字节保留的原始源码/配置文件、长度和 SHA-256，其中 17 个为 Python。运行配置仍位于 `agent_ppo/conf/` 和根 `conf/`。清单不含模型、私有路径、账号或内部迁移记录。

`current_preload_checkpoint=73189` 仅记录保存配置；对应权重不分发，`final_submission_checkpoint=null` 表示 UNKNOWN。`robot_count=4` 指 NPC 数量。当前 PPO 参数不能推广到每个历史窗口。
