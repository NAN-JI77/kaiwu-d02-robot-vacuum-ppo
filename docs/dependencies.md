# 依赖与复现说明

| 模块 | 依赖 | 当前状态 |
|---|---|---|
| Model / Algorithm | PyTorch、Config | 源码可读；历史模块级检查通过 |
| Preprocessor | NumPy、Config | 基础字段与 reward 代码可读 |
| GAE data classes | `common_python.utils.common_func.create_cls` | 缺失；数据容器/序列化接口，不是GAE数学本身 |
| Agent | `kaiwudrl.interface.agent.BaseAgent` | 缺失；生命周期、样本发送与部署接口 |
| Workflow | `tools.metrics_utils`、`tools.train_env_conf_validate`、common_python容灾 | 缺失；日志/配置校验与采样工作流 |
| Monitor | `kaiwudrl` 的 MonitorConfigBuilder | 缺失；平台监控 |
| train/eval/exam | `kaiwudrl` 运行器、`tools.eval.workflow.*` | 未随候选提供 |

`requirements.txt` 只记录曾用于模块检查的 NumPy 1.24.4、PyTorch 2.4.1 和旧 Python 所需 tomli，不是原比赛 runtime 的完整锁定文件。不要从 PyPI 用同名包替代官方平台模块，也不要将静态隔离检查当作真实环境运行。

当前公开树可以执行标准库静态检查和 JSONL 指标重算。完整策略推理和重新训练分别为 Level 3 未达到、Level 4 NOT_RUN。

普通算法核心主要依赖torch/Config，环境通信、训练工作流和部署生命周期依赖平台；两者的边界可以从import和调用链阅读，但本仓库没有伪造官方实现。`tools/self_check.py` 是发布检查工具，不是恢复的官方 `tools.*`。

授权允许公开与SDK实际可得是不同问题。没有恢复可验证的匹配SDK发行包/版本；本阶段无需继续搜索或重新配置平台。第三方库外部安装、未复制进仓库，许可证入口见 [NOTICE](../NOTICE.md)。
