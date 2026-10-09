# 发布检查表

本文件是 `kaiwu-d02-robot-vacuum-ppo` 的本地发布准备记录。当前没有创建 GitHub 仓库、提交远程、Release 或推送。

## 已完成

- [x] 真实 `agent_ppo` 源码、配置和官方入口保留；22 个源码/配置文件与原件 SHA-256 一致，其中 17 个为 Python。
- [x] 候选目录中的 12 个 checkpoint 已与原始工程及完整备份核对后移出；公开树模型文件数量为 0。
- [x] 内部审查、私有日志、账号/平台元数据、验证脚本和历史备份共 95 个文件已移到候选目录外的本地归档，移动后哈希一致。
- [x] 原始工程 399 个文件哈希未改变。2026-10-08核查时，候选与原始目录均不是Git仓库；2026-10-09仅在独立候选目录初始化本地Git，原始工程保持不变。
- [x] 208 条终局记录和 79 条 learner 指标转为不含进程/IP/账号字段的结构化 JSONL。
- [x] 中英文 README、技术文档、NOTICE、`.gitignore` 和标准库静态检查已准备。
- [x] 未修改 PPO、reward、observation、model、workflow 或运行配置行为。

## 实测结果（2026-10-08）

| 检查 | 结果 | 验证范围 |
|---|---|---|
| `python -B tools/self_check.py` | PASS，49 个公开文件，20 个 Python 语法通过 | 内容、JSON、Markdown路径、22源码指纹、208/79数据格式、受限文件和常见凭证形态 |
| `python -B -m unittest discover -s tests -p test_candidate.py -v` | 7/7 PASS | 维度/动作/未知提交、参数快照及排除门禁；不是PPO训练或Agent测试 |
| `python -B scripts/summarize_experiments.py --check` | PASS | 四窗口的结果计数、均值、n和配置与公开数值记录一致 |
| 22源码/配置对原件 SHA-256 | 全部一致 | 17 Python、4 TOML、1原始gitignore |
| 4个TOML语法解析 | PASS | 只读解析，未改参数或调用平台 |
| Tencent copyright | 8个原始Python注释头保留 | 原模板作者声明未删除 |
| 全候选树模型扫描 | 0 | 包括隐藏目录；没有权重、checkpoint、模型备份或模型下载链接 |
| 忽略规则 | 14/14排除，10/10允许 | 临时空Git元数据中check-ignore；模型后缀/凭证/内部资料排除，核心源码/公开数据不误伤 |
| 敏感信息/路径扫描 | 未发现常见形态 | 无私有账号文件、原始日志、绝对机器路径、内部IP或凭证形态值；不构成任意敏感信息的绝对保证 |

`*.bin` 等规则在当前公开树没有同后缀的合法非模型资源可误伤。第三方库、SDK、地图和训练数据没有被复制进公开树。中英文README的维度、参数、结果表、缺失依赖与无模型政策人工核对一致。

## 待定发布事项

- [x] 所有者已确认比赛材料可以公开（USER_CONFIRMED_AUTHORIZATION）；据此完成本地源码整理，无需再次确认这项范围。
- [ ] 确定作者/团队署名与已有授权条款的公开引用方式。当前未提供可以统一再许可的文本，因此没有添加LICENSE；具体使用、修改、再分发和商业用途条件不能自行补造。
- [ ] 所有者确认正式发布；本任务到本地准备结束。SDK恢复、模型与完整训练验证不属于发布要求。

## 发布元数据（本地准备）

- Repository: `kaiwu-d02-robot-vacuum-ppo`
- Description: `Real competition implementation of a PPO-based robot cleaning agent for Tencent Kaiwu D02.`
- Topics: `reinforcement-learning`, `ppo`, `actor-critic`, `robotics`, `pytorch`, `robot-cleaning`, `tencent-kaiwu`

## 运行边界

候选树可做 Python 编译、文档/数据/受限文件静态检查和结构化指标重算；不承诺完整 Agent、官方环境、checkpoint 推理或重新训练。完整运行依赖未随仓库提供的 `common_python`、`kaiwudrl`、平台 `tools.*` 和官方环境服务。

## 发布前人工门禁

1. 复核 `NOTICE.md`、作者署名和授权条款；保留Tencent版权，未有依据不套用MIT。
2. 如发布前再修改文件，重跑上述三个标准库命令。
3. 本地Git已初始化；每次提交和远程发布前检查实际暂存区、历史blob与LFS指针。`.gitignore`不能移除已入历史的权重，不能仅靠忽略规则判断提交安全。
4. 仅公开当前候选树，禁止把本地完整备份或内部归档整体上传。

## 本地Git准备（2026-10-09）

- 本地分支：`main`；Git作者名：`NAN-JI77`。邮箱只保存在当前仓库Git配置和提交元数据中，不添加到README。
- 未修改全局Git配置。本项目`core.autocrlf=false`，用于保持原始源码与提交blob字节一致。
- 本项目`core.whitespace`识别CRLF为有效换行，同时保留行尾空格和末尾空行检查；没有关闭格式检查。
- 静态检查PASS、候选测试7/7、208条终局/79条learner记录重算PASS；22个源码/配置与原件SHA-256一致，原始399个文件未变。
- 初始提交仅纳入明确审核的49个发布文件；工作目录、暂存blob、提交历史和LFS分别核查，不执行`git add .`。
- 实际暂存49个文件逐个检查：与工作文件字节一致，22个原始源码/配置blob匹配清单SHA-256；未发现权重、二进制模型、LFS指针、凭证形态、个人路径或内部IP。`git lfs ls-files --all`为空，Git filter无LFS关联。
- `git diff --cached --stat`为49个文件。`git diff --cached --check`返回2，有9处格式提示：原始conf.py末尾空行、环境TOML的2行行尾空格，以及.gitignore、3个技术文档、候选测试和自检脚本的末尾空行。未发现冲突标记；为保留原文件字节，未格式化这些文件，不能把该命令标为PASS。
- 没有设置GitHub远程、推送或发布Release。完整训练、Agent和模型验证不在此阶段执行。
- 许可证仍以NOTICE中的来源说明为准；未授予统一MIT许可。官方模板/修改文件的再利用、商业使用与署名条款，以及外部第三方依赖的许可条件，须依据实际授权确认。
