# Kaiwu D02 Robot Vacuum PPO

[中文](README.md) | [Architecture](docs/architecture.md) | [PPO implementation](docs/ppo.md) | [Experiments](docs/experiments.md)

This is the actual robot-cleaning project used in the preliminary round of the **2026 Tencent AI Arena D02 competition**. It preserves competition PPO, the CNN + MLP network, observation memory, reward shaping and rule-assisted decisions. The task concerns discrete navigation, exploration and energy management in a partially observed grid.

**No trained weights, checkpoints or model download links are provided. Full training and inference have not been verified in an independent open-source environment. Some functions require the official Kaiwu SDK and services. The final submitted model, official score and ranking remain UNKNOWN.**

## Competition Background

A single learned robot cleans dirt, avoids official NPC robots and reaches chargers before its battery runs out. Cleaning, battery consumption, charging, collisions, termination and scoring are environment mechanisms. The [official track page](https://tencentarena.com/aiarena/zh/match/open-competition-2026/open-competition-2026-2) documents the task, not this project's results.

The saved configuration uses ten random maps, four NPCs, three chargers, battery capacity 150 and a 1000-step limit. `robot_count=4` denotes NPCs, not four PPO agents. This saved configuration is not a confirmed final-submission configuration.

## Project Architecture

Environment → observation preprocessing and map memory → CNN + MLP Actor-Critic → legal-action probabilities → soft guidance and rule fallback → movement → environment. Episode samples receive shaped and terminal rewards plus GAE, then enter platform buffering and learner services.

| Component | Source / function | Responsibility |
|---|---|---|
| Observation and state | [agent.py](agent_ppo/agent.py) / `observation_process` | Maintain maps, visits, trajectories and charge state; construct 4277D input |
| Actor-Critic | [model.py](agent_ppo/model/model.py) / `Model.forward` | Eight action logits and one state value |
| Hybrid decisions | [agent.py](agent_ppo/agent.py) / `predict`, `exploit` | Masking, guidance, sampling/greedy selection and rule adjustment |
| Reward | [preprocessor.py](agent_ppo/feature/preprocessor.py) / `reward_process` | Conditional cleaning, exploration, repeat and energy shaping |
| Advantages | [definition.py](agent_ppo/feature/definition.py) / `_calc_gae` | Reverse GAE recursion and value targets |
| PPO update | [algorithm.py](agent_ppo/algorithm/algorithm.py) / `_compute_loss`, `learn` | Clipped policy/value losses, entropy and one optimizer step |
| Collection | [train_workflow.py](agent_ppo/workflow/train_workflow.py) / `workflow` | Environment interaction, terminal shaping and sample submission |

See [architecture](docs/architecture.md) and [training data flow](docs/training.md).

## PPO Algorithm and Actor-Critic Network

The implementation uses discrete-action Clipped PPO with masked softmax, batch advantage normalization, clipped value loss, an entropy bonus and gradient clipping. Each `Algorithm.learn` call performs one optimizer step. There is no local multi-epoch loop; the full platform sampling schedule has not been recovered.

GAE computes `delta = reward + gamma * next_value - value`, then `advantage = delta + gamma * lambda * next_advantage` and `reward_sum = advantage + value`. The function does not read `done`; the workflow supplies zero `next_value` at the episode tail.

The 678753-parameter network combines a legacy 73→128→64 MLP, local 12-channel and global 14-channel CNNs (Conv16→Conv32→Flatten→64), a scalar MLP and additive actor/value delta heads. The critic also has three gated phase value heads. There is no RNN; Agent state provides memory. Head names do not establish supervised semantic specialization. See [ppo.md](docs/ppo.md) for exact layers and losses.

## Observation and Action Spaces

`4277 = 73 + 588 + 3584 + 32`:

| Segment | Content |
|---|---|
| 73D base | 49 local cells, 16 state features and eight legal-action entries |
| 588D local | 7×7×12 HWC local map, visit and trajectory memory |
| 3584D global | 14×16×16 CHW accumulated map memory |
| 32D scalar | Battery, repetition, stalls, routes, NPCs and phase state |

Global memory accumulates observations rather than receiving full-map ground truth. Actions 0..7 map to `(dx,dz)`: `(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1)`. There is no separate charge or stop action. See [observation_action.md](docs/observation_action.md).

## Reward Design and Hybrid Decision Strategy

Shaped training reward, environment reward and competition `total_score` are different quantities. Current terms include +0.16 per newly cleaned cell, +0.045 when cleaning occurs, -0.002 per step, +0.008 for a new cell, and conditional repetition, loops, stalls, energy risk, charge progress, NPC and terminal-efficiency terms. Their gates determine whether they apply. See [reward.md](docs/reward.md).

Network probabilities are blended with hand-written repeat suppression, known-dirt/frontier and charger-route guidance. Rules may then rewrite actions. BFS, NPC avoidance and stuck recovery are rule assistance. Base guidance is 0.10, with pressure settings of 0.22/0.30 and additional score bias.

Current `USE_GUIDED_BEHAVIOR_PROB=False` stores raw PPO probabilities even when behavior differs. This preserved consistency issue prevents a claim of verified strict on-policy sampling. Hard override rate excludes soft guidance. See [hybrid_decision.md](docs/hybrid_decision.md) for the network/rule/environment boundary.

## Saved Training Configuration

| Parameter | Saved value |
|---|---|
| gamma / lambda | 0.995 / 0.97 |
| Adam lr / eps / betas | 3e-5 / 1e-8 / (0.9, 0.999) |
| policy / value clip | 0.10 |
| value coefficient | 0.7 |
| entropy coefficient | 0.008→0.0015 over 150000 learner calls |
| gradient norm limit | 0.5 |
| platform buffer / batch | 10000 / 2048; Uniform, FIFO, MinSize |
| environment | Battery 150; three chargers; 1000 steps; four NPCs; ten random maps |

Sources: [conf.py](agent_ppo/conf/conf.py), [environment TOML](agent_ppo/conf/train_env_conf.toml), [platform TOML](conf/configure_app.toml). Historical `preload_model=true / 73189` and `latest` settings remain intact, but the required weight is excluded. The original entry points therefore cannot run directly from this distribution, and these settings do not identify the final submission. `train_test.py` requires the official runner.

## Engineering Challenges and Experimental Results

Supported engineering topics include map indexing/corner/blocked-TTL diagnosis, separating cleaning and charging-route repetition, soft guidance versus hard takeover, network evolution, low-resource charge failures and KL/clip monitoring. Seventeen historical attempts retain failures and rollbacks and distinguish VERIFIED, USER_CONFIRMED and UNVERIFIED evidence.

The following are **continuing-training windows**, not frozen-model official evaluations. Parameters continue to update after initialization.

| Window | Initial ID | Battery / chargers | Step cap / NPCs / maps | Episodes | Mean total_score |
|---|---|---|---|---|---|
| W10, Apr 19 | 39171 | 200 / 4 | 1000 / 4 / ten random maps | 53 | 755.830 |
| W11, Apr 19 | 39171 | 200 / 4 | 1000 / 4 / ten random maps | 40 | 701.300 |
| W12, Apr 19 | 39171 | 200 / 4 | 1000 / 4 / ten random maps | 82 | 772.659 |
| W27, Apr 27 | 73189 | 150 / 3 | 1000 / 4 / ten random maps | 33 | 354.939 |

W27 has 18 `FAIL_BATTERY` and seven `FAIL_BATTERY_LOW_CLEAN` episodes. It does not show that low-battery failures were solved. Different resources, code, rewards and continuing updates prevent fair checkpoint rankings. Custom `WIN` is not a competition victory. Public evidence includes [208 episode rows](experiments/episodes.jsonl), [79 learner rows](experiments/learner_metrics.jsonl) and [aggregates](experiments/window_metrics.json), with source-line references but without private logs or runtime metadata. See [experiments](docs/experiments.md) and [metric definitions](docs/evaluation.md).

## Project Structure

```text
agent_ppo/                 Original Agent, PPO, model, features, workflow and configuration
conf/                      Original platform configuration
train_test.py              Original platform training entry
configs/                   Documentation snapshot and 22-file SHA-256 manifest
experiments/               Structured historical metrics and aggregates
scripts/summarize_experiments.py  Standard-library aggregation
tools/self_check.py        Static publication checks
tests/test_candidate.py    Documentation/data/exclusion contract checks
docs/                      Technical explanations, experiments and release checklist
```

## Dependencies and Lightweight Checks

Ordinary modules use NumPy and PyTorch. [requirements.txt](requirements.txt) records versions used for prior module checks, not the original competition runtime lockfile. The complete Agent/workflow requires official `common_python`, `kaiwudrl`, platform `tools.*` and environment services. They are not distributed here. Do not substitute unverified packages with matching PyPI names. See [dependencies.md](docs/dependencies.md).

These commands use only Python's standard library. They do not import competition modules, load models or train:

```powershell
python -B tools/self_check.py
python -B -m unittest discover -s tests -p test_candidate.py -v
python -B scripts/summarize_experiments.py --check
```

## Reproducibility and Limitations

The actual source is readable (Level 1). Key modules previously passed Level 2 module checks on CPU with Python 3.8.20, PyTorch 2.4.1+cu118 and NumPy 1.24.4. Those runtime tests are archived locally. The public package provides static content checks and historical metric recomputation; this phase does not repeat Level 2 verification or promise standalone Agent or training execution.

Level 3 full-policy inference has not been achieved; Level 4 training and official evaluation have not been verified independently. Limitations include excluded weights, absent SDKs, Model padding/truncation, partial loading, GAE without `done`, behavior/old-probability consistency and incomplete experiment controls. See [reproducibility.md](docs/reproducibility.md).

## Copyright and Acknowledgements

The owner confirmed permission to disclose competition materials, recorded as USER_CONFIRMED_AUTHORIZATION. Original Tencent copyright and author notices remain intact. Official templates and personal modifications are not all claimed as original personal work. No blanket MIT or other relicensing is applied; see [NOTICE.md](NOTICE.md) for the unresolved exact redistribution terms. Thanks to Tencent AI Arena for the task, templates and platform, and to PyTorch, NumPy and related projects.

[RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) records local preparation and the remaining publication decisions.
