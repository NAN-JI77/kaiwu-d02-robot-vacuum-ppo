#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Robot Vacuum.
清扫大作战训练工作流。
"""

import os
import time
from collections import Counter

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery

def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read and validate user configuration
    # 读取和校验用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            try:
                agent.send_sample_data(g_data)
            finally:
                g_data.clear()

            now = time.time()
            if now - last_save_model_time >= int(getattr(Config, "SAVE_MODEL_INTERVAL_SECONDS", 1800)):
                try:
                    agent.save_model(id="latest")
                    if logger:
                        logger.info("periodic save_model(id='latest') success")
                except Exception as e:
                    if logger:
                        logger.error(f"periodic save_model failed: {e}")
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.model_loaded_once = False
        self._debug_event(
            "[DEBUG_READY] "
            f"cwd:{os.getcwd()} "
            f"workflow_file:{__file__} "
            f"workflow_proxy_path:None "
            # 运行标记：用于确认平台实际跑到的是当前 conf / Step2-lite 配置。
            f"debug_code_version:{getattr(Config, 'DEBUG_CODE_VERSION', None)} "
            f"load_latest:{getattr(Config, 'LOAD_LATEST_MODEL', None)} "
            f"load_model_id:{getattr(Config, 'LOAD_MODEL_ID', None)} "
            f"workflow_force_load_id:{getattr(Config, 'WORKFLOW_FORCE_LOAD_MODEL_ID', None)} "
            f"strict_baseline_model_id:{getattr(Config, 'STRICT_BASELINE_MODEL_ID', None)} "
            f"strict_baseline_reload_every_ep:{getattr(Config, 'STRICT_BASELINE_RELOAD_EVERY_EPISODE', None)} "
            f"model_arch:{getattr(Config, 'MODEL_ARCH_NAME', None)} "
            f"v22_score_stable:{getattr(Config, 'V22_SCORE_STABLE', None)} "
            f"v23_new_model_score_fix:{getattr(Config, 'V23_NEW_MODEL_SCORE_FIX', None)} "
            f"v24_opening_charge_guard:{getattr(Config, 'V24_OPENING_CHARGE_GUARD', None)} "
            f"v15_repeat_safe:{getattr(Config, 'V15_REPEAT_SAFE_ROLLBACK', None)} "
            f"v17_charge_path_fix:{getattr(Config, 'V17_CHARGE_PATH_FIX', None)} "
            f"v18_3c150_safe_charge:{getattr(Config, 'V18_3C150_SAFE_CHARGE', None)} "
            f"v19_3c150_early_charge:{getattr(Config, 'V19_3C150_EARLY_CHARGE', None)} "
            f"v20_charge_repeat_balance:{getattr(Config, 'V20_CHARGE_REPEAT_BALANCE', None)} "
            f"v21_score_push:{getattr(Config, 'V21_SCORE_PUSH', None)} "
            f"scalar_features:{getattr(Config, 'ENABLE_SCALAR_RISK_FEATURES', None)} "
            f"clean_eff_reward:{getattr(Config, 'ENABLE_CLEAN_EFFICIENCY_REWARD', None)} "
            f"charge_route_eff_reward:{getattr(Config, 'ENABLE_CHARGE_ROUTE_EFFICIENCY_REWARD', None)} "
            f"precharge_reward:{getattr(Config, 'ENABLE_PRE_CHARGE_POTENTIAL_REWARD', None)} "
            f"short_loop_reward:{getattr(Config, 'ENABLE_SHORT_LOOP_REWARD', None)} "
            f"progress_rule_override:{getattr(Config, 'ENABLE_PROGRESS_RULE_OVERRIDE', None)} "
            f"progress_override_min_stall:{getattr(Config, 'PROGRESS_OVERRIDE_MIN_STALL_STEPS', None)} "
            f"progress_override_cooldown:{getattr(Config, 'PROGRESS_OVERRIDE_COOLDOWN_STEPS', None)} "
            f"hard_override_behavior_prob:{getattr(Config, 'HARD_OVERRIDE_BEHAVIOR_PROB', None)} "
            f"guided_prob_blend:{getattr(Config, 'GUIDED_PROB_BLEND', None)} "
            f"guided_prob_pressure:{getattr(Config, 'GUIDED_PROB_BLEND_PRESSURE', None)} "
            f"guided_prob_hard_pressure:{getattr(Config, 'GUIDED_PROB_BLEND_HARD_PRESSURE', None)} "
            f"use_guided_behavior_prob:{getattr(Config, 'USE_GUIDED_BEHAVIOR_PROB', None)} "
            f"simple_charge_route_trigger:{getattr(Config, 'ENABLE_SIMPLE_CHARGE_ROUTE_TRIGGER', None)} "
            f"simple_charge_direct_trigger:{getattr(Config, 'ENABLE_SIMPLE_CHARGE_DIRECT_TRIGGER_DIST', None)} "
            f"simple_charge_direct_progress:{getattr(Config, 'ENABLE_SIMPLE_CHARGE_DIRECT_PROGRESS_DIST', None)} "
            f"simple_charge_route_reserve:{getattr(Config, 'SIMPLE_CHARGE_ROUTE_RESERVE_STEPS', None)} "
            f"simple_charge_route_cap:{getattr(Config, 'SIMPLE_CHARGE_ROUTE_TRIGGER_CAP', None)} "
            f"simple_charge_entry_path_max:{getattr(Config, 'SIMPLE_CHARGE_ENTRY_PATH_MAX', None)} "
            f"simple_charge_route_memory:{getattr(Config, 'ENABLE_SIMPLE_CHARGE_ROUTE_MEMORY', None)} "
            f"simple_charge_route_memory_steps:{getattr(Config, 'SIMPLE_CHARGE_ROUTE_MEMORY_STEPS', None)} "
            f"simple_charge_direct_path:{getattr(Config, 'ENABLE_SIMPLE_CHARGE_DIRECT_PATH_DIST', None)} "
            f"simple_charge_direct_trigger:{getattr(Config, 'ENABLE_SIMPLE_CHARGE_DIRECT_TRIGGER_DIST', None)} "
            f"simple_charge_direct_bfs:{getattr(Config, 'SIMPLE_CHARGE_DIRECT_BFS_LIMIT', None)} "
            f"simple_charge_early_safe_ratio:{getattr(Config, 'SIMPLE_CHARGE_EARLY_SAFE_RATIO', None)} "
            f"simple_charge_early_safe_min_dist:{getattr(Config, 'SIMPLE_CHARGE_EARLY_SAFE_MIN_DIST', None)} "
            f"simple_charge_low_capture:{getattr(Config, 'ENABLE_SIMPLE_CHARGE_LOW_BATTERY_CAPTURE', None)} "
            f"simple_charge_low_capture_ratio:{getattr(Config, 'SIMPLE_CHARGE_LOW_BATTERY_CAPTURE_RATIO', None)} "
            f"simple_charge_low_capture_path:{getattr(Config, 'SIMPLE_CHARGE_LOW_BATTERY_CAPTURE_PATH_DIST', None)} "
            f"simple_charge_reenter_cooldown:{getattr(Config, 'SIMPLE_CHARGE_REENTER_COOLDOWN_STEPS', None)} "
            f"simple_charge_unknown_trigger:{getattr(Config, 'SIMPLE_CHARGE_UNKNOWN_TRIGGER_RATIO', None)} "
            f"opening_no_charge_guard:{getattr(Config, 'ENABLE_OPENING_NO_CHARGE_GUARD', None)} "
            f"opening_no_charge_far:{getattr(Config, 'OPENING_NO_CHARGE_FAR_TRIGGER', None)} "
            f"opening_no_charge_unknown:{getattr(Config, 'OPENING_NO_CHARGE_UNKNOWN_TRIGGER', None)} "
            f"local_fresh_rescue:{getattr(Config, 'ENABLE_LOCAL_FRESH_PROGRESS_RESCUE', None)} "
            f"local_low_visit_rescue:{getattr(Config, 'ENABLE_LOCAL_LOW_VISIT_PROGRESS_RESCUE', None)} "
            f"progress_route_min_stall:{getattr(Config, 'PROGRESS_ROUTE_RESCUE_MIN_STALL_STEPS', None)} "
            f"clean_repeat_density_reward:{getattr(Config, 'ENABLE_CLEAN_REPEAT_DENSITY_REWARD', None)} "
            f"progress_score_soft_bias:{getattr(Config, 'ENABLE_PROGRESS_SCORE_SOFT_BIAS', None)} "
            f"progress_score_soft_blend:{getattr(Config, 'PROGRESS_SCORE_SOFT_BIAS_BLEND', None)} "
            f"progress_score_soft_hard_blend:{getattr(Config, 'PROGRESS_SCORE_SOFT_BIAS_HARD_BLEND', None)} "
            f"post_charge_guidance:{getattr(Config, 'ENABLE_POST_CHARGE_GUIDANCE', None)} "
            f"post_charge_escape:{getattr(Config, 'ENABLE_POST_CHARGE_ESCAPE', None)} "
            f"charge_route_escape_soft:{getattr(Config, 'ENABLE_CHARGE_ROUTE_ESCAPE_SOFT_BIAS', None)} "
            f"charge_route_escape_blend:{getattr(Config, 'CHARGE_ROUTE_ESCAPE_BLEND', None)} "
            f"charge_route_escape_hard:{getattr(Config, 'CHARGE_ROUTE_ESCAPE_HARD_BLEND', None)} "
            f"clean_target_route_soft:{getattr(Config, 'ENABLE_CLEAN_TARGET_ROUTE_SOFT_BIAS', None)} "
            f"clean_target_route_blend:{getattr(Config, 'CLEAN_TARGET_ROUTE_SOFT_BLEND', None)} "
            f"clean_target_route_hard:{getattr(Config, 'CLEAN_TARGET_ROUTE_SOFT_HARD_BLEND', None)} "
            f"avoid_charger_zone:{getattr(Config, 'AVOID_CHARGER_ZONE_WHEN_NOT_CHARGING', None)} "
            f"charger_zone_avoid_ratio:{getattr(Config, 'AVOID_CHARGER_ZONE_BATTERY_RATIO', None)} "
            f"high_battery_charger_avoid_dist:{getattr(Config, 'HIGH_BATTERY_CHARGER_AVOID_DIST', None)} "
            f"debug_paths:{self._debug_log_paths()}"
        )

    def _log_info(self, msg):
        if self.logger:
            self.logger.info(msg)

    def _debug_log_paths(self):
        log_file = getattr(Config, "DEBUG_LOG_FILE", "")
        if not log_file:
            return []
        if os.path.isabs(log_file):
            return [log_file]

        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paths = [os.path.join(repo_dir, log_file)]
        cwd_path = os.path.abspath(log_file)
        if cwd_path not in paths:
            paths.append(cwd_path)
        return paths

    def _write_debug_log(self, msg):
        log_paths = self._debug_log_paths()
        if not log_paths:
            return
        for log_path in log_paths:
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"write debug log failed for {log_path}: {e}")

    def _debug_event(self, msg):
        self._log_info(msg)
        self._write_debug_log(msg)

    def _map_values_summary(self, fm):
        values = sorted(int(v) for v in getattr(fm, "map_values_seen", set()))
        if not values:
            return "none"
        return "/".join(str(v) for v in values)

    def _dist_or_none(self, value):
        """把距离字段规整成数字或 None，避免日志里混入路径字符串。"""
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value) or value < 0:
            return None
        return int(value)

    def _valid_path_dist(self, value):
        value = self._dist_or_none(value)
        if value is None or value >= 99:
            return None
        return value

    def _charger_sources(self, fm):
        """汇总当前可见充电桩和可能存在的缓存充电桩，只用于日志和距离诊断。"""
        visible = list(getattr(fm, "charger_list", []) or [])
        cache = []
        for owner in (self.agent, fm):
            for attr in ("cached_charger_list", "known_charger_list", "known_charger_positions", "charger_cache"):
                values = getattr(owner, attr, None)
                if not values:
                    continue
                if isinstance(values, dict):
                    values = list(values.keys())
                cache.extend(list(values))

        merged = []
        seen = set()
        for pos in visible + cache:
            try:
                item = (int(pos[0]), int(pos[1]))
            except Exception:
                continue
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
        return visible, cache, merged

    def _workflow_proxy_charger_path(self, charger_list):
        """通过 agent 暴露的路径接口取距离；simple charge 下可能返回 None。"""
        if not charger_list:
            return None
        try:
            return self._dist_or_none(self.agent._charger_route_distance(charger_list))
        except Exception:
            return None

    def _workflow_direct_charger_path(self, charger_list):
        """绕过 simple-charge 的动作分支，只为日志直接计算一次 BFS 路径距离。"""
        if not charger_list:
            return None
        try:
            targets = []
            if hasattr(self.agent, "_charger_zone_targets"):
                targets = self.agent._charger_zone_targets()
            if not targets and hasattr(self.agent, "_npc_safe_charger_targets"):
                targets = self.agent._npc_safe_charger_targets(charger_list)
            if not targets:
                targets = charger_list
            if not hasattr(self.agent, "_bfs_distance_to_targets"):
                return None
            limit = int(getattr(Config, "CHARGE_ROUTE_BFS_LIMIT", 2600))
            return self._dist_or_none(self.agent._bfs_distance_to_targets(targets, max_nodes=limit))
        except Exception:
            return None

    def _effective_charger_path_dist(self, agent_current_path, workflow_direct_path, workflow_proxy_path, charger_min_dist):
        """GAMEOVER 统一距离：path 可信时用 path，否则退回最近充电桩距离。"""
        for value in (workflow_direct_path, agent_current_path, workflow_proxy_path, charger_min_dist):
            valid = self._valid_path_dist(value)
            if valid is not None:
                return valid
        return 99

    def _try_load_latest_model_once(self):
        if self.model_loaded_once:
            return

        if not getattr(Config, "LOAD_LATEST_MODEL", False):
            self._log_info("LOAD_LATEST_MODEL is False, keep current/preloaded model weights")
            self.model_loaded_once = True
            return

        # v23：按用户要求，不再强制回 71467，也不每局重载基线。
        # 这里尊重平台/根配置或本地 latest，让新训练模型继续作为当前模型；
        # 本轮只修代码行为，不改变模型结构、不覆盖 checkpoint。
        model_id = getattr(Config, "WORKFLOW_FORCE_LOAD_MODEL_ID", None) or getattr(Config, "LOAD_MODEL_ID", "latest")
        if model_id is None or str(model_id).strip() == "":
            model_id = "latest"
        try:
            loaded = self.agent.load_model(id=model_id)
            if loaded:
                self._log_info(f"load_model(id='{model_id}') success; using current/new model path")
            else:
                self._log_info(f"load_model(id='{model_id}') skipped: no compatible model file; keep current/preloaded model")
        except Exception as e:
            self._log_info(f"load_model(id='{model_id}') skipped: {e}; keep current/preloaded model")
        finally:
            self.model_loaded_once = True

    def _calc_repeat_ratio(self):
        visit_count = getattr(self.agent.preprocessor, "visit_count", {})
        total_visits = sum(visit_count.values())
        if total_visits <= 0:
            return 0.0
        return 1.0 - float(len(visit_count)) / float(total_visits)

    def _local_view_3x3(self):
        fm = self.agent.preprocessor
        view_map = getattr(fm, "_view_map", None)
        if view_map is None:
            return []
        center = int(getattr(fm, "VIEW_HALF", 10))
        try:
            return view_map[center - 1 : center + 2, center - 1 : center + 2].astype(int).tolist()
        except Exception:
            return []

    def _log_stuck_debug(self, step, obs_data, last_logged_stuck):
        fm = self.agent.preprocessor
        stuck_steps = int(getattr(fm, "stuck_steps", 0))
        threshold = int(getattr(Config, "STUCK_DEBUG_LOG_STEPS", 80))
        interval = int(getattr(Config, "STUCK_DEBUG_LOG_INTERVAL", 80))
        if stuck_steps < threshold or stuck_steps < last_logged_stuck + interval:
            return last_logged_stuck

        cur_pos = getattr(fm, "cur_pos", None)
        charger_list = getattr(fm, "charger_list", [])
        npc_list = getattr(fm, "npc_list", [])
        charger_dist = self.agent._min_dist(cur_pos, charger_list) if cur_pos is not None else None
        npc_dist = self.agent._min_dist(cur_pos, npc_list) if cur_pos is not None else None
        battery = float(getattr(fm, "battery", 0))
        battery_max = float(max(getattr(fm, "battery_max", 1), 1))
        blocked_action_ttl = list(getattr(fm, "blocked_action_ttl", []))
        blocked_cell_ttl = getattr(fm, "blocked_cell_ttl", {})
        top_blocked_cells = sorted(blocked_cell_ttl.items(), key=lambda item: item[1], reverse=True)[:5]
        legal_action = getattr(obs_data, "legal_action", None)
        if legal_action is None:
            legal_action = getattr(fm, "_legal_act", None)

        self._debug_event(
            "[STUCK_DEBUG] "
            f"ep:{self.episode_cnt} step:{step} stuck:{stuck_steps} "
            f"pos:{cur_pos} last_action:{getattr(self.agent, 'last_action', None)} "
            f"battery_ratio:{battery / battery_max:.3f} "
            f"is_charging:{getattr(fm, 'is_charging', None)} "
            f"force_charge:{getattr(self.agent, 'force_charge_mode', None)} "
            f"charge_phase_steps:{getattr(fm, 'charge_phase_steps', 0)} "
            f"charger_dist:{charger_dist if charger_dist is not None else 99} "
            f"npc_dist:{npc_dist if npc_dist is not None else 99} "
            f"legal:{legal_action} "
            f"blocked_ttl:{blocked_action_ttl} "
            f"blocked_cells:{top_blocked_cells} "
            f"local3x3:{self._local_view_3x3()}"
        )
        return stuck_steps

    def _phase_ratios(self, phase_stats, step):
        clean_steps = max(int(phase_stats.get("clean_steps", 0)), 1)
        charge_route_steps = max(int(phase_stats.get("charge_route_steps", 0)), 1)
        charge_steps = int(phase_stats.get("charge_route_steps", 0)) + int(phase_stats.get("charge_zone_steps", 0))
        return {
            "clean_repeat_ratio": float(phase_stats.get("clean_repeat_steps", 0)) / float(clean_steps),
            "charge_route_revisit_ratio": float(phase_stats.get("charge_route_revisit_steps", 0)) / float(charge_route_steps),
            "charge_step_ratio": float(charge_steps) / float(max(step, 1)),
        }

    def _rule_rate(self, rule_override_reasons, step, *names):
        """按原因拆分规则接管率，避免只看总 override rate 误判。"""
        if step <= 0:
            return 0.0
        return float(sum(rule_override_reasons.get(name, 0) for name in names)) / float(max(step, 1))

    def _update_phase_stats(self, phase_stats):
        fm = self.agent.preprocessor
        cur_pos = getattr(fm, "cur_pos", None)
        if cur_pos is None:
            return

        all_seen = phase_stats["all_seen"]
        clean_seen = phase_stats["clean_seen"]
        charge_route_seen = phase_stats["charge_route_seen"]
        force_charge = bool(getattr(self.agent, "force_charge_mode", False))
        is_charging = bool(getattr(fm, "is_charging", False))
        battery_gained = bool(getattr(fm, "battery_gain", False))
        in_charge_zone = bool(self.agent._in_charger_zone(cur_pos))

        if force_charge and not in_charge_zone and not is_charging:
            phase_stats["charge_route_steps"] += 1
            if cur_pos in all_seen:
                phase_stats["charge_route_revisit_steps"] += 1
            charge_route_seen.add(cur_pos)
        elif force_charge or is_charging or in_charge_zone:
            phase_stats["charge_zone_steps"] += 1
            if battery_gained:
                phase_stats["charge_gain_steps"] += 1
            elif force_charge or in_charge_zone:
                phase_stats["charge_no_gain_steps"] += 1
        else:
            phase_stats["clean_steps"] += 1
            if cur_pos in clean_seen:
                phase_stats["clean_repeat_steps"] += 1
            clean_seen.add(cur_pos)

        all_seen.add(cur_pos)

    def _log_phase_debug(self, step, phase_stats, last_logged_phase):
        interval = int(getattr(Config, "PHASE_DEBUG_LOG_INTERVAL", 120))
        if step < last_logged_phase + interval:
            return last_logged_phase

        ratios = self._phase_ratios(phase_stats, step)
        overall_repeat = self._calc_repeat_ratio()
        fm = self.agent.preprocessor
        npc_clean_visible = int(getattr(fm, "suspected_npc_clean_visible_count", 0))
        npc_clean_suspects = len(getattr(fm, "suspected_npc_cleaned_cells", {}))
        map_values = self._map_values_summary(fm)
        map_id = int(getattr(fm, "map_id", -1))
        agent_current_path = getattr(self.agent, "current_charger_path_dist", None)
        agent_current_path = self._dist_or_none(agent_current_path)
        if agent_current_path is None:
            agent_current_path = 99
        visible_chargers, cache_chargers, merged_chargers = self._charger_sources(fm)
        workflow_proxy_path = self._workflow_proxy_charger_path(merged_chargers)
        workflow_direct_path = self._workflow_direct_charger_path(merged_chargers)
        cur_pos = getattr(fm, "cur_pos", None)
        charger_min_dist = self.agent._min_dist(cur_pos, getattr(fm, "charger_list", [])) if cur_pos is not None else None
        if charger_min_dist is None:
            charger_min_dist = 99
        charger_path_dist = self._effective_charger_path_dist(
            agent_current_path,
            workflow_direct_path,
            workflow_proxy_path,
            charger_min_dist,
        )
        battery = float(getattr(fm, "battery", 0))
        battery_max = float(max(getattr(fm, "battery_max", 1), 1))
        nonstandard_map_values = any(int(v) not in (0, 1, 2) for v in getattr(fm, "map_values_seen", set()))
        if (
            overall_repeat < 0.40
            and ratios["clean_repeat_ratio"] < 0.35
            and ratios["charge_route_revisit_ratio"] < 0.70
            and npc_clean_visible < int(getattr(Config, "NPC_CLEAN_LOG_VISIBLE_THRESHOLD", 6))
            and not nonstandard_map_values
            and int(getattr(self.agent.preprocessor, "stuck_steps", 0)) < int(getattr(Config, "STUCK_DEBUG_LOG_STEPS", 80))
        ):
            return last_logged_phase

        self._debug_event(
            "[PHASE_DEBUG] "
            f"ep:{self.episode_cnt} step:{step} "
            f"map_id:{map_id} "
            f"repeat_ratio:{overall_repeat:.3f} "
            f"clean_repeat_ratio:{ratios['clean_repeat_ratio']:.3f} "
            f"charge_route_revisit_ratio:{ratios['charge_route_revisit_ratio']:.3f} "
            f"charge_step_ratio:{ratios['charge_step_ratio']:.3f} "
            f"battery_ratio:{battery / battery_max:.3f} "
            f"remaining_charge:{battery:.0f} "
            f"map_values:{map_values} "
            f"npc_clean_visible:{npc_clean_visible} "
            f"npc_clean_suspects:{npc_clean_suspects} "
            f"clean_steps:{phase_stats['clean_steps']} "
            f"charge_route_steps:{phase_stats['charge_route_steps']} "
            f"charge_zone_steps:{phase_stats['charge_zone_steps']} "
            f"charge_gain_steps:{phase_stats['charge_gain_steps']} "
            f"charge_no_gain_steps:{phase_stats['charge_no_gain_steps']} "
            f"agent_episode_charge_gain:{phase_stats['charge_gain_steps']} "
            f"agent_episode_charge_no_gain:{phase_stats['charge_no_gain_steps']} "
            f"agent_zone_no_gain:{getattr(self.agent, 'charge_zone_no_gain_steps', 0)} "
            f"agent_charge_gain:{getattr(self.agent, 'current_charge_gain_steps', 0)} "
            f"agent_charge_gain_last:{getattr(self.agent, 'last_charge_gain_steps', 0)} "
            f"agent_current_charge_gain_segment:{getattr(self.agent, 'current_charge_gain_steps', 0)} "
            f"agent_current_path:{agent_current_path} "
            f"simple_trigger_path:{getattr(self.agent, 'simple_charge_trigger_path_dist', 99)} "
            f"workflow_direct_path:{workflow_direct_path if workflow_direct_path is not None else 99} "
            f"workflow_proxy_path:{workflow_proxy_path if workflow_proxy_path is not None else 99} "
            f"charger_min_dist:{charger_min_dist} "
            f"charger_path_dist:{charger_path_dist} "
            f"scalar_safe_charger_dist:{getattr(self.agent, 'scalar_safe_charger_dist', 99.0):.1f} "
            f"scalar_charger_dist_delta:{getattr(self.agent, 'scalar_charger_dist_delta', 0.0):.3f} "
            f"scalar_pre_charge_active:{getattr(self.agent, 'scalar_pre_charge_active', 0.0):.0f} "
            f"scalar_battery_risk_active:{getattr(self.agent, 'scalar_battery_risk_active', 0.0):.0f} "
            f"scalar_short_loop_score:{getattr(self.agent, 'scalar_short_loop_score', 0.0):.3f} "
            f"scalar_battery_no_progress:{getattr(self.agent, 'scalar_battery_no_progress_steps', 0.0):.3f} "
            f"visible_chargers:{len(visible_chargers)} "
            f"merged_chargers:{len(merged_chargers)} "
            f"cache_chargers:{len(cache_chargers)} "
            f"pos:{getattr(fm, 'cur_pos', None)} "
            f"force_charge:{getattr(self.agent, 'force_charge_mode', None)} "
            f"is_charging:{getattr(fm, 'is_charging', None)} "
            f"battery_gain:{getattr(fm, 'battery_gain', None)} "
            f"stuck:{getattr(fm, 'stuck_steps', 0)}"
        )
        return step

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        单局流程（generator），完成一局后 yield 整局样本。
        """
        while True:
            # Periodically get training metrics
            # 定期打印训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics: {training_metrics}")

            # Reset environment
            # 重置环境
            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent and load latest model
            # 重置 Agent，加载最新模型
            self.agent.reset(env_obs)
            self._try_load_latest_model_once()

            # Initial observation processing
            # 初始观测
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            rule_override_count = 0
            rule_override_reasons = Counter()
            max_stuck_steps = 0
            last_logged_stuck = 0
            last_logged_phase = 0
            start_pos = getattr(self.agent.preprocessor, "cur_pos", None)
            phase_stats = {
                "all_seen": set([start_pos]) if start_pos is not None else set(),
                "clean_seen": set([start_pos]) if start_pos is not None else set(),
                "charge_route_seen": set(),
                "clean_steps": 0,
                "clean_repeat_steps": 0,
                "charge_route_steps": 0,
                "charge_route_revisit_steps": 0,
                "charge_zone_steps": 0,
                "charge_gain_steps": 0,
                "charge_no_gain_steps": 0,
            }

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Agent inference / 推理动作
                act_data_list = self.agent.predict([obs_data])
                if not act_data_list:
                    self.logger.error("agent.predict returned empty action data, skip current episode")
                    break
                act_data = act_data_list[0]
                override_flag = int(getattr(self.agent, "last_rule_override", 0))
                rule_override_count += override_flag
                if override_flag:
                    reason = str(getattr(self.agent, "last_rule_override_reason", "unknown"))
                    rule_override_reasons[reason] += 1
                act = self.agent.action_process(act_data)

                # Environment step / 与环境交互
                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                frame_no = env_obs["frame_no"]
                step += 1
                done = terminated or truncated

                # Process next observation
                # 特征处理
                _obs_data, _ = self.agent.observation_process(env_obs)
                _obs_data.frame_no = frame_no
                max_stuck_steps = max(max_stuck_steps, int(getattr(self.agent.preprocessor, "stuck_steps", 0)))
                self._update_phase_stats(phase_stats)
                last_logged_stuck = self._log_stuck_debug(step, obs_data, last_logged_stuck)
                last_logged_phase = self._log_phase_debug(step, phase_stats, last_logged_phase)

                reward_scalar = float(self.agent.last_reward)
                total_reward += reward_scalar

                # Terminal reward calculation
                # 终局奖励
                final_reward = 0.0
                if done:
                    fm = self.agent.preprocessor
                    env_info = env_obs["observation"]["env_info"]
                    total_score = float(env_info.get("total_score", 0.0))
                    clean_score = float(env_info.get("clean_score", total_score))
                    known_cell_count = len(getattr(fm, "known_cells", {}))
                    known_dirt_count = len(getattr(fm, "known_dirt_positions", set()))
                    npc_clean_suspect_count = len(getattr(fm, "suspected_npc_cleaned_cells", {}))
                    npc_clean_visible_count = int(getattr(fm, "suspected_npc_clean_visible_count", 0))
                    npc_clean_events = int(getattr(fm, "suspected_npc_clean_events", 0))
                    dirty_disappear_transitions = int(getattr(fm, "dirty_disappear_transitions", 0))
                    npc_dirty_clear_suspect_transitions = int(getattr(fm, "npc_dirty_clear_suspect_transitions", 0))
                    map_values = self._map_values_summary(fm)
                    map_id = int(getattr(fm, "map_id", -1))
                    blocked_action_hits = int(getattr(fm, "blocked_action_hits", 0))
                    battery_max_seen = int(getattr(fm, "battery_max", 0))
                    visible_chargers, cache_chargers, merged_chargers = self._charger_sources(fm)
                    charger_count_seen = len(visible_chargers)
                    agent_charge_steps = int(phase_stats.get("charge_route_steps", 0)) + int(
                        phase_stats.get("charge_zone_steps", 0)
                    )
                    env_charge_count = float(env_info.get("charge_count", phase_stats.get("charge_gain_steps", 0)))
                    env_remaining_charge = float(env_info.get("remaining_charge", getattr(fm, "battery", 0)))
                    _, _, charge_env_pressure = self.agent._charge_env_pressure()
                    cleaning_ratio = fm.dirt_cleaned / max(fm.total_dirt, 1)
                    clean_per_100_steps = 100.0 * float(fm.dirt_cleaned) / float(max(step, 1))
                    clean_score_step_ratio = clean_score / float(max(step, 1))
                    steps_since_clean = int(getattr(fm, "steps_since_clean", 0))
                    battery_ratio = float(getattr(fm, "battery", 0)) / float(max(getattr(fm, "battery_max", 1), 1))
                    phase_ratios = self._phase_ratios(phase_stats, step)
                    cur_pos = getattr(fm, "cur_pos", None)
                    npc_min_dist = self.agent._min_dist(cur_pos, getattr(fm, "npc_list", [])) if cur_pos is not None else None
                    if npc_min_dist is None:
                        npc_min_dist = 99
                    charger_min_dist = self.agent._min_dist(cur_pos, getattr(fm, "charger_list", [])) if cur_pos is not None else None
                    if charger_min_dist is None:
                        charger_min_dist = 99
                    workflow_proxy_path = self._workflow_proxy_charger_path(merged_chargers) if cur_pos is not None else None
                    workflow_direct_path = self._workflow_direct_charger_path(merged_chargers) if cur_pos is not None else None
                    agent_current_path = self._dist_or_none(getattr(self.agent, "current_charger_path_dist", None))
                    if agent_current_path is None:
                        agent_current_path = 99
                    charger_path_dist = self._effective_charger_path_dist(
                        agent_current_path,
                        workflow_direct_path,
                        workflow_proxy_path,
                        charger_min_dist,
                    )

                    reason_topk = int(getattr(Config, "RULE_OVERRIDE_REASON_LOG_TOPK", 6))
                    rule_override_reason_str = "/".join(
                        f"{name}:{count}" for name, count in rule_override_reasons.most_common(reason_topk)
                    ) or "none"
                    rule_force_charge_rate = self._rule_rate(rule_override_reasons, step, "force_charge")
                    rule_low_battery_capture_rate = self._rule_rate(rule_override_reasons, step, "low_battery_charge_capture")
                    rule_leave_charger_rate = self._rule_rate(rule_override_reasons, step, "leave_charger")
                    rule_local_fresh_rate = self._rule_rate(rule_override_reasons, step, "local_fresh_rescue")
                    rule_progress_repeat_rate = self._rule_rate(rule_override_reasons, step, "progress_repeat_rescue")
                    rule_progress_route_rate = self._rule_rate(rule_override_reasons, step, "progress_route_rescue")
                    rule_progress_rescue_rate = self._rule_rate(
                        rule_override_reasons,
                        step,
                        "progress_repeat_rescue",
                        "progress_route_rescue",
                        "local_fresh_rescue",
                    )
                    rule_stuck_escape_rate = self._rule_rate(rule_override_reasons, step, "stuck_or_loop_escape")
                    rule_npc_avoid_rate = self._rule_rate(rule_override_reasons, step, "npc_avoid")

                    target_ratio = max(float(Config.CLEAN_SCORE_STEP_TARGET_RATIO), 1e-6)
                    low_ratio = max(float(Config.CLEAN_SCORE_STEP_LOW_RATIO), 1e-6)
                    low_score_min_steps = int(getattr(Config, "CLEAN_SCORE_STEP_FAIL_MIN_STEPS", 300))
                    low_score_penalty = 0.0
                    if step >= low_score_min_steps and clean_score_step_ratio < low_ratio:
                        low_score_penalty = Config.LOW_SCORE_FAIL_EXTRA * (1.0 - clean_score_step_ratio / low_ratio)

                    stale_repeat_penalty = 0.0
                    clean_repeat_ratio = float(phase_ratios["clean_repeat_ratio"])
                    stale_repeat_ratio = float(getattr(Config, "LOW_CLEAN_REPEAT_RATIO", 0.18))
                    stale_repeat_steps = int(getattr(Config, "LOW_CLEAN_STEPS_SINCE_CLEAN", 60))
                    if (
                        low_score_penalty < 0.0
                        and clean_repeat_ratio >= stale_repeat_ratio
                        and steps_since_clean >= stale_repeat_steps
                    ):
                        repeat_scale = min(1.5, clean_repeat_ratio / max(stale_repeat_ratio, 1e-6) - 1.0)
                        stale_scale = min(1.5, float(steps_since_clean) / float(max(stale_repeat_steps, 1)) - 1.0)
                        stale_repeat_penalty = float(getattr(Config, "LOW_CLEAN_REPEAT_STALE_EXTRA", -1.2)) * (
                            1.0 + 0.5 * repeat_scale + 0.5 * stale_scale
                        )

                    charge_overhead_penalty = 0.0
                    charge_step_ratio = float(phase_ratios["charge_step_ratio"])
                    charge_loop_ratio = float(phase_ratios["charge_route_revisit_ratio"])
                    charge_route_steps = int(phase_stats.get("charge_route_steps", 0))
                    low_clean_charge_step_ratio = float(getattr(Config, "LOW_CLEAN_CHARGE_STEP_RATIO", 0.18))
                    low_clean_charge_loop_ratio = float(getattr(Config, "LOW_CLEAN_CHARGE_ROUTE_REVISIT_RATIO", 0.55))
                    low_clean_charge_route_steps = int(getattr(Config, "LOW_CLEAN_CHARGE_ROUTE_STEPS", 140))
                    if (
                        low_score_penalty < 0.0
                        and clean_repeat_ratio < stale_repeat_ratio
                        and steps_since_clean < stale_repeat_steps
                        and (
                            charge_step_ratio >= low_clean_charge_step_ratio
                            or charge_loop_ratio >= low_clean_charge_loop_ratio
                            or charge_route_steps >= low_clean_charge_route_steps
                        )
                    ):
                        step_scale = max(0.0, charge_step_ratio - low_clean_charge_step_ratio) / max(
                            1.0 - low_clean_charge_step_ratio,
                            1e-6,
                        )
                        loop_scale = max(0.0, charge_loop_ratio - low_clean_charge_loop_ratio) / max(
                            1.0 - low_clean_charge_loop_ratio,
                            1e-6,
                        )
                        route_step_scale = max(0.0, float(charge_route_steps - low_clean_charge_route_steps)) / float(
                            max(low_clean_charge_route_steps, 1)
                        )
                        charge_overhead_penalty = float(getattr(Config, "LOW_CLEAN_CHARGE_OVERHEAD_EXTRA", -1.0)) * (
                            0.5 + 0.5 * min(1.5, max(step_scale, loop_scale, route_step_scale))
                        )

                    high_remain_penalty = 0.0
                    high_remain_ratio = float(getattr(Config, "LOW_CLEAN_HIGH_REMAIN_RATIO", 0.58))
                    high_remain_min_steps = int(getattr(Config, "LOW_CLEAN_HIGH_REMAIN_MIN_STEPS", 900))
                    if (
                        low_score_penalty < 0.0
                        and step >= high_remain_min_steps
                        and battery_ratio >= high_remain_ratio
                    ):
                        remain_scale = min(1.5, (battery_ratio - high_remain_ratio) / max(1.0 - high_remain_ratio, 1e-6))
                        score_scale = min(1.5, max(0.0, low_ratio - clean_score_step_ratio) / max(low_ratio, 1e-6))
                        high_remain_penalty = float(getattr(Config, "LOW_CLEAN_HIGH_REMAIN_EXTRA", -1.6)) * (
                            0.5 + 0.5 * max(remain_scale, score_scale)
                        )

                    battery_fail_penalty = 0.0
                    if battery_ratio <= 0.02:
                        near_path_limit = int(getattr(Config, "FAIL_BATTERY_NEAR_CHARGER_PATH_DIST", 20))
                        near_min_limit = int(getattr(Config, "FAIL_BATTERY_NEAR_CHARGER_MIN_DIST", 8))
                        if charger_path_dist <= near_path_limit or charger_min_dist <= near_min_limit:
                            path_scale = 1.0 - min(float(charger_path_dist), float(near_path_limit)) / float(max(near_path_limit, 1))
                            min_scale = 1.0 - min(float(charger_min_dist), float(near_min_limit)) / float(max(near_min_limit, 1))
                            near_scale = max(path_scale, min_scale)
                            battery_fail_penalty += float(getattr(Config, "FAIL_BATTERY_NEAR_CHARGER_EXTRA", -1.0)) * (
                                0.5 + 0.5 * near_scale
                            )

                        high_charge_step_ratio = float(getattr(Config, "FAIL_BATTERY_HIGH_CHARGE_STEP_RATIO", 0.22))
                        high_charge_loop_ratio = float(getattr(Config, "FAIL_BATTERY_ROUTE_LOOP_RATIO", 0.55))
                        if charge_step_ratio >= high_charge_step_ratio or charge_loop_ratio >= high_charge_loop_ratio:
                            step_scale = max(0.0, charge_step_ratio - high_charge_step_ratio) / max(
                                1.0 - high_charge_step_ratio,
                                1e-6,
                            )
                            loop_scale = max(0.0, charge_loop_ratio - high_charge_loop_ratio) / max(
                                1.0 - high_charge_loop_ratio,
                                1e-6,
                            )
                            battery_fail_penalty += float(getattr(Config, "FAIL_BATTERY_ROUTE_LOOP_EXTRA", -1.0)) * (
                                0.5 + 0.5 * max(step_scale, loop_scale)
                            )

                    if truncated:
                        # Max-step survival has no value by itself; reward clean-score efficiency.
                        efficiency_delta = (clean_score_step_ratio - target_ratio) / target_ratio
                        efficiency_delta = min(1.5, max(-1.0, efficiency_delta))
                        final_reward = (
                            Config.WIN_CLEAN_RATIO_REWARD * cleaning_ratio
                            + Config.CLEAN_SCORE_STEP_EFFICIENCY_REWARD * efficiency_delta
                        )
                        if low_score_penalty < 0.0:
                            final_reward += low_score_penalty
                            final_reward += stale_repeat_penalty
                            final_reward += charge_overhead_penalty
                            result_str = "LOW_CLEAN_SCORE_STEPS"
                            final_reward += high_remain_penalty
                        else:
                            result_str = "WIN"
                    else:
                        # Early termination (battery depleted or collision): small penalty
                        # 提前结束（电量耗尽或碰撞）：小惩罚
                        final_reward = Config.FAIL_PENALTY
                        result_str = "FAIL"
                        if low_score_penalty < 0.0:
                            final_reward += low_score_penalty
                            final_reward += stale_repeat_penalty
                            final_reward += charge_overhead_penalty
                            result_str = "FAIL_LOW_CLEAN_SCORE"
                            final_reward += high_remain_penalty
                        if battery_ratio <= 0.02:
                            final_reward += Config.BATTERY_DEPLETED_FAIL_PENALTY
                            final_reward += battery_fail_penalty
                            result_str = "FAIL_BATTERY_LOW_CLEAN" if low_score_penalty < 0.0 else "FAIL_BATTERY"

                    self.logger.info(
                        f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} "
                        f"map_id:{map_id} "
                        f"result:{result_str} final_bonus:{final_reward:.2f} "
                        f"total_score:{total_score:.1f} "
                        f"clean_score:{clean_score:.1f} "
                        f"clean_score_step_ratio:{clean_score_step_ratio:.3f} "
                        f"total_reward:{total_reward:.3f} "
                        f"dirt_cleaned:{fm.dirt_cleaned}/{fm.total_dirt} "
                        f"dirt_total:{fm.total_dirt} "
                        f"clean_per_100_steps:{clean_per_100_steps:.2f} "
                        f"steps_since_clean:{steps_since_clean} "
                        f"repeat_ratio:{self._calc_repeat_ratio():.3f} "
                        f"clean_repeat_ratio:{phase_ratios['clean_repeat_ratio']:.3f} "
                        f"charge_route_revisit_ratio:{phase_ratios['charge_route_revisit_ratio']:.3f} "
                        f"charge_step_ratio:{phase_ratios['charge_step_ratio']:.3f} "
                        f"charge_count:{env_charge_count:.0f} "
                        f"remaining_charge:{env_remaining_charge:.0f} "
                        f"battery_ratio:{battery_ratio:.3f} "
                        f"agent_charge_steps:{agent_charge_steps} "
                        f"charge_route_steps:{phase_stats['charge_route_steps']} "
                        f"charge_zone_steps:{phase_stats['charge_zone_steps']} "
                        f"charge_gain_steps:{phase_stats['charge_gain_steps']} "
                        f"charge_no_gain_steps:{phase_stats['charge_no_gain_steps']} "
                        f"agent_episode_charge_gain:{phase_stats['charge_gain_steps']} "
                        f"agent_episode_charge_no_gain:{phase_stats['charge_no_gain_steps']} "
                        f"agent_episode_charge_zone:{phase_stats['charge_zone_steps']} "
                        f"agent_zone_no_gain:{getattr(self.agent, 'charge_zone_no_gain_steps', 0)} "
                        f"agent_charge_gain:{getattr(self.agent, 'current_charge_gain_steps', 0)} "
                        f"agent_charge_gain_last:{getattr(self.agent, 'last_charge_gain_steps', 0)} "
                        f"agent_current_charge_gain_segment:{getattr(self.agent, 'current_charge_gain_steps', 0)} "
                        f"agent_current_path:{agent_current_path} "
                        f"simple_trigger_path:{getattr(self.agent, 'simple_charge_trigger_path_dist', 99)} "
                        f"workflow_direct_path:{workflow_direct_path if workflow_direct_path is not None else 99} "
                        f"workflow_proxy_path:{workflow_proxy_path if workflow_proxy_path is not None else 99} "
                        f"charger_min_dist:{charger_min_dist} "
                        f"charger_path_dist:{charger_path_dist} "
                        f"scalar_safe_charger_dist:{getattr(self.agent, 'scalar_safe_charger_dist', 99.0):.1f} "
                        f"scalar_charger_dist_delta:{getattr(self.agent, 'scalar_charger_dist_delta', 0.0):.3f} "
                        f"scalar_pre_charge_active:{getattr(self.agent, 'scalar_pre_charge_active', 0.0):.0f} "
                        f"scalar_battery_risk_active:{getattr(self.agent, 'scalar_battery_risk_active', 0.0):.0f} "
                        f"scalar_short_loop_score:{getattr(self.agent, 'scalar_short_loop_score', 0.0):.3f} "
                        f"scalar_battery_no_progress:{getattr(self.agent, 'scalar_battery_no_progress_steps', 0.0):.3f} "
                        f"visible_chargers:{len(visible_chargers)} "
                        f"merged_chargers:{len(merged_chargers)} "
                        f"cache_chargers:{len(cache_chargers)} "
                        f"npc_min_dist:{npc_min_dist} "
                        f"charger_count_seen:{charger_count_seen} "
                        f"battery_max_seen:{battery_max_seen} "
                        f"charge_env_pressure:{charge_env_pressure:.2f} "
                        f"known_cells:{known_cell_count} "
                        f"known_dirt:{known_dirt_count} "
                        f"map_values:{map_values} "
                        f"npc_clean_suspects:{npc_clean_suspect_count} "
                        f"npc_clean_visible:{npc_clean_visible_count} "
                        f"npc_clean_events:{npc_clean_events} "
                        f"dirty_disappear:{dirty_disappear_transitions} "
                        f"npc_dirty_suspect:{npc_dirty_clear_suspect_transitions} "
                        f"max_stuck_steps:{max_stuck_steps} "
                        f"blocked_action_hits:{blocked_action_hits} "
                        f"rule_override_rate:{rule_override_count / max(step, 1):.3f} "
                        f"rule_force_charge:{rule_force_charge_rate:.3f} "
                        f"rule_low_charge_capture:{rule_low_battery_capture_rate:.3f} "
                        f"rule_leave_charger:{rule_leave_charger_rate:.3f} "
                        f"rule_local_fresh:{rule_local_fresh_rate:.3f} "
                        f"rule_progress_route:{rule_progress_route_rate:.3f} "
                        f"rule_progress_total:{rule_progress_rescue_rate:.3f} "
                        f"rule_stuck_escape:{rule_stuck_escape_rate:.3f} "
                        f"rule_npc_avoid:{rule_npc_avoid_rate:.3f} "
                        f"rule_override_reasons:{rule_override_reason_str}"
                    )
                    self._write_debug_log(
                        f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} "
                        f"map_id:{map_id} "
                        f"result:{result_str} total_score:{total_score:.1f} "
                        f"clean_score:{clean_score:.1f} "
                        f"dirt_total:{fm.total_dirt} "
                        f"repeat_ratio:{self._calc_repeat_ratio():.3f} "
                        f"clean_repeat_ratio:{phase_ratios['clean_repeat_ratio']:.3f} "
                        f"charge_route_revisit_ratio:{phase_ratios['charge_route_revisit_ratio']:.3f} "
                        f"charge_step_ratio:{phase_ratios['charge_step_ratio']:.3f} "
                        f"charge_count:{env_charge_count:.0f} "
                        f"remaining_charge:{env_remaining_charge:.0f} "
                        f"battery_ratio:{battery_ratio:.3f} "
                        f"agent_charge_steps:{agent_charge_steps} "
                        f"clean_steps:{phase_stats['clean_steps']} "
                        f"charge_route_steps:{phase_stats['charge_route_steps']} "
                        f"charge_zone_steps:{phase_stats['charge_zone_steps']} "
                        f"charge_gain_steps:{phase_stats['charge_gain_steps']} "
                        f"charge_no_gain_steps:{phase_stats['charge_no_gain_steps']} "
                        f"agent_episode_charge_gain:{phase_stats['charge_gain_steps']} "
                        f"agent_episode_charge_no_gain:{phase_stats['charge_no_gain_steps']} "
                        f"agent_episode_charge_zone:{phase_stats['charge_zone_steps']} "
                        f"agent_zone_no_gain:{getattr(self.agent, 'charge_zone_no_gain_steps', 0)} "
                        f"agent_charge_gain:{getattr(self.agent, 'current_charge_gain_steps', 0)} "
                        f"agent_charge_gain_last:{getattr(self.agent, 'last_charge_gain_steps', 0)} "
                        f"agent_current_charge_gain_segment:{getattr(self.agent, 'current_charge_gain_steps', 0)} "
                        f"agent_current_path:{agent_current_path} "
                        f"workflow_direct_path:{workflow_direct_path if workflow_direct_path is not None else 99} "
                        f"workflow_proxy_path:{workflow_proxy_path if workflow_proxy_path is not None else 99} "
                        f"charger_min_dist:{charger_min_dist} "
                        f"charger_path_dist:{charger_path_dist} "
                        f"scalar_safe_charger_dist:{getattr(self.agent, 'scalar_safe_charger_dist', 99.0):.1f} "
                        f"scalar_charger_dist_delta:{getattr(self.agent, 'scalar_charger_dist_delta', 0.0):.3f} "
                        f"scalar_pre_charge_active:{getattr(self.agent, 'scalar_pre_charge_active', 0.0):.0f} "
                        f"scalar_battery_risk_active:{getattr(self.agent, 'scalar_battery_risk_active', 0.0):.0f} "
                        f"scalar_short_loop_score:{getattr(self.agent, 'scalar_short_loop_score', 0.0):.3f} "
                        f"scalar_battery_no_progress:{getattr(self.agent, 'scalar_battery_no_progress_steps', 0.0):.3f} "
                        f"visible_chargers:{len(visible_chargers)} "
                        f"merged_chargers:{len(merged_chargers)} "
                        f"cache_chargers:{len(cache_chargers)} "
                        f"npc_min_dist:{npc_min_dist} "
                        f"charger_count_seen:{charger_count_seen} "
                        f"battery_max_seen:{battery_max_seen} "
                        f"charge_env_pressure:{charge_env_pressure:.2f} "
                        f"map_values:{map_values} "
                        f"npc_clean_suspects:{npc_clean_suspect_count} "
                        f"npc_clean_visible:{npc_clean_visible_count} "
                        f"npc_clean_events:{npc_clean_events} "
                        f"dirty_disappear:{dirty_disappear_transitions} "
                        f"npc_dirty_suspect:{npc_dirty_clear_suspect_transitions} "
                        f"max_stuck_steps:{max_stuck_steps} "
                        f"blocked_action_hits:{blocked_action_hits} "
                        f"rule_override_rate:{rule_override_count / max(step, 1):.3f} "
                        f"rule_force_charge:{rule_force_charge_rate:.3f} "
                        f"rule_low_charge_capture:{rule_low_battery_capture_rate:.3f} "
                        f"rule_leave_charger:{rule_leave_charger_rate:.3f} "
                        f"rule_local_fresh:{rule_local_fresh_rate:.3f} "
                        f"rule_progress_route:{rule_progress_route_rate:.3f} "
                        f"rule_progress_total:{rule_progress_rescue_rate:.3f} "
                        f"rule_stuck_escape:{rule_stuck_escape_rate:.3f} "
                        f"rule_npc_avoid:{rule_npc_avoid_rate:.3f} "
                        f"rule_override_reasons:{rule_override_reason_str}"
                    )

                # Build sample frame
                # 构造样本帧
                reward_arr = np.array([reward_scalar], dtype=np.float32)
                value_arr = act_data.value.flatten()[: Config.VALUE_NUM]

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array(act_data.action),
                    reward=reward_arr,
                    done=np.array([float(done)]),
                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    value=value_arr,
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    # Add terminal reward to last frame
                    # 终局奖励叠加到最后一步
                    collector[-1].reward = collector[-1].reward + np.array([final_reward], dtype=np.float32)

                    # Monitor reporting / 监控上报
                    if self.monitor:
                        self.monitor.put_data(
                            {
                                os.getpid(): {
                                    "debug_code_version": float(getattr(Config, "DEBUG_CODE_VERSION", 0)),
                                    "repeat_ratio": self._calc_repeat_ratio(),
                                    "clean_repeat_ratio": phase_ratios["clean_repeat_ratio"],
                                    "charge_route_revisit_ratio": phase_ratios["charge_route_revisit_ratio"],
                                    "charge_step_ratio": phase_ratios["charge_step_ratio"],
                                    "clean_ratio": cleaning_ratio,
                                    "clean_score_step_ratio": clean_score_step_ratio,
                                    "clean_per_100_steps": clean_per_100_steps,
                                    "steps_since_clean": steps_since_clean,
                                    "battery_ratio": battery_ratio,
                                    "map_id": float(map_id),
                                    "episode_charge_count": env_charge_count,
                                    "episode_remaining_charge": env_remaining_charge,
                                    "agent_charge_steps": float(agent_charge_steps),
                                    "agent_episode_charge_gain": float(phase_stats["charge_gain_steps"]),
                                    "agent_episode_charge_no_gain": float(phase_stats["charge_no_gain_steps"]),
                                    "agent_episode_charge_zone": float(phase_stats["charge_zone_steps"]),
                                    "agent_current_charge_gain_segment": float(
                                        getattr(self.agent, "current_charge_gain_steps", 0)
                                    ),
                                    "agent_current_path": float(agent_current_path),
                                    "scalar_safe_charger_dist": float(getattr(self.agent, "scalar_safe_charger_dist", 99.0)),
                                    "scalar_charger_dist_delta": float(getattr(self.agent, "scalar_charger_dist_delta", 0.0)),
                                    "scalar_pre_charge_active": float(getattr(self.agent, "scalar_pre_charge_active", 0.0)),
                                    "scalar_battery_risk_active": float(getattr(self.agent, "scalar_battery_risk_active", 0.0)),
                                    "scalar_short_loop_score": float(getattr(self.agent, "scalar_short_loop_score", 0.0)),
                                    "scalar_battery_no_progress": float(getattr(self.agent, "scalar_battery_no_progress_steps", 0.0)),
                                    "npc_min_dist": npc_min_dist,
                                    "npc_clean_suspect_count": float(npc_clean_suspect_count),
                                    "npc_clean_visible_count": float(npc_clean_visible_count),
                                    "blocked_action_hits": blocked_action_hits,
                                    "max_stuck_steps": float(max_stuck_steps),
                                    "rule_override_rate": rule_override_count / max(step, 1),
                                    "rule_force_charge": rule_force_charge_rate,
                                    "rule_leave_charger": rule_leave_charger_rate,
                                    "rule_local_fresh": rule_local_fresh_rate,
                                    "rule_progress_repeat": rule_progress_repeat_rate,
                                    "rule_progress_route": rule_progress_route_rate,
                                    "rule_progress_rescue": rule_progress_rescue_rate,
                                    "rule_stuck_escape": rule_stuck_escape_rate,
                                    "rule_npc_avoid": rule_npc_avoid_rate,
                                }
                            }
                        )
                        self.last_report_monitor_time = time.time()

                    # Compute GAE and yield samples
                    # GAE 计算并 yield 样本
                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Advance state / 状态推进
                obs_data = _obs_data
