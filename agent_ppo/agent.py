#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum Agent.
清扫大作战 Agent 主类。
"""

import os
import json
from collections import Counter, deque

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from kaiwudrl.interface.agent import BaseAgent


DIR_MAP = {
    0: (1, 0),
    1: (1, -1),
    2: (0, -1),
    3: (-1, -1),
    4: (-1, 0),
    5: (-1, 1),
    6: (0, 1),
    7: (1, 1),
}


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0
        self.last_rule_override = 0
        self.last_rule_override_reason = "none"
        self.last_progress_override_step = -10**9
        self._pending_rule_override_reason = "none"
        self.force_charge_mode = False
        self._reset_charge_route_memory()

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        """Reset per-episode state.

        每局开始时重置 Agent 内部状态。
        """
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0
        self.last_rule_override = 0
        self.last_rule_override_reason = "none"
        self.last_progress_override_step = -10**9
        self._pending_rule_override_reason = "none"
        self.force_charge_mode = False
        self._reset_charge_route_memory()

    def _cfg(self, name, default):
        return getattr(Config, name, default)

    def _flag(self, name, default=False):
        return bool(getattr(Config, name, default))

    def _post_charge_guidance_enabled(self):
        return self._flag("ENABLE_POST_CHARGE_GUIDANCE", False)

    def _post_charge_escape_enabled(self):
        return self._post_charge_guidance_enabled() and self._flag("ENABLE_POST_CHARGE_ESCAPE", False)

    def _npc_clean_guidance_enabled(self):
        return self._flag("ENABLE_NPC_CLEAN_GUIDANCE", False)

    def _charge_env_pressure(self, max_scale=None):
        battery_max = float(max(getattr(self.preprocessor, "battery_max", 1), 1))
        charger_count = max(len(getattr(self.preprocessor, "charger_list", [])), 1)
        base_battery = float(max(self._cfg("ADAPTIVE_CHARGE_BASE_BATTERY_MAX", 200), 1))
        base_charger_count = float(max(self._cfg("ADAPTIVE_CHARGE_BASE_CHARGER_COUNT", 4), 1))
        battery_factor = max(1.0, base_battery / battery_max)
        charger_factor = max(1.0, base_charger_count / float(charger_count))
        pressure = 1.0
        pressure += float(self._cfg("ADAPTIVE_CHARGE_BATTERY_WEIGHT", 0.60)) * (battery_factor - 1.0)
        pressure += float(self._cfg("ADAPTIVE_CHARGE_SCARCITY_WEIGHT", 0.80)) * (charger_factor - 1.0)
        if max_scale is not None:
            pressure = min(float(max_scale), pressure)
        return battery_factor, charger_factor, max(1.0, pressure)

    def _adaptive_charge_steps(self, base_steps, max_scale_name):
        if isinstance(base_steps, str):
            base_steps = self._cfg(base_steps, 0)
        _, _, pressure = self._charge_env_pressure(self._cfg(max_scale_name, 1.0))
        return float(base_steps) * float(pressure)

    def _adaptive_charge_cap(self, base_cap, extra_name="ADAPTIVE_CHARGE_CAP_MAX_EXTRA"):
        _, _, pressure = self._charge_env_pressure(self._cfg("ADAPTIVE_CHARGE_RESERVE_MAX_SCALE", 1.0))
        extra_max = float(self._cfg(extra_name, 0.0))
        extra = min(extra_max, 0.20 * max(0.0, float(pressure) - 1.0))
        hard_max = float(self._cfg("ADAPTIVE_CHARGE_CAP_HARD_MAX", 0.68))
        return min(hard_max, float(base_cap) + extra)

    def _reset_charge_route_memory(self):
        memory = (
            int(self._cfg("SIMPLE_CHARGE_ROUTE_MEMORY_STEPS", 160))
            if self._flag("ENABLE_SIMPLE_CHARGE_MODE", False) and self._flag("ENABLE_SIMPLE_CHARGE_ROUTE_MEMORY", False)
            else (1 if self._flag("ENABLE_SIMPLE_CHARGE_MODE", False) else 160)
        )
        self.recent_charge_route_cells = deque(maxlen=max(memory, 1))
        self.post_charge_clean_steps = 0
        self.post_charge_escape_steps = 0
        self.post_charge_escape_ratio = 0.0
        self.current_charge_route_steps = 0
        self.current_charge_zone_steps = 0
        self.current_charge_gain_steps = 0
        self.episode_charge_gain_steps = 0
        self.last_charge_route_steps = 0
        self.last_charge_zone_steps = 0
        self.last_charge_gain_steps = 0
        self._last_charge_context = False
        self.force_charge_search_steps = 0
        self.force_charge_no_progress_steps = 0
        self.charge_zone_no_gain_steps = 0
        self.current_charger_path_dist = None
        self.last_charger_path_dist = None
        self.simple_charge_trigger_path_dist = 99
        self.clean_phase_seen = set()
        self.clean_phase_steps = 0
        self.clean_phase_repeat_steps = 0
        self.clean_escape_steps = 0
        self.clean_escape_ratio = 0.0
        self._last_phase_update_step_no = None
        self.last_force_charge_exit_step = -10**9

    def _next_pos(self, cur_pos, action):
        dx, dz = DIR_MAP.get(int(action), (0, 0))
        return (cur_pos[0] + dx, cur_pos[1] + dz)

    def _manhattan(self, a, b):
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    def _chebyshev(self, a, b):
        return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))

    def _min_dist(self, cur_pos, pos_list):
        if not pos_list:
            return None
        return min(self._manhattan(cur_pos, p) for p in pos_list)

    def _min_chebyshev_dist(self, cur_pos, pos_list):
        if not pos_list:
            return None
        return min(self._chebyshev(cur_pos, p) for p in pos_list)

    def _npc_dist(self, pos):
        npc_list = getattr(self.preprocessor, "npc_list", [])
        if pos is None or not npc_list:
            return None
        return self._min_dist(pos, npc_list)

    def _npc_clean_suspect_ttl(self, pos):
        if not self._npc_clean_guidance_enabled():
            return 0
        suspects = getattr(self.preprocessor, "suspected_npc_cleaned_cells", {})
        if pos is None or not suspects:
            return 0
        return int(suspects.get(pos, 0))

    def _is_npc_clean_suspect(self, pos):
        return self._npc_clean_suspect_ttl(pos) > 0

    def _npc_crowded(self, pos, avoid_dist=None):
        dist = self._npc_dist(pos)
        if dist is None:
            return False
        if avoid_dist is None:
            avoid_dist = self._cfg("CHARGE_TARGET_NPC_AVOID_DIST", self._cfg("NPC_AVOID_DIST", 4))
        return int(dist) <= int(avoid_dist)

    def _npc_safe_charger_targets(self, charger_list):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if not charger_list:
            return []

        avoid_dist = int(self._cfg("CHARGE_TARGET_NPC_AVOID_DIST", self._cfg("NPC_AVOID_DIST", 4)))
        safe = []
        crowded = []
        for charger in charger_list:
            npc_dist = self._npc_dist(charger)
            score = self._manhattan(cur_pos, charger) if cur_pos is not None else 0
            if npc_dist is None or int(npc_dist) > avoid_dist:
                safe.append((score, charger))
            else:
                crowded.append((score + 100 + (avoid_dist - int(npc_dist) + 1) * 10, charger))

        targets = safe if safe else crowded
        targets.sort(key=lambda item: item[0])
        return [pos for _, pos in targets]

    def _charger_zone_dist(self, pos):
        charger_list = getattr(self.preprocessor, "charger_list", [])
        if pos is None or not charger_list:
            return None
        return self._min_chebyshev_dist(pos, charger_list)

    def _in_charger_zone(self, pos=None):
        cur_pos = getattr(self.preprocessor, "cur_pos", None) if pos is None else pos
        dist = self._charger_zone_dist(cur_pos)
        return dist is not None and dist <= self._cfg("CHARGER_NEAR_DIST", 1)

    def _battery_gained_last_step(self):
        if hasattr(self.preprocessor, "battery_gain"):
            return bool(getattr(self.preprocessor, "battery_gain", False))
        last_battery = getattr(self.preprocessor, "last_battery", None)
        battery = getattr(self.preprocessor, "battery", None)
        return last_battery is not None and battery is not None and battery > last_battery

    def _norm01(self, value, scale):
        scale = float(max(scale, 1.0))
        return min(1.0, max(0.0, float(value) / scale))

    def _proximity_heat(self, dist, radius):
        if dist is None:
            return 0.0
        radius = float(max(radius, 1.0))
        return max(0.0, 1.0 - float(dist) / radius)

    def _local_cell(self, pos):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        view_map = getattr(self.preprocessor, "_view_map", None)
        if cur_pos is None or view_map is None:
            return None

        li = pos[1] - cur_pos[1] + self.preprocessor.VIEW_HALF
        lj = pos[0] - cur_pos[0] + self.preprocessor.VIEW_HALF
        if 0 <= li < view_map.shape[0] and 0 <= lj < view_map.shape[1]:
            return int(view_map[li, lj])
        return None

    def _opposite_action(self, action):
        opposite = {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}
        return opposite.get(int(action), None)

    def _legal_actions(self, legal_action):
        return [a for a, ok in enumerate(legal_action) if ok > 0]

    def _raw_legal_actions(self, fallback_legal_action):
        raw_legal_action = getattr(self.preprocessor, "_legal_act", None)
        if not raw_legal_action:
            raw_legal_action = fallback_legal_action
        return [a for a, ok in enumerate(raw_legal_action) if ok > 0]

    def _passable_legal_actions(self, legal_action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return self._legal_actions(legal_action)

        return [
            action
            for action in self._legal_actions(legal_action)
            if self._action_is_passable(cur_pos, action)
        ]

    def _static_passable_legal_actions(self, legal_action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return self._legal_actions(legal_action)

        return [
            action
            for action in self._legal_actions(legal_action)
            if self._action_is_static_passable(cur_pos, action)
        ]

    def _action_is_passable(self, cur_pos, action):
        if not self._action_is_static_passable(cur_pos, action):
            return False
        dx, dz = DIR_MAP.get(int(action), (0, 0))
        nxt = (cur_pos[0] + dx, cur_pos[1] + dz)
        return getattr(self.preprocessor, "blocked_cell_ttl", {}).get(nxt, 0) <= 0

    def _action_is_static_passable(self, cur_pos, action):
        dx, dz = DIR_MAP.get(int(action), (0, 0))
        nxt = (cur_pos[0] + dx, cur_pos[1] + dz)
        if self._local_cell(nxt) == 0:
            return False

        if self._cfg("AVOID_DIAGONAL_CORNER_CUT", True) and dx != 0 and dz != 0:
            side_x = (cur_pos[0] + dx, cur_pos[1])
            side_z = (cur_pos[0], cur_pos[1] + dz)
            if self._local_cell(side_x) == 0 or self._local_cell(side_z) == 0:
                return False

        return True

    def _recent_position_penalty(self, pos):
        history = list(getattr(self.preprocessor, "pos_history", []))
        penalty = 0.0
        if len(history) >= 2 and pos == history[-2]:
            penalty += 3.0
        penalty += 1.0 * float(history[-8:].count(pos))
        return penalty

    def _is_recent_loop(self):
        history = list(getattr(self.preprocessor, "pos_history", []))
        if len(history) < 8:
            return False
        return len(set(history[-8:])) <= 3

    def _local_open_neighbor_count(self, pos):
        open_count = 0
        for dx, dz in DIR_MAP.values():
            nxt = (pos[0] + dx, pos[1] + dz)
            if self._local_cell(nxt) not in (1, 2):
                continue

            if self._cfg("AVOID_DIAGONAL_CORNER_CUT", True) and dx != 0 and dz != 0:
                side_x = (pos[0] + dx, pos[1])
                side_z = (pos[0], pos[1] + dz)
                if self._local_cell(side_x) not in (1, 2) or self._local_cell(side_z) not in (1, 2):
                    continue

            open_count += 1
        return open_count

    def _visible_dirt_exists(self):
        view_map = getattr(self.preprocessor, "_view_map", None)
        return view_map is not None and bool(np.any(view_map == 2))

    def _visible_dirt_positions(self):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        view_map = getattr(self.preprocessor, "_view_map", None)
        if cur_pos is None or view_map is None:
            return []

        dirt_cells = np.argwhere(view_map == 2)
        targets = []
        center = self.preprocessor.VIEW_HALF
        for ri, ci in dirt_cells:
            targets.append((int(cur_pos[0] + int(ci) - center), int(cur_pos[1] + int(ri) - center)))
        targets.sort(key=lambda p: self._manhattan(cur_pos, p))
        return targets

    def _visible_dirt_bfs_action(self, legal_action):
        targets = self._visible_dirt_positions()
        if not targets:
            return None
        return self._bfs_first_action(targets[:40], legal_action)

    def _action_hits_dirt(self, action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return False
        return self._local_cell(self._next_pos(cur_pos, action)) == 2

    def _action_revisits_recent(self, action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return False
        nxt = self._next_pos(cur_pos, action)
        history = list(getattr(self.preprocessor, "pos_history", []))
        visit_count = getattr(self.preprocessor, "visit_count", {})
        window = int(self._cfg("RECENT_REVISIT_WINDOW", 10))
        threshold = int(self._cfg("REPEAT_VISIT_THRESHOLD", 2))
        return nxt in history[-window:] or visit_count.get(nxt, 0) >= threshold

    def _known_cell(self, pos):
        return getattr(self.preprocessor, "known_cells", {}).get(pos)

    def _is_known_passable(self, pos):
        return self._known_cell(pos) in (1, 2)

    def _is_frontier_cell(self, pos):
        if not self._is_known_passable(pos):
            return False
        known_cells = getattr(self.preprocessor, "known_cells", {})
        for dx, dz in DIR_MAP.values():
            if (pos[0] + dx, pos[1] + dz) not in known_cells:
                return True
        return False

    def _known_open_neighbor_count(self, pos):
        open_count = 0
        for dx, dz in DIR_MAP.values():
            if self._is_known_passable((pos[0] + dx, pos[1] + dz)):
                open_count += 1
        return open_count

    def _current_open_neighbor_count(self):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return 0
        return max(
            self._local_open_neighbor_count(cur_pos),
            self._known_open_neighbor_count(cur_pos),
        )

    def _narrow_area_pressure(self):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return False

        max_open = int(self._cfg("NARROW_AREA_MAX_OPEN_NEIGHBORS", 3))
        if self._current_open_neighbor_count() > max_open:
            return False

        visit_count = getattr(self.preprocessor, "visit_count", {})
        repeat_visits = int(self._cfg("NARROW_AREA_REPEAT_VISITS", 2))
        return int(visit_count.get(cur_pos, 0)) >= repeat_visits or self._is_recent_loop()

    def _repeat_area_pressure(self, legal_action=None):
        if self.force_charge_mode:
            return False

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return False

        if legal_action is None:
            legal_action = getattr(self.preprocessor, "_legal_act", None)
            if not legal_action:
                legal_action = [1] * 8

        history = list(getattr(self.preprocessor, "pos_history", []))
        visit_count = getattr(self.preprocessor, "visit_count", {})
        if len(history) >= 12:
            unique_limit = int(self._cfg("REPEAT_PRESSURE_RECENT_UNIQUE_CELLS", 5))
            if len(set(history[-12:])) <= unique_limit:
                return True

        if int(visit_count.get(cur_pos, 0)) >= int(self._cfg("REPEAT_PRESSURE_CUR_VISITS", 3)):
            return True

        passable_actions = [
            action
            for action in self._legal_actions(legal_action)
            if self._action_is_passable(cur_pos, action)
        ]
        if not passable_actions:
            return False

        fresh_count = 0
        repeated_count = 0
        for action in passable_actions:
            nxt = self._next_pos(cur_pos, action)
            if int(visit_count.get(nxt, 0)) <= 0:
                fresh_count += 1
            else:
                repeated_count += 1

        return fresh_count == 0 and repeated_count >= 2

    def _clean_phase_repeat_ratio(self):
        clean_steps = int(getattr(self, "clean_phase_steps", 0))
        if clean_steps <= 0:
            return 0.0
        return float(getattr(self, "clean_phase_repeat_steps", 0)) / float(clean_steps)

    def _overall_repeat_ratio(self):
        visit_count = getattr(self.preprocessor, "visit_count", {})
        total_visits = sum(int(v) for v in visit_count.values())
        if total_visits <= 0:
            return 0.0
        return 1.0 - float(len(visit_count)) / float(total_visits)

    def _clean_repeat_pressure(self):
        if self.force_charge_mode:
            return False

        clean_steps = int(getattr(self, "clean_phase_steps", 0))
        min_steps = int(self._cfg("CLEAN_REPEAT_PRESSURE_MIN_STEPS", 180))
        clean_ratio = self._clean_phase_repeat_ratio()
        stall_steps = int(getattr(self.preprocessor, "steps_since_clean", 0))
        no_visible_dirt = not self._visible_dirt_exists()
        hard_ratio = float(self._cfg("CLEAN_REPEAT_PRESSURE_HARD_RATIO", 0.18))
        if clean_ratio >= hard_ratio:
            return True

        if clean_steps < max(1, min_steps):
            stall_min_steps = int(self._cfg("CLEAN_REPEAT_STALL_MIN_STEPS", 24))
            stall_ratio = float(self._cfg("CLEAN_REPEAT_STALL_RATIO", 0.14))
            return bool(
                no_visible_dirt
                and stall_steps >= max(1, stall_min_steps)
                and clean_ratio >= stall_ratio
            )

        pressure_ratio = float(self._cfg("CLEAN_REPEAT_PRESSURE_RATIO", 0.12))
        overall_ratio = self._overall_repeat_ratio()
        overall_pressure = float(self._cfg("OVERALL_REPEAT_PRESSURE_RATIO", 0.22))
        stall_min_steps = int(self._cfg("CLEAN_REPEAT_STALL_MIN_STEPS", 24))
        stall_ratio = float(self._cfg("CLEAN_REPEAT_STALL_RATIO", pressure_ratio))
        if no_visible_dirt and stall_steps >= max(1, stall_min_steps) and clean_ratio >= stall_ratio:
            return True
        return clean_ratio >= pressure_ratio and overall_ratio >= overall_pressure

    def _clean_repeat_hard_pressure(self):
        if self.force_charge_mode:
            return False

        clean_steps = int(getattr(self, "clean_phase_steps", 0))
        min_steps = int(self._cfg("CLEAN_REPEAT_PRESSURE_MIN_STEPS", 180))
        hard_ratio = float(self._cfg("CLEAN_REPEAT_PRESSURE_HARD_RATIO", 0.18))
        clean_ratio = self._clean_phase_repeat_ratio()
        stall_steps = int(getattr(self.preprocessor, "steps_since_clean", 0))
        no_visible_dirt = not self._visible_dirt_exists()
        hard_stall_min_steps = int(self._cfg("CLEAN_REPEAT_HARD_STALL_MIN_STEPS", 40))
        hard_stall_ratio = float(self._cfg("CLEAN_REPEAT_HARD_STALL_RATIO", 0.16))
        if clean_ratio >= hard_ratio:
            return True
        if clean_steps < max(1, min_steps):
            return bool(
                no_visible_dirt
                and stall_steps >= max(1, hard_stall_min_steps)
                and clean_ratio >= hard_stall_ratio
            )
        return bool(
            no_visible_dirt
            and stall_steps >= max(1, hard_stall_min_steps)
            and clean_ratio >= hard_stall_ratio
        )

    def _action_under_clean_repeat_pressure(self, action):
        if not self._clean_repeat_hard_pressure():
            return False

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return False

        nxt = self._next_pos(cur_pos, action)
        visit_count = getattr(self.preprocessor, "visit_count", {})
        return (
            int(visit_count.get(nxt, 0)) > 0
            or self._action_revisits_recent(action)
            or self._post_charge_route_revisit(nxt)
        )

    def _recent_route_revisit_ratio(self, route_history=None, window=None):
        if route_history is None:
            route_history = list(getattr(self, "recent_charge_route_cells", []))
        if window is not None:
            route_history = route_history[-max(1, int(window)) :]
        route_len = len(route_history)
        if route_len <= 0:
            return 0.0
        return 1.0 - float(len(set(route_history))) / float(route_len)

    def _charge_route_repeat_pressure(self):
        if not self.force_charge_mode:
            return False

        window = int(self._cfg("CHARGE_ROUTE_REPEAT_WINDOW", 24))
        unique_limit = int(self._cfg("CHARGE_ROUTE_REPEAT_UNIQUE_CELLS", 10))
        min_steps = int(self._cfg("CHARGE_ROUTE_REPEAT_MIN_STEPS", 12))
        ratio_threshold = float(self._cfg("CHARGE_ROUTE_REPEAT_RATIO_THRESHOLD", 0.34))
        route_history = list(getattr(self, "recent_charge_route_cells", []))[-max(1, window):]
        route_len = len(route_history)
        if route_len < max(8, min_steps):
            return False

        revisit_ratio = self._recent_route_revisit_ratio(route_history)
        return len(set(route_history)) <= unique_limit or revisit_ratio >= ratio_threshold

    def _charge_final_capture_active(self, path_charger_dist=None):
        if not self.force_charge_mode:
            return False

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return False

        if path_charger_dist is None:
            path_charger_dist = self.current_charger_path_dist
        if path_charger_dist is None:
            path_charger_dist = self._charger_route_distance(getattr(self.preprocessor, "charger_list", []))

        final_path_max = int(self._cfg("CHARGE_FINAL_CAPTURE_PATH_MAX", 2))
        if path_charger_dist is not None and int(path_charger_dist) <= final_path_max:
            return True

        return bool(self._in_charger_zone(cur_pos))

    def _charge_mid_route_loop_pressure(self, path_charger_dist=None):
        if not self.force_charge_mode:
            return False

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None or self._in_charger_zone(cur_pos):
            return False

        if path_charger_dist is None:
            path_charger_dist = self.current_charger_path_dist
        if path_charger_dist is None:
            path_charger_dist = self._charger_route_distance(getattr(self.preprocessor, "charger_list", []))
        if path_charger_dist is None:
            return False

        dist = int(path_charger_dist)
        if dist < int(self._cfg("CHARGE_MID_ROUTE_LOOP_MIN_PATH", 8)):
            return False
        if dist > int(self._cfg("CHARGE_MID_ROUTE_LOOP_MAX_PATH", 15)):
            return False

        route_steps = int(getattr(self, "current_charge_route_steps", 0))
        if route_steps < int(self._cfg("CHARGE_MID_ROUTE_LOOP_MIN_STEPS", 48)):
            return False

        route_ratio = float(self._recent_route_revisit_ratio())
        if route_ratio >= float(self._cfg("CHARGE_MID_ROUTE_LOOP_RATIO", 0.34)):
            return True

        no_progress_steps = int(getattr(self, "force_charge_no_progress_steps", 0))
        return bool(
            no_progress_steps >= int(self._cfg("CHARGE_MID_ROUTE_LOOP_NO_PROGRESS_STEPS", 4))
            and route_ratio >= float(self._cfg("CHARGE_MID_ROUTE_LOOP_PROGRESS_RATIO", 0.25))
        )

    def _update_charge_route_memory(self):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return

        step_no = getattr(self.preprocessor, "step_no", None)
        if step_no is not None and step_no == self._last_phase_update_step_no:
            return
        self._last_phase_update_step_no = step_no

        in_zone = self._in_charger_zone(cur_pos)
        is_charging = bool(getattr(self.preprocessor, "is_charging", False))
        battery_gained = self._battery_gained_last_step()
        if self._flag("ENABLE_SIMPLE_CHARGE_MODE", False):
            charge_context = bool(self.force_charge_mode or is_charging or in_zone)
            cooldown = int(self._cfg("POST_CHARGE_CLEAN_COOLDOWN_STEPS", 120))
            leaving_charge_context = bool(self._last_charge_context and not charge_context and not self.force_charge_mode)
            route_memory_enabled = self._flag("ENABLE_SIMPLE_CHARGE_ROUTE_MEMORY", False)

            charger_list = getattr(self.preprocessor, "charger_list", [])
            route_targets = self._charger_zone_targets() or charger_list
            # v22: simple charge 不再把 direct path 用作动作接管，
            # 但允许它作为“回充进展/奖励/日志”信号。
            # v20/v21 的坏样本里 agent_current_path 经常是 99，而 workflow_direct_path 有效，
            # force_charge_no_progress_steps 因此失真，回充路线重复也无法被稳定识别。
            if route_targets and self._flag("ENABLE_SIMPLE_CHARGE_DIRECT_PROGRESS_DIST", False):
                current_path_dist = self._charger_route_distance_direct(route_targets)
            else:
                current_path_dist = self._charger_route_distance(route_targets) if route_targets else None
            self.last_charger_path_dist = self.current_charger_path_dist
            self.current_charger_path_dist = current_path_dist

            if self.force_charge_mode and not in_zone and not is_charging:
                if route_memory_enabled:
                    self.recent_charge_route_cells.append(cur_pos)
                self.force_charge_search_steps += 1
                self.current_charge_route_steps += 1
                if current_path_dist is None or self.last_charger_path_dist is None:
                    self.force_charge_no_progress_steps += 1
                elif int(current_path_dist) >= int(self.last_charger_path_dist):
                    self.force_charge_no_progress_steps += 1
                else:
                    self.force_charge_no_progress_steps = 0
            elif not self.force_charge_mode and not is_charging and not in_zone:
                self.clean_phase_steps += 1
                if cur_pos in self.clean_phase_seen:
                    self.clean_phase_repeat_steps += 1
                self.clean_phase_seen.add(cur_pos)
                self.force_charge_search_steps = 0
                self.force_charge_no_progress_steps = 0
            else:
                self.force_charge_no_progress_steps = 0

            if self.force_charge_mode and in_zone and not battery_gained:
                self.charge_zone_no_gain_steps += 1
            else:
                self.charge_zone_no_gain_steps = 0

            if self.force_charge_mode or is_charging or in_zone:
                if battery_gained:
                    self.current_charge_gain_steps += 1
                self.episode_charge_gain_steps += 1
                if in_zone or is_charging:
                    self.current_charge_zone_steps += 1

            if charge_context:
                self.post_charge_clean_steps = max(int(self.post_charge_clean_steps), cooldown)
            elif self._last_charge_context and not self.force_charge_mode:
                self.post_charge_clean_steps = max(int(self.post_charge_clean_steps), cooldown)
            elif self.post_charge_clean_steps > 0:
                self.post_charge_clean_steps -= 1

            if leaving_charge_context:
                route_ratio = float(self._recent_route_revisit_ratio()) if route_memory_enabled else 0.0
                route_steps = int(self.current_charge_route_steps)
                self.last_charge_route_steps = int(self.current_charge_route_steps)
                self.last_charge_zone_steps = int(self.current_charge_zone_steps)
                self.last_charge_gain_steps = int(self.current_charge_gain_steps)
                self.current_charge_route_steps = 0
                self.current_charge_zone_steps = 0
                self.current_charge_gain_steps = 0
                if (
                    route_memory_enabled
                    and route_steps >= int(self._cfg("POST_CHARGE_ESCAPE_MIN_ROUTE_STEPS", 48))
                    and route_ratio >= float(self._cfg("POST_CHARGE_ESCAPE_ROUTE_RATIO", 0.38))
                ):
                    self.post_charge_escape_steps = max(
                        int(self.post_charge_escape_steps),
                        int(self._cfg("POST_CHARGE_ESCAPE_STEPS", 96)),
                    )
                    self.post_charge_escape_ratio = max(float(self.post_charge_escape_ratio), route_ratio)
            elif not charge_context and self.post_charge_escape_steps > 0:
                self.post_charge_escape_steps -= 1
                if self.post_charge_escape_steps <= 0:
                    self.post_charge_escape_ratio = 0.0

            if charge_context or self.force_charge_mode:
                self.clean_escape_steps = 0
                self.clean_escape_ratio = 0.0
            else:
                clean_escape_window = int(self._cfg("CLEAN_ESCAPE_WINDOW", 24))
                traj_revisit_ratio = self._recent_traj_revisit_ratio(window=clean_escape_window)
                unique_ratio = self._recent_traj_unique_ratio(window=clean_escape_window)
                clean_repeat_ratio = float(self._clean_phase_repeat_ratio())
                min_steps_since_clean = int(self._cfg("CLEAN_ESCAPE_MIN_STEPS_SINCE_CLEAN", 28))
                should_start_clean_escape = (
                    not self._visible_dirt_exists()
                    and int(getattr(self.preprocessor, "steps_since_clean", 0)) >= min_steps_since_clean
                    and (
                        clean_repeat_ratio >= float(self._cfg("CLEAN_ESCAPE_CLEAN_REPEAT_RATIO", 0.16))
                        or (
                            traj_revisit_ratio >= float(self._cfg("CLEAN_ESCAPE_TRAJ_REVISIT_RATIO", 0.42))
                            and unique_ratio <= float(self._cfg("CLEAN_ESCAPE_MAX_UNIQUE_RATIO", 0.58))
                        )
                    )
                )
                if should_start_clean_escape:
                    self.clean_escape_steps = max(
                        int(self.clean_escape_steps),
                        int(self._cfg("CLEAN_ESCAPE_STEPS", 20)),
                    )
                    self.clean_escape_ratio = max(
                        float(self.clean_escape_ratio),
                        max(float(traj_revisit_ratio), float(clean_repeat_ratio)),
                    )
                elif self.clean_escape_steps > 0:
                    self.clean_escape_steps -= 1
                    if self.clean_escape_steps <= 0:
                        self.clean_escape_ratio = 0.0

            self._last_charge_context = charge_context
            return


        charge_context = bool(self.force_charge_mode or is_charging or in_zone)
        cooldown = int(self._cfg("POST_CHARGE_CLEAN_COOLDOWN_STEPS", 120))
        leaving_charge_context = bool(self._last_charge_context and not charge_context and not self.force_charge_mode)
        charger_list = getattr(self.preprocessor, "charger_list", [])
        current_path_dist = self._charger_route_distance(charger_list) if charger_list else None
        self.last_charger_path_dist = self.current_charger_path_dist
        self.current_charger_path_dist = current_path_dist

        if self.force_charge_mode and not in_zone:
            self.recent_charge_route_cells.append(cur_pos)
            self.force_charge_search_steps += 1
            self.current_charge_route_steps += 1
            if current_path_dist is None or self.last_charger_path_dist is None:
                self.force_charge_no_progress_steps += 1
            elif int(current_path_dist) >= int(self.last_charger_path_dist):
                self.force_charge_no_progress_steps += 1
            else:
                self.force_charge_no_progress_steps = 0
        elif not self.force_charge_mode and not is_charging and not in_zone:
            self.clean_phase_steps += 1
            if cur_pos in self.clean_phase_seen:
                self.clean_phase_repeat_steps += 1
            self.clean_phase_seen.add(cur_pos)
            self.force_charge_search_steps = 0
            self.force_charge_no_progress_steps = 0
        else:
            self.force_charge_search_steps = 0
            self.force_charge_no_progress_steps = 0

        if self.force_charge_mode and in_zone and not battery_gained:
            self.charge_zone_no_gain_steps += 1
        else:
            self.charge_zone_no_gain_steps = 0

        if self.force_charge_mode or is_charging or in_zone:
            if battery_gained:
                self.current_charge_gain_steps += 1
                self.episode_charge_gain_steps += 1
            if self.force_charge_mode and not in_zone:
                pass
            else:
                self.current_charge_zone_steps += 1

        if is_charging or in_zone:
            self.post_charge_clean_steps = max(int(self.post_charge_clean_steps), cooldown)
        elif self._last_charge_context and not self.force_charge_mode:
            self.post_charge_clean_steps = max(int(self.post_charge_clean_steps), cooldown)
        elif self.post_charge_clean_steps > 0:
            self.post_charge_clean_steps -= 1

        if leaving_charge_context:
            route_ratio = float(self._recent_route_revisit_ratio())
            route_steps = len(getattr(self, "recent_charge_route_cells", []))
            self.last_charge_route_steps = int(self.current_charge_route_steps)
            self.last_charge_zone_steps = int(self.current_charge_zone_steps)
            self.last_charge_gain_steps = int(self.current_charge_gain_steps)
            self.current_charge_route_steps = 0
            self.current_charge_zone_steps = 0
            self.current_charge_gain_steps = 0
            if (
                route_steps >= int(self._cfg("POST_CHARGE_ESCAPE_MIN_ROUTE_STEPS", 72))
                and route_ratio >= float(self._cfg("POST_CHARGE_ESCAPE_ROUTE_RATIO", 0.48))
            ):
                self.post_charge_escape_steps = max(
                    int(self.post_charge_escape_steps),
                    int(self._cfg("POST_CHARGE_ESCAPE_STEPS", 96)),
                )
                self.post_charge_escape_ratio = max(float(self.post_charge_escape_ratio), route_ratio)
        elif not charge_context and self.post_charge_escape_steps > 0:
            self.post_charge_escape_steps -= 1
            if self.post_charge_escape_steps <= 0:
                self.post_charge_escape_ratio = 0.0
        elif not charge_context and not self._last_charge_context:
            if self.current_charge_route_steps > 0 or self.current_charge_zone_steps > 0:
                if self.current_charge_gain_steps > 0:
                    self.last_charge_gain_steps = int(self.current_charge_gain_steps)
                self.current_charge_route_steps = 0
                self.current_charge_zone_steps = 0
                self.current_charge_gain_steps = 0

        if (
            not charge_context
            and self._post_charge_clean_pressure()
            and int(self.post_charge_escape_steps) <= 0
        ):
            clean_repeat_ratio = float(self._clean_phase_repeat_ratio())
            steps_since_clean = int(getattr(self.preprocessor, "steps_since_clean", 0))
            total_charge_steps = int(self.last_charge_route_steps) + int(self.last_charge_zone_steps)
            early_escape = (
                total_charge_steps >= int(self._cfg("POST_CHARGE_ESCAPE_TOTAL_CHARGE_STEPS", 120))
                and clean_repeat_ratio >= float(self._cfg("POST_CHARGE_ESCAPE_EARLY_CLEAN_REPEAT_RATIO", 0.18))
            )
            overhead_escape = (
                int(self.last_charge_route_steps) >= int(self._cfg("POST_CHARGE_ESCAPE_OVERHEAD_ROUTE_STEPS", 140))
                and float(self._recent_route_revisit_ratio()) >= float(self._cfg("POST_CHARGE_ESCAPE_OVERHEAD_ROUTE_RATIO", 0.42))
                and clean_repeat_ratio >= float(self._cfg("POST_CHARGE_ESCAPE_OVERHEAD_CLEAN_REPEAT_RATIO", 0.10))
            )
            mid_repeat_escape = (
                int(self.last_charge_route_steps) >= int(self._cfg("POST_CHARGE_ESCAPE_MID_ROUTE_STEPS", 84))
                and float(self._recent_route_revisit_ratio()) >= float(self._cfg("POST_CHARGE_ESCAPE_MID_ROUTE_RATIO", 0.36))
                and (
                    clean_repeat_ratio >= float(self._cfg("POST_CHARGE_ESCAPE_MID_CLEAN_REPEAT_RATIO", 0.12))
                    or steps_since_clean >= int(self._cfg("POST_CHARGE_ESCAPE_MID_STEPS_SINCE_CLEAN", 6))
                )
            )
            if early_escape or overhead_escape or mid_repeat_escape:
                self.post_charge_escape_steps = max(
                    int(self.post_charge_escape_steps),
                    int(self._cfg("POST_CHARGE_ESCAPE_STEPS", 96)),
                )
                self.post_charge_escape_ratio = max(
                    float(self.post_charge_escape_ratio),
                    max(float(self._recent_route_revisit_ratio()), clean_repeat_ratio),
                )

        if charge_context or self.force_charge_mode:
            self.clean_escape_steps = 0
            self.clean_escape_ratio = 0.0
        else:
            clean_escape_window = int(self._cfg("CLEAN_ESCAPE_WINDOW", 24))
            traj_revisit_ratio = self._recent_traj_revisit_ratio(window=clean_escape_window)
            unique_ratio = self._recent_traj_unique_ratio(window=clean_escape_window)
            clean_repeat_ratio = float(self._clean_phase_repeat_ratio())
            min_steps_since_clean = int(self._cfg("CLEAN_ESCAPE_MIN_STEPS_SINCE_CLEAN", 28))
            should_start_clean_escape = (
                not self._visible_dirt_exists()
                and (
                    (
                        int(getattr(self.preprocessor, "steps_since_clean", 0)) >= min_steps_since_clean
                        and (
                            clean_repeat_ratio >= float(self._cfg("CLEAN_ESCAPE_CLEAN_REPEAT_RATIO", 0.16))
                            or (
                                traj_revisit_ratio >= float(self._cfg("CLEAN_ESCAPE_TRAJ_REVISIT_RATIO", 0.42))
                                and unique_ratio <= float(self._cfg("CLEAN_ESCAPE_MAX_UNIQUE_RATIO", 0.58))
                            )
                        )
                    )
                    or (
                        int(getattr(self, "clean_phase_steps", 0)) >= int(self._cfg("CLEAN_ESCAPE_HARD_MIN_CLEAN_STEPS", 220))
                        and clean_repeat_ratio >= float(self._cfg("CLEAN_ESCAPE_HARD_CLEAN_REPEAT_RATIO", 0.22))
                        and traj_revisit_ratio >= float(self._cfg("CLEAN_ESCAPE_HARD_TRAJ_REVISIT_RATIO", 0.48))
                        and unique_ratio <= float(self._cfg("CLEAN_ESCAPE_HARD_MAX_UNIQUE_RATIO", 0.48))
                    )
                )
            )
            if should_start_clean_escape:
                self.clean_escape_steps = max(
                    int(self.clean_escape_steps),
                    int(self._cfg("CLEAN_ESCAPE_STEPS", 20)),
                )
                self.clean_escape_ratio = max(
                    float(self.clean_escape_ratio),
                    max(float(traj_revisit_ratio), float(clean_repeat_ratio)),
                )
            elif self.clean_escape_steps > 0:
                self.clean_escape_steps -= 1
                if self.clean_escape_steps <= 0:
                    self.clean_escape_ratio = 0.0

        self._last_charge_context = charge_context

    def _post_charge_clean_pressure(self):
        if not self._post_charge_guidance_enabled():
            return False
        if self.force_charge_mode:
            return False

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        battery = getattr(self.preprocessor, "battery", None)
        battery_max = getattr(self.preprocessor, "battery_max", None)
        if cur_pos is None or battery is None or battery_max is None or battery_max <= 0:
            return False

        battery_ratio = float(battery) / float(battery_max)
        if battery_ratio < float(self._cfg("POST_CHARGE_MIN_BATTERY_RATIO", 0.32)):
            return False

        if int(getattr(self, "post_charge_clean_steps", 0)) > 0:
            return True

        zone_dist = self._charger_zone_dist(cur_pos)
        if zone_dist is None:
            return False
        return int(zone_dist) <= int(self._cfg("POST_CHARGE_NEAR_CHARGER_DIST", 4))

    def _post_charge_escape_pressure(self):
        if not self._post_charge_escape_enabled():
            return False
        return self._post_charge_clean_pressure() and int(getattr(self, "post_charge_escape_steps", 0)) > 0

    def _post_charge_route_revisit(self, pos):
        route_cells = set(getattr(self, "recent_charge_route_cells", []))
        if pos not in route_cells:
            return False

        if not self._flag("ENABLE_SIMPLE_CHARGE_MODE", False):
            return True

        # v17：simple charge 的短路线记忆只在“刚充完需要离开旧路线”时生效。
        # 不在清扫全局长期惩罚这条路线，避免把通往远处污渍/充电桩的必要通道罚死。
        if not self._flag("ENABLE_SIMPLE_CHARGE_ROUTE_MEMORY", False):
            return False
        if self.force_charge_mode:
            return False
        return bool(self._post_charge_clean_pressure() or self._post_charge_escape_pressure())

    def _recent_traj_history(self, window=None):
        history = list(getattr(self.preprocessor, "pos_history", []))
        if window is not None:
            history = history[-max(1, int(window)) :]
        return history

    def _recent_traj_revisit_ratio(self, window=None):
        history = self._recent_traj_history(window=window)
        if not history:
            return 0.0
        return 1.0 - float(len(set(history))) / float(len(history))

    def _recent_traj_unique_ratio(self, window=None):
        history = self._recent_traj_history(window=window)
        if not history:
            return 0.0
        return float(len(set(history))) / float(len(history))

    def _projected_npc_tracks(self, steps=None):
        npc_list = list(getattr(self.preprocessor, "npc_list", []))
        if not npc_list:
            return []

        recent_positions = list(getattr(self.preprocessor, "recent_npc_positions", []))
        if len(recent_positions) >= 2:
            prev_positions = list(recent_positions[-2])
        elif recent_positions:
            prev_positions = list(recent_positions[-1])
        else:
            prev_positions = []
        prev_pool = list(prev_positions)
        grid_size = int(getattr(self.preprocessor, "GRID_SIZE", 128))
        steps = max(1, int(steps or self._cfg("NPC_PROJECT_STEPS", 3)))
        decay = float(self._cfg("NPC_PROJECT_DECAY", 0.72))
        tracks = []

        for cur in npc_list:
            cur = (int(cur[0]), int(cur[1]))
            prev = cur
            if prev_pool:
                prev = min(prev_pool, key=lambda p: self._manhattan(p, cur))
                prev_pool.remove(prev)
            vel_x = int(cur[0]) - int(prev[0])
            vel_z = int(cur[1]) - int(prev[1])
            for step in range(1, steps + 1):
                px = min(grid_size - 1, max(0, int(cur[0] + vel_x * step)))
                pz = min(grid_size - 1, max(0, int(cur[1] + vel_z * step)))
                tracks.append(((px, pz), float(decay ** (step - 1))))

        return tracks

    def _projected_npc_dist_to_pos(self, pos, tracks=None):
        if pos is None:
            return None
        tracks = self._projected_npc_tracks() if tracks is None else tracks
        if not tracks:
            return None
        return min(self._manhattan(pos, npc_pos) for npc_pos, _ in tracks)

    def _projected_npc_heat(self, pos, tracks=None, scale=None):
        if pos is None:
            return 0.0
        tracks = self._projected_npc_tracks() if tracks is None else tracks
        if not tracks:
            return 0.0
        scale = float(scale or self._cfg("NPC_PROJECT_HEAT_LOCAL_SCALE", 6.0))
        best = 0.0
        for npc_pos, weight in tracks:
            best = max(best, float(weight) * self._proximity_heat(self._manhattan(pos, npc_pos), scale))
        return float(best)

    def _clean_escape_pressure(self):
        return (not self.force_charge_mode) and int(getattr(self, "clean_escape_steps", 0)) > 0

    def _augmented_local_memory_feature(self):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return np.zeros(Config.AUGMENTED_LOCAL_LEN, dtype=np.float32)

        size = int(getattr(Config, "AUGMENTED_LOCAL_SIZE", 7))
        half = size // 2
        known_cells = getattr(self.preprocessor, "known_cells", {})
        known_dirt = getattr(self.preprocessor, "known_dirt_positions", set())
        visit_count = getattr(self.preprocessor, "visit_count", {})
        charger_list = getattr(self.preprocessor, "charger_list", [])
        npc_list = getattr(self.preprocessor, "npc_list", [])
        suspect_cells = getattr(self.preprocessor, "suspected_npc_cleaned_cells", {})
        suspect_ttl_scale = float(max(getattr(Config, "NPC_CLEAN_MEMORY_STEPS", 220), 1))
        route_window = int(self._cfg("CHARGE_ROUTE_REPEAT_WINDOW", 24))
        route_counter = Counter(list(getattr(self, "recent_charge_route_cells", []))[-max(1, route_window):])
        traj_counter = Counter(self._recent_traj_history(window=16))
        clean_phase_seen = getattr(self, "clean_phase_seen", set())
        projected_tracks = self._projected_npc_tracks()

        channels = []
        for dz in range(-half, half + 1):
            for dx in range(-half, half + 1):
                pos = (cur_pos[0] + dx, cur_pos[1] + dz)
                known_cell = known_cells.get(pos)
                local_cell = self._local_cell(pos)
                effective_cell = known_cell if known_cell is not None else local_cell

                passable_known = 1.0 if effective_cell in (1, 2) else 0.0
                unknown = 1.0 if pos not in known_cells else 0.0
                dirt = 1.0 if (pos in known_dirt or effective_cell == 2) else 0.0
                visit_heat = min(float(visit_count.get(pos, 0)), 4.0) / 4.0
                frontier = 1.0 if pos in known_cells and self._is_frontier_cell(pos) else 0.0
                charger_heat = self._proximity_heat(self._charger_zone_dist(pos), 4.0)
                npc_heat = self._proximity_heat(self._min_dist(pos, npc_list) if npc_list else None, 6.0)
                projected_npc_heat = self._projected_npc_heat(
                    pos,
                    tracks=projected_tracks,
                    scale=self._cfg("NPC_PROJECT_HEAT_LOCAL_SCALE", 6.0),
                )
                suspect_heat = min(float(suspect_cells.get(pos, 0)), suspect_ttl_scale) / suspect_ttl_scale
                route_heat = min(float(route_counter.get(pos, 0)), 4.0) / 4.0
                traj_heat = min(float(traj_counter.get(pos, 0)), 3.0) / 3.0
                clean_seen_heat = 1.0 if pos in clean_phase_seen and dirt <= 0.0 else 0.0

                channels.extend([
                    passable_known,
                    unknown,
                    dirt,
                    visit_heat,
                    frontier,
                    charger_heat,
                    npc_heat,
                    projected_npc_heat,
                    suspect_heat,
                    route_heat,
                    traj_heat,
                    clean_seen_heat,
                ])

        feature = np.asarray(channels, dtype=np.float32)
        if feature.shape[0] != int(getattr(Config, "AUGMENTED_LOCAL_LEN", feature.shape[0])):
            raise ValueError(f"unexpected augmented local feature length: {feature.shape[0]}")
        return feature

    def _augmented_global_memory_feature(self):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return np.zeros(Config.AUGMENTED_GLOBAL_LEN, dtype=np.float32)

        size = int(getattr(Config, "AUGMENTED_GLOBAL_SIZE", 16))
        grid_size = int(getattr(self.preprocessor, "GRID_SIZE", 128))
        cell_span = max(1, grid_size // max(size, 1))
        block_area = float(cell_span * cell_span)
        known_cells = getattr(self.preprocessor, "known_cells", {})
        known_dirt = getattr(self.preprocessor, "known_dirt_positions", set())
        visit_count = getattr(self.preprocessor, "visit_count", {})
        charger_list = getattr(self.preprocessor, "charger_list", [])
        npc_list = getattr(self.preprocessor, "npc_list", [])
        suspect_cells = getattr(self.preprocessor, "suspected_npc_cleaned_cells", {})
        route_history = list(getattr(self, "recent_charge_route_cells", []))
        traj_history = self._recent_traj_history(window=24)
        clean_phase_seen = getattr(self, "clean_phase_seen", set())
        projected_tracks = self._projected_npc_tracks()

        known_count = np.zeros((size, size), dtype=np.float32)
        passable_count = np.zeros((size, size), dtype=np.float32)
        dirt_count = np.zeros((size, size), dtype=np.float32)
        clean_count = np.zeros((size, size), dtype=np.float32)
        visit_sum = np.zeros((size, size), dtype=np.float32)
        frontier_count = np.zeros((size, size), dtype=np.float32)
        suspect_count = np.zeros((size, size), dtype=np.float32)
        route_count = np.zeros((size, size), dtype=np.float32)
        traj_count = np.zeros((size, size), dtype=np.float32)
        clean_seen_count = np.zeros((size, size), dtype=np.float32)

        def bucket(pos):
            bx = min(size - 1, max(0, int(pos[0]) * size // max(grid_size, 1)))
            bz = min(size - 1, max(0, int(pos[1]) * size // max(grid_size, 1)))
            return bx, bz

        for pos, cell in known_cells.items():
            bx, bz = bucket(pos)
            known_count[bz, bx] += 1.0
            if cell in (1, 2):
                passable_count[bz, bx] += 1.0
            if cell == 2 or pos in known_dirt:
                dirt_count[bz, bx] += 1.0
            elif cell == 1:
                clean_count[bz, bx] += 1.0
            visit_sum[bz, bx] += float(min(visit_count.get(pos, 0), 4))
            if self._is_frontier_cell(pos):
                frontier_count[bz, bx] += 1.0
            if suspect_cells.get(pos, 0) > 0:
                suspect_count[bz, bx] += 1.0
            if pos in clean_phase_seen and cell != 2:
                clean_seen_count[bz, bx] += 1.0

        for pos in route_history:
            bx, bz = bucket(pos)
            route_count[bz, bx] += 1.0
        for pos in traj_history:
            bx, bz = bucket(pos)
            traj_count[bz, bx] += 1.0

        passable_density = np.clip(passable_count / block_area, 0.0, 1.0)
        unknown_density = np.clip(1.0 - known_count / block_area, 0.0, 1.0)
        dirt_density = np.clip(dirt_count / block_area, 0.0, 1.0)
        clean_density = np.clip(clean_count / block_area, 0.0, 1.0)
        visit_heat = np.clip(visit_sum / block_area / 2.0, 0.0, 1.0)
        frontier_density = np.clip(frontier_count / block_area, 0.0, 1.0)
        suspect_density = np.clip(suspect_count / block_area, 0.0, 1.0)
        route_density = np.clip(route_count / max(1.0, float(len(route_history) or 1.0)), 0.0, 1.0)
        traj_density = np.clip(traj_count / max(1.0, float(len(traj_history) or 1.0)), 0.0, 1.0)
        clean_seen_density = np.clip(clean_seen_count / block_area, 0.0, 1.0)

        charger_heat = np.zeros((size, size), dtype=np.float32)
        npc_heat = np.zeros((size, size), dtype=np.float32)
        projected_npc_heat = np.zeros((size, size), dtype=np.float32)
        self_heat = np.zeros((size, size), dtype=np.float32)
        self_bx, self_bz = bucket(cur_pos)
        self_heat[self_bz, self_bx] = 1.0

        for bz in range(size):
            for bx in range(size):
                center = (
                    int((bx + 0.5) * cell_span),
                    int((bz + 0.5) * cell_span),
                )
                charger_heat[bz, bx] = self._proximity_heat(
                    self._min_dist(center, charger_list) if charger_list else None,
                    48.0,
                )
                npc_heat[bz, bx] = self._proximity_heat(
                    self._min_dist(center, npc_list) if npc_list else None,
                    36.0,
                )
                projected_npc_heat[bz, bx] = self._projected_npc_heat(
                    center,
                    tracks=projected_tracks,
                    scale=self._cfg("NPC_PROJECT_HEAT_GLOBAL_SCALE", 36.0),
                )

        feature = np.stack(
            [
                passable_density,
                unknown_density,
                dirt_density,
                clean_density,
                visit_heat,
                frontier_density,
                charger_heat,
                npc_heat,
                projected_npc_heat,
                suspect_density,
                route_density,
                traj_density,
                clean_seen_density,
                self_heat,
            ],
            axis=0,
        ).astype(np.float32)
        feature = feature.reshape(-1)
        if feature.shape[0] != int(getattr(Config, "AUGMENTED_GLOBAL_LEN", feature.shape[0])):
            raise ValueError(f"unexpected augmented global feature length: {feature.shape[0]}")
        return feature


    def _valid_scalar_dist(self, dist):
        """清洗 observation scalar 里使用的距离字段；99/None/负数都视为未知。"""
        if dist is None:
            return None
        try:
            dist = int(float(dist))
        except (TypeError, ValueError):
            return None
        if dist < 0 or dist >= 99:
            return None
        return dist

    def _scalar_safe_charger_dist(self, cur_pos=None, charger_list=None):
        """
        给 scalar observation 使用的安全充电距离。

        优先复用 preprocessor reward 已经算出的 safe distance，避免重复 BFS；
        如果没有，则在低电/预回充/force_charge 阶段尝试算一次路径距离；
        最后退回最近充电桩曼哈顿距离。这里只是 observation，不改动作。
        """
        cur_pos = getattr(self.preprocessor, "cur_pos", None) if cur_pos is None else cur_pos
        charger_list = getattr(self.preprocessor, "charger_list", []) if charger_list is None else charger_list

        for attr in ("reward_safe_charger_dist", "reward_direct_charger_path_dist"):
            valid = self._valid_scalar_dist(getattr(self.preprocessor, attr, None))
            if valid is not None:
                return valid

        valid = self._valid_scalar_dist(getattr(self, "current_charger_path_dist", None))
        if valid is not None:
            return valid

        if cur_pos is None or not charger_list:
            return None

        battery = float(getattr(self.preprocessor, "battery", 0.0))
        battery_max = float(max(getattr(self.preprocessor, "battery_max", 1), 1))
        battery_ratio = battery / battery_max
        pre_charge_start = float(self._cfg("PRE_CHARGE_START_RATIO", 0.55))
        risk_start = float(self._cfg("BATTERY_RISK_START_RATIO", 0.45))
        should_try_path = bool(
            self.force_charge_mode
            or battery_ratio < pre_charge_start
            or battery_ratio < risk_start
        )

        if should_try_path and hasattr(self, "_bfs_distance_to_targets"):
            try:
                targets = self._charger_zone_targets() or self._npc_safe_charger_targets(charger_list) or charger_list
                path_dist = self._valid_scalar_dist(
                    self._bfs_distance_to_targets(
                        targets,
                        max_nodes=int(self._cfg("CHARGE_ROUTE_BFS_LIMIT", 2600)),
                    )
                )
                if path_dist is not None:
                    return path_dist
            except Exception:
                pass

        return self._valid_scalar_dist(self._min_dist(cur_pos, charger_list))

    def _scalar_pre_charge_active(self, battery_ratio, safe_dist):
        """预回充阶段标记：只作为模型输入，不触发强制回充。"""
        if safe_dist is None:
            return 0.0
        pre_charge_start = float(self._cfg("PRE_CHARGE_START_RATIO", 0.55))
        pre_charge_end = float(self._cfg("PRE_CHARGE_END_RATIO", 0.30))
        min_dist = int(self._cfg("PRE_CHARGE_MIN_DIST", 10))
        return 1.0 if (pre_charge_end <= battery_ratio < pre_charge_start and int(safe_dist) >= min_dist) else 0.0

    def _scalar_battery_risk_active(self, battery_ratio):
        """低电风险阶段标记：让模型直接看见当前是否进入低电风险区。"""
        return 1.0 if battery_ratio < float(self._cfg("BATTERY_RISK_START_RATIO", 0.45)) else 0.0

    def _scalar_charger_dist_delta_norm(self, safe_dist):
        """
        充电距离变化特征，正数表示这一步更靠近充电桩，负数表示远离。
        使用 safe_charger_dist_history 的前一个距离，避免 reward_process 更新 last 后 delta 恒为 0。
        """
        safe_dist = self._valid_scalar_dist(safe_dist)
        if safe_dist is None:
            return 0.0

        hist = list(getattr(self.preprocessor, "safe_charger_dist_history", []))
        prev_dist = None
        if len(hist) >= 2:
            prev_dist = self._valid_scalar_dist(hist[-2])
        elif len(hist) == 1:
            prev_dist = self._valid_scalar_dist(hist[-1])

        if prev_dist is None:
            return 0.0

        scale = float(max(self._cfg("SCALAR_CHARGER_DIST_DELTA_NORM", 16.0), 1.0))
        return float(np.clip((float(prev_dist) - float(safe_dist)) / scale, -1.0, 1.0))

    def _scalar_short_loop_score(self):
        """
        最近短周期循环强度：ABAB、小三角、窗口内重复。
        只作为 observation，让模型看见“自己在来回走”，不替模型改动作。
        """
        hist = list(getattr(self.preprocessor, "pos_history", []))
        if len(hist) < 3:
            return 0.0

        cur_pos = hist[-1]
        score = 0.0
        if len(hist) >= 3 and cur_pos == hist[-3]:
            score += 1.0
        if len(hist) >= 4 and cur_pos == hist[-4]:
            score += 0.75

        window = max(4, int(self._cfg("SHORT_LOOP_WINDOW", 8)))
        recent = hist[-window:]
        revisit_hits = max(0, recent.count(cur_pos) - 1)
        if revisit_hits > 0:
            score += min(1.25, float(revisit_hits) / 2.0)

        norm = float(max(self._cfg("SCALAR_SHORT_LOOP_NORM", 3.0), 1.0))
        return float(np.clip(score / norm, 0.0, 1.0))

    def _scalar_battery_no_progress_steps_norm(self):
        """
        最近充电距离连续没有下降的步数归一化。
        只依赖 preprocessor 里 reward-safe 距离历史，不影响动作。
        """
        hist = [self._valid_scalar_dist(v) for v in list(getattr(self.preprocessor, "safe_charger_dist_history", []))]
        hist = [v for v in hist if v is not None]
        if len(hist) < 2:
            return 0.0

        no_progress = 0
        # 从最新往前数，只要没有更靠近（dist 没下降）就累计。
        for idx in range(len(hist) - 1, 0, -1):
            if int(hist[idx]) >= int(hist[idx - 1]):
                no_progress += 1
            else:
                break

        norm = float(max(self._cfg("SCALAR_BATTERY_NO_PROGRESS_NORM", 8.0), 1.0))
        return float(np.clip(float(no_progress) / norm, 0.0, 1.0))


    def _augmented_scalar_memory_feature(self):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return np.zeros(Config.AUGMENTED_SCALAR_LEN, dtype=np.float32)

        known_cells = getattr(self.preprocessor, "known_cells", {})
        known_dirt = getattr(self.preprocessor, "known_dirt_positions", set())
        battery = float(getattr(self.preprocessor, "battery", 0.0))
        battery_max = float(max(getattr(self.preprocessor, "battery_max", 1), 1))
        battery_ratio = battery / battery_max
        charger_list = getattr(self.preprocessor, "charger_list", [])
        npc_list = getattr(self.preprocessor, "npc_list", [])
        simple_charge = self._flag("ENABLE_SIMPLE_CHARGE_MODE", False)
        simple_route_memory = simple_charge and self._flag("ENABLE_SIMPLE_CHARGE_ROUTE_MEMORY", False)
        route_history = list(getattr(self, "recent_charge_route_cells", [])) if (not simple_charge or simple_route_memory) else []
        route_window = int(self._cfg("CHARGE_ROUTE_REPEAT_WINDOW", 24)) if (not simple_charge or simple_route_memory) else 1
        route_memory_limit = float(self._cfg("SIMPLE_CHARGE_ROUTE_MEMORY_STEPS", 160)) if simple_route_memory else (1.0 if simple_charge else 160.0)
        charger_zone_dist = self._charger_zone_dist(cur_pos)
        nearest_charger = self._min_dist(cur_pos, charger_list) if charger_list else None
        nearest_npc = self._min_dist(cur_pos, npc_list) if npc_list else None
        projected_npc = self._projected_npc_dist_to_pos(cur_pos)
        known_cells_ratio = self._norm01(len(known_cells), float(self.preprocessor.GRID_SIZE * self.preprocessor.GRID_SIZE))
        known_dirt_ratio = self._norm01(len(known_dirt), float(max(getattr(self.preprocessor, "total_dirt", 1), 1)))
        recent_unique_ratio = self._recent_traj_unique_ratio(window=24)
        clean_seen_ratio = (
            float(len(getattr(self, "clean_phase_seen", set()))) / float(max(len(known_cells), 1))
            if known_cells
            else 0.0
        )

        # ===== 针对当前失败模式的新 scalar 特征 =====
        # 不改变总维度，替换 simple charge 下价值较低的 post-charge / escape 槽位。
        # 这些只是 observation，模型仍然自己选动作，不是硬规则。
        scalar_risk_enabled = self._flag("ENABLE_SCALAR_RISK_FEATURES", False)
        safe_charger_dist = self._scalar_safe_charger_dist(cur_pos, charger_list) if scalar_risk_enabled else None
        safe_charger_dist_norm = self._norm01(
            safe_charger_dist if safe_charger_dist is not None else float(self._cfg("SCALAR_SAFE_CHARGER_DIST_NORM", 60.0)),
            float(self._cfg("SCALAR_SAFE_CHARGER_DIST_NORM", 60.0)),
        )
        safe_charger_dist_delta_norm = self._scalar_charger_dist_delta_norm(safe_charger_dist) if scalar_risk_enabled else 0.0
        pre_charge_active = self._scalar_pre_charge_active(battery_ratio, safe_charger_dist) if scalar_risk_enabled else 0.0
        battery_risk_active = self._scalar_battery_risk_active(battery_ratio) if scalar_risk_enabled else 0.0
        short_loop_score = self._scalar_short_loop_score() if scalar_risk_enabled else 0.0
        battery_no_progress_steps_norm = self._scalar_battery_no_progress_steps_norm() if scalar_risk_enabled else 0.0

        # 保存给 workflow 日志/监控使用。
        self.scalar_safe_charger_dist = float(safe_charger_dist if safe_charger_dist is not None else 99.0)
        self.scalar_charger_dist_delta = float(safe_charger_dist_delta_norm)
        self.scalar_pre_charge_active = float(pre_charge_active)
        self.scalar_battery_risk_active = float(battery_risk_active)
        self.scalar_short_loop_score = float(short_loop_score)
        self.scalar_battery_no_progress_steps = float(battery_no_progress_steps_norm)

        if not scalar_risk_enabled:
            # 可选兼容开关：关闭时回到旧 scalar 语义。
            pre_charge_active = self._norm01(
                getattr(self, "post_charge_clean_steps", 0),
                float(max(getattr(Config, "POST_CHARGE_CLEAN_COOLDOWN_STEPS", 120), 1)),
            )
            safe_charger_dist_norm = 0.0 if simple_charge else self._norm01(
                self.current_charger_path_dist if self.current_charger_path_dist is not None else 60.0,
                60.0,
            )
            safe_charger_dist_delta_norm = 0.0 if simple_charge else (
                1.0 if self._charge_near_emergency_pressure(self.current_charger_path_dist) else 0.0
            )
            battery_risk_active = 1.0 if self._clean_escape_pressure() else 0.0
            short_loop_score = self._norm01(
                getattr(self, "post_charge_escape_steps", 0),
                float(max(getattr(Config, "POST_CHARGE_ESCAPE_STEPS", 96), 1)),
            )
            battery_no_progress_steps_norm = float(np.clip(getattr(self, "post_charge_escape_ratio", 0.0), 0.0, 1.0))

        feature = np.array(
            [
                1.0 if self.force_charge_mode else 0.0,
                1.0 if self._in_charger_zone(cur_pos) else 0.0,
                1.0 if getattr(self.preprocessor, "is_charging", False) else 0.0,
                1.0 if self._battery_gained_last_step() else 0.0,
                battery_ratio,
                self._norm01(getattr(self.preprocessor, "steps_since_clean", 0), 200.0),
                self._norm01(getattr(self.preprocessor, "charge_phase_steps", 0), 40.0),
                self._norm01(getattr(self.preprocessor, "charge_no_gain_steps", 0), 12.0),
                self._clean_phase_repeat_ratio(),
                self._overall_repeat_ratio(),
                self._recent_route_revisit_ratio(route_history, route_window),
                self._norm01(len(route_history), route_memory_limit),
                known_cells_ratio,
                known_dirt_ratio,
                self._norm01(charger_zone_dist if charger_zone_dist is not None else 20.0, 20.0),
                self._norm01(nearest_charger if nearest_charger is not None else 40.0, 40.0),
                self._norm01(nearest_npc if nearest_npc is not None else 20.0, 20.0),
                self._norm01(projected_npc if projected_npc is not None else 20.0, 20.0),
                float(np.clip(getattr(self.preprocessor, "nearest_npc_velocity", (0.0, 0.0))[0] / 2.0, -1.0, 1.0)),
                float(np.clip(getattr(self.preprocessor, "nearest_npc_velocity", (0.0, 0.0))[1] / 2.0, -1.0, 1.0)),

                # 20: 原 post_charge_clean_steps，当前 post-charge 关闭，改为预回充激活标记。
                pre_charge_active,
                self._recent_traj_revisit_ratio(window=24),
                recent_unique_ratio,
                clean_seen_ratio,
                self._norm01(getattr(self, "current_charge_route_steps", 0), 240.0),
                self._norm01(getattr(self, "current_charge_zone_steps", 0), 48.0),

                # 26~30: simple charge 下原本多为 0 或低价值 post-charge 槽位，替换成风险/循环特征。
                safe_charger_dist_norm,
                safe_charger_dist_delta_norm,
                battery_risk_active,
                short_loop_score,
                battery_no_progress_steps_norm,

                # 31: 保留 clean_escape_ratio，兼容已有清扫逃逸状态。
                float(np.clip(getattr(self, "clean_escape_ratio", 0.0), 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        if feature.shape[0] != int(getattr(Config, "AUGMENTED_SCALAR_LEN", feature.shape[0])):
            raise ValueError(f"unexpected augmented scalar feature length: {feature.shape[0]}")
        return feature

    def _augment_observation_feature(self, base_feature):
        base_arr = np.asarray(base_feature, dtype=np.float32).reshape(-1)
        if base_arr.shape[0] != int(getattr(Config, "BASE_FEATURE_LEN", base_arr.shape[0])):
            raise ValueError(f"unexpected base feature length: {base_arr.shape[0]}")

        local_memory = self._augmented_local_memory_feature()
        global_memory = self._augmented_global_memory_feature()
        scalar_memory = self._augmented_scalar_memory_feature()
        feature = np.concatenate([base_arr, local_memory, global_memory, scalar_memory], axis=0).astype(np.float32)
        if feature.shape[0] != int(getattr(Config, "DIM_OF_OBSERVATION", feature.shape[0])):
            raise ValueError(f"unexpected observation feature length: {feature.shape[0]}")
        return feature

    def _phase_reward_kwargs(self):
        if self._flag("ENABLE_SIMPLE_CHARGE_MODE", False):
            return {
                "force_charge_mode": bool(self.force_charge_mode),
                "charge_route_revisit_ratio": float(self._recent_route_revisit_ratio()),
                "clean_repeat_ratio": float(self._clean_phase_repeat_ratio()),
                "post_charge_pressure": bool(self._post_charge_clean_pressure()),
                "post_charge_escape_pressure": bool(self._post_charge_escape_pressure()),
                "post_charge_escape_ratio": float(getattr(self, "post_charge_escape_ratio", 0.0)),
                "charger_path_dist": self.current_charger_path_dist,
                "last_charger_path_dist": self.last_charger_path_dist,
                "force_charge_search_steps": int(getattr(self, "force_charge_search_steps", 0)),
                "charge_zone_no_gain_steps": int(getattr(self, "charge_zone_no_gain_steps", 0)),
            }
        return {
            "force_charge_mode": bool(self.force_charge_mode),
            "charge_route_revisit_ratio": float(self._recent_route_revisit_ratio()),
            "clean_repeat_ratio": float(self._clean_phase_repeat_ratio()),
            "post_charge_pressure": bool(self._post_charge_clean_pressure()),
            "post_charge_escape_pressure": bool(self._post_charge_escape_pressure()),
            "post_charge_escape_ratio": float(getattr(self, "post_charge_escape_ratio", 0.0)),
            "charger_path_dist": self.current_charger_path_dist,
            "last_charger_path_dist": self.last_charger_path_dist,
            "force_charge_search_steps": int(getattr(self, "force_charge_search_steps", 0)),
            "charge_zone_no_gain_steps": int(getattr(self, "charge_zone_no_gain_steps", 0)),
        }

    def _bfs_first_action(self, targets, legal_action, max_nodes=2200):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        known_cells = getattr(self.preprocessor, "known_cells", {})
        if cur_pos is None or not targets or not known_cells:
            return None

        targets = set(targets)
        if cur_pos in targets:
            return None

        queue = deque()
        visited = {cur_pos}
        for action in self._safe_legal_actions(legal_action):
            nxt = self._next_pos(cur_pos, action)
            if nxt in visited:
                continue
            if nxt in targets:
                return int(action)
            if self._is_known_passable(nxt):
                visited.add(nxt)
                queue.append((nxt, int(action)))

        nodes = 0
        while queue and nodes < max_nodes:
            pos, first_action = queue.popleft()
            nodes += 1
            for action in range(Config.ACTION_NUM):
                nxt = self._next_pos(pos, action)
                if nxt in visited:
                    continue
                if nxt in targets:
                    return int(first_action)
                if not self._is_known_passable(nxt):
                    continue
                visited.add(nxt)
                queue.append((nxt, first_action))

        return None

    def _bfs_distance_to_targets(self, targets, max_nodes=2600):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        known_cells = getattr(self.preprocessor, "known_cells", {})
        if cur_pos is None or not targets or not known_cells:
            return None

        targets = set(targets)
        if cur_pos in targets:
            return 0

        queue = deque([(cur_pos, 0)])
        visited = {cur_pos}
        nodes = 0
        while queue and nodes < max_nodes:
            pos, dist = queue.popleft()
            nodes += 1
            for action in range(Config.ACTION_NUM):
                nxt = self._next_pos(pos, action)
                if nxt in visited:
                    continue
                if nxt in targets:
                    return dist + 1
                if not self._is_known_passable(nxt):
                    continue
                visited.add(nxt)
                queue.append((nxt, dist + 1))

        return None

    def _charger_route_distance_direct(self, charger_list=None, max_nodes=None):
        """真实 BFS 充电路径距离。

        v17：之前 simple charge 模式下 _charger_route_distance 直接返回 None，
        造成 agent_current_path 长期是 99；但 workflow_direct_path 明明能算出
        5~20 格路径，低电临近充电区时 agent 仍按未知路径处理，容易 200/300 步没电死。
        这个函数不改变地图，只让 agent 在 simple mode 下也能拿到保守路径距离。
        """
        charger_list = getattr(self.preprocessor, "charger_list", []) if charger_list is None else charger_list
        if not charger_list:
            return None

        targets = self._charger_zone_targets() or self._npc_safe_charger_targets(charger_list) or charger_list
        if not targets:
            return None

        if max_nodes is None:
            max_nodes = (
                int(self._cfg("SIMPLE_CHARGE_DIRECT_BFS_LIMIT", self._cfg("CHARGE_ROUTE_BFS_LIMIT", 2600)))
                if self._flag("ENABLE_SIMPLE_CHARGE_MODE", False)
                else int(self._cfg("CHARGE_ROUTE_BFS_LIMIT", 2600))
            )
        return self._bfs_distance_to_targets(targets, max_nodes=int(max_nodes))

    def _charger_route_distance(self, charger_list=None):
        if self._flag("ENABLE_SIMPLE_CHARGE_MODE", False) and not self._flag(
            "ENABLE_SIMPLE_CHARGE_DIRECT_PATH_DIST",
            False,
        ):
            return None
        return self._charger_route_distance_direct(charger_list)

    def _is_charge_route_unknown_or_far(self, path_charger_dist):
        if path_charger_dist is None:
            return True
        soft_limit = int(self._cfg("CHARGE_UNKNOWN_ROUTE_SOFT_PATH_DIST", 24))
        return int(path_charger_dist) > soft_limit

    def _action_advances_to_charger(self, action, charger_list=None):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        charger_list = getattr(self.preprocessor, "charger_list", []) if charger_list is None else charger_list
        if cur_pos is None or not charger_list:
            return False

        nxt = self._next_pos(cur_pos, action)
        cur_zone_dist = self._charger_zone_dist(cur_pos)
        next_zone_dist = self._charger_zone_dist(nxt)
        if cur_zone_dist is not None and next_zone_dist is not None and int(next_zone_dist) < int(cur_zone_dist):
            return True

        cur_charger_dist = min(self._manhattan(cur_pos, target) for target in charger_list)
        next_charger_dist = min(self._manhattan(nxt, target) for target in charger_list)
        return int(next_charger_dist) < int(cur_charger_dist)

    def _charge_near_lock_active(self, path_charger_dist=None):
        if not self.force_charge_mode:
            return False

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None or self._in_charger_zone(cur_pos):
            return False

        cur_zone_dist = self._charger_zone_dist(cur_pos)
        if cur_zone_dist is None:
            return False

        if int(cur_zone_dist) <= int(self._cfg("CHARGE_NEAR_LOCK_ZONE_DIST", 2)):
            return True
        return path_charger_dist is not None and int(path_charger_dist) <= int(
            self._cfg("CHARGE_NEAR_LOCK_PATH_DIST", 3)
        )

    def _charge_near_loop_pressure(self, path_charger_dist=None):
        if not self.force_charge_mode:
            return False

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None or self._in_charger_zone(cur_pos):
            return False

        if path_charger_dist is None:
            path_charger_dist = self.current_charger_path_dist
        if path_charger_dist is None:
            path_charger_dist = self._charger_route_distance(getattr(self.preprocessor, "charger_list", []))
        if path_charger_dist is None:
            return False

        if int(path_charger_dist) > int(self._cfg("CHARGE_NEAR_LOOP_PATH_MAX", 5)):
            return False

        if int(getattr(self, "current_charge_zone_steps", 0)) > int(self._cfg("CHARGE_NEAR_LOOP_MAX_ZONE_STEPS", 0)):
            return False

        route_ratio = float(self._recent_route_revisit_ratio())
        no_progress_steps = int(getattr(self, "force_charge_no_progress_steps", 0))
        return bool(
            route_ratio >= float(self._cfg("CHARGE_NEAR_LOOP_ROUTE_RATIO", 0.55))
            or no_progress_steps >= int(self._cfg("CHARGE_NEAR_LOOP_NO_PROGRESS_STEPS", 2))
        )

    def _charge_entry_action(self, legal_action, charger_list=None):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        charger_list = getattr(self.preprocessor, "charger_list", []) if charger_list is None else charger_list
        if cur_pos is None or not charger_list or self._in_charger_zone(cur_pos):
            return None

        cur_zone_dist = self._charger_zone_dist(cur_pos)
        if cur_zone_dist is None:
            return None

        legal_actions = self._safe_legal_actions(legal_action)
        if not legal_actions:
            legal_actions = self._passable_legal_actions(legal_action)
        if not legal_actions:
            legal_actions = self._static_passable_legal_actions(legal_action)
        if not legal_actions:
            return None

        near_dist = int(self._cfg("CHARGER_NEAR_DIST", 1))
        path_charger_dist = self.current_charger_path_dist
        if path_charger_dist is None:
            path_charger_dist = self._charger_route_distance(charger_list)
        near_loop_pressure = self._charge_near_loop_pressure(path_charger_dist)
        route_history = list(getattr(self, "recent_charge_route_cells", []))
        cur_raw_dist = min(self._manhattan(cur_pos, target) for target in charger_list)
        best_action = None
        best_score = -10**9
        for action in legal_actions:
            nxt = self._next_pos(cur_pos, action)
            if self._local_cell(nxt) == 0:
                continue

            next_zone_dist = self._charger_zone_dist(nxt)
            if next_zone_dist is None:
                continue

            visit_n = float(getattr(self.preprocessor, "visit_count", {}).get(nxt, 0))
            charger_dist = min(self._manhattan(nxt, target) for target in charger_list)
            score = -0.25 * float(charger_dist)
            score += float(self._cfg("CHARGE_NEAR_ENTRY_PROGRESS_BONUS", 18.0)) * (
                float(int(cur_zone_dist) - int(next_zone_dist))
            )
            if int(next_zone_dist) <= near_dist:
                score += float(self._cfg("CHARGE_NEAR_ENTRY_ZONE_BONUS", 48.0))
            elif int(next_zone_dist) > int(cur_zone_dist):
                score -= float(self._cfg("CHARGE_NEAR_ENTRY_BACKTRACK_PENALTY", 28.0)) * (
                    float(int(next_zone_dist) - int(cur_zone_dist))
                )

            score -= float(self._cfg("CHARGE_NEAR_ENTRY_VISIT_PENALTY", 0.8)) * visit_n
            score -= self._recent_position_penalty(nxt)
            if self._local_cell(nxt) == 2:
                score += 2.0
            if self._is_frontier_cell(nxt) and int(next_zone_dist) <= int(cur_zone_dist):
                score += 0.8
            if int(action) == int(self.last_action):
                score += 0.2
            if near_loop_pressure:
                route_hits = float(route_history[-24:].count(nxt))
                next_raw_dist = min(self._manhattan(nxt, target) for target in charger_list)
                if int(next_zone_dist) < int(cur_zone_dist):
                    score += float(self._cfg("CHARGE_NEAR_LOOP_ZONE_BONUS", 18.0)) * (
                        float(int(cur_zone_dist) - int(next_zone_dist))
                    )
                if int(next_raw_dist) < int(cur_raw_dist):
                    score += float(self._cfg("CHARGE_NEAR_LOOP_PROGRESS_BONUS", 12.0)) * (
                        float(int(cur_raw_dist) - int(next_raw_dist))
                    )
                elif int(next_raw_dist) > int(cur_raw_dist):
                    score -= float(self._cfg("CHARGE_NEAR_LOOP_BACKTRACK_PENALTY", 16.0)) * (
                        float(int(next_raw_dist) - int(cur_raw_dist))
                    )
                score -= float(self._cfg("CHARGE_NEAR_LOOP_REVISIT_PENALTY", 10.0)) * route_hits
                if self._action_revisits_recent(action):
                    score -= 0.6 * float(self._cfg("CHARGE_NEAR_LOOP_REVISIT_PENALTY", 10.0))
                if int(next_zone_dist) >= int(cur_zone_dist) and int(next_raw_dist) >= int(cur_raw_dist):
                    score -= 0.8 * float(self._cfg("CHARGE_NEAR_LOOP_BACKTRACK_PENALTY", 16.0))

            npc_dist = self._nearest_npc_dist_after(action)
            if npc_dist is not None:
                score += 0.15 * float(npc_dist)
                if int(npc_dist) <= int(self._cfg("NPC_DANGER_DIST", 2)):
                    score -= 40.0

            if score > best_score:
                best_score = score
                best_action = int(action)

        return best_action

    def _charger_frontier_bfs_action(self, legal_action, charger_targets=None):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        known_cells = getattr(self.preprocessor, "known_cells", {})
        charger_targets = charger_targets or getattr(self.preprocessor, "charger_list", [])
        if cur_pos is None or not known_cells or not charger_targets:
            return None

        cur_charger_dist = min(self._manhattan(cur_pos, target) for target in charger_targets)
        visit_count = getattr(self.preprocessor, "visit_count", {})
        route_history = list(getattr(self, "recent_charge_route_cells", []))
        route_repeat_pressure = self._charge_route_repeat_pressure()
        closer_frontiers = []
        fallback_frontiers = []
        for pos, cell in known_cells.items():
            if cell not in (1, 2) or not self._is_frontier_cell(pos):
                continue

            charger_dist = min(self._manhattan(pos, target) for target in charger_targets)
            open_neighbors = self._known_open_neighbor_count(pos)
            route_repeats = route_history.count(pos)
            if route_repeat_pressure:
                score_key = (
                    route_repeats,
                    visit_count.get(pos, 0),
                    charger_dist,
                    -open_neighbors,
                    self._manhattan(cur_pos, pos),
                    self._recent_position_penalty(pos),
                )
            else:
                score_key = (
                    charger_dist,
                    visit_count.get(pos, 0),
                    -open_neighbors,
                    self._manhattan(cur_pos, pos),
                    self._recent_position_penalty(pos),
                )
            if charger_dist < cur_charger_dist:
                closer_frontiers.append((score_key, pos))
            else:
                fallback_frontiers.append((score_key, pos))

        frontiers = closer_frontiers or fallback_frontiers
        if not frontiers:
            return None

        frontiers.sort(key=lambda item: item[0])
        limit_name = "CHARGE_FRONTIER_REPEAT_BFS_LIMIT" if route_repeat_pressure else "CHARGE_FRONTIER_BFS_LIMIT"
        limit = int(self._cfg(limit_name, self._cfg("CHARGE_FRONTIER_BFS_LIMIT", 180)))
        return self._bfs_first_action([pos for _, pos in frontiers[:limit]], legal_action, max_nodes=4000)

    def _charger_open_frontier_bfs_action(self, legal_action, charger_targets=None):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        known_cells = getattr(self.preprocessor, "known_cells", {})
        charger_targets = charger_targets or getattr(self.preprocessor, "charger_list", [])
        if cur_pos is None or not known_cells or not charger_targets:
            return None

        cur_charger_dist = min(self._manhattan(cur_pos, target) for target in charger_targets)
        visit_count = getattr(self.preprocessor, "visit_count", {})
        route_history = list(getattr(self, "recent_charge_route_cells", []))
        min_neighbors = int(self._cfg("OPEN_FRONTIER_MIN_KNOWN_NEIGHBORS", 4))
        route_repeat_pressure = self._charge_route_repeat_pressure()
        closer_frontiers = []
        fallback_frontiers = []
        for pos, cell in known_cells.items():
            if cell not in (1, 2) or not self._is_frontier_cell(pos):
                continue

            open_neighbors = self._known_open_neighbor_count(pos)
            if open_neighbors < min_neighbors:
                continue

            charger_dist = min(self._manhattan(pos, target) for target in charger_targets)
            route_repeats = route_history.count(pos)
            score_key = (
                route_repeats if route_repeat_pressure else 0,
                visit_count.get(pos, 0),
                charger_dist,
                -open_neighbors,
                self._manhattan(cur_pos, pos),
                self._recent_position_penalty(pos),
            )
            if charger_dist < cur_charger_dist:
                closer_frontiers.append((score_key, pos))
            else:
                fallback_frontiers.append((score_key, pos))

        frontiers = closer_frontiers or fallback_frontiers
        if not frontiers:
            return None

        frontiers.sort(key=lambda item: item[0])
        limit = int(self._cfg("CHARGE_OPEN_FRONTIER_BFS_LIMIT", 140))
        return self._bfs_first_action([pos for _, pos in frontiers[:limit]], legal_action, max_nodes=4000)

    def _known_dirt_bfs_action(self, legal_action):
        known_dirt = getattr(self.preprocessor, "known_dirt_positions", set())
        if not known_dirt:
            return None
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is not None:
            known_dirt = sorted(known_dirt, key=lambda p: self._manhattan(cur_pos, p))
        limit = int(self._cfg("KNOWN_DIRT_BFS_LIMIT", 160))
        return self._bfs_first_action(known_dirt[:limit] if isinstance(known_dirt, list) else known_dirt, legal_action)

    def _frontier_bfs_action(self, legal_action):
        known_cells = getattr(self.preprocessor, "known_cells", {})
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None or not known_cells:
            return None

        frontier = []
        for pos, cell in known_cells.items():
            if cell not in (1, 2):
                continue
            if self._is_frontier_cell(pos):
                frontier.append(pos)

        if not frontier:
            return None

        visit_count = getattr(self.preprocessor, "visit_count", {})
        frontier.sort(
            key=lambda p: (
                self._npc_clean_suspect_ttl(p) > 0,
                visit_count.get(p, 0),
                self._manhattan(cur_pos, p),
            )
        )
        limit = int(self._cfg("FRONTIER_BFS_LIMIT", 180))
        return self._bfs_first_action(frontier[:limit], legal_action)

    def _open_frontier_bfs_action(self, legal_action):
        known_cells = getattr(self.preprocessor, "known_cells", {})
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None or not known_cells:
            return None

        min_neighbors = int(self._cfg("OPEN_FRONTIER_MIN_KNOWN_NEIGHBORS", 4))
        open_frontier = []
        for pos, cell in known_cells.items():
            if cell not in (1, 2):
                continue
            if not self._is_frontier_cell(pos):
                continue
            open_neighbors = self._known_open_neighbor_count(pos)
            if open_neighbors < min_neighbors:
                continue
            open_frontier.append((pos, open_neighbors))

        if not open_frontier:
            return None

        visit_count = getattr(self.preprocessor, "visit_count", {})
        open_frontier.sort(
            key=lambda item: (
                self._npc_clean_suspect_ttl(item[0]) > 0,
                visit_count.get(item[0], 0),
                -item[1],
                self._manhattan(cur_pos, item[0]),
            )
        )
        limit = int(self._cfg("OPEN_FRONTIER_BFS_LIMIT", 220))
        return self._bfs_first_action([pos for pos, _ in open_frontier[:limit]], legal_action)

    def _nearest_npc_dist_after(self, action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        npc_list = getattr(self.preprocessor, "npc_list", [])
        if cur_pos is None or not npc_list:
            return None
        return self._min_dist(self._next_pos(cur_pos, action), npc_list)

    def _moving_npc_dist_after(self, action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return None

        nxt = self._next_pos(cur_pos, action)
        return self._projected_npc_dist_to_pos(nxt, tracks=self._projected_npc_tracks(steps=1))

    def _projected_npc_dist_after(self, action):
        npc_dist = self._nearest_npc_dist_after(action)
        if npc_dist is None:
            return None
        moving_dist = self._moving_npc_dist_after(action)
        if moving_dist is not None:
            npc_dist = min(int(npc_dist), int(moving_dist))
        return max(0, int(npc_dist) - 1)

    def _charge_near_emergency_pressure(self, path_charger_dist=None):
        if not self.force_charge_mode:
            return False

        battery = getattr(self.preprocessor, "battery", None)
        battery_max = getattr(self.preprocessor, "battery_max", None)
        if battery is None or battery_max is None or battery_max <= 0:
            return False

        if path_charger_dist is None:
            path_charger_dist = self.current_charger_path_dist
        if path_charger_dist is None:
            path_charger_dist = self._charger_route_distance(getattr(self.preprocessor, "charger_list", []))
        if path_charger_dist is None:
            return False

        battery_ratio = float(battery) / float(battery_max)
        dist = int(path_charger_dist)
        near_limit = int(self._cfg("CHARGE_NEAR_EMERGENCY_PATH_DIST", 8))
        mid_limit = int(self._cfg("CHARGE_MID_EMERGENCY_PATH_DIST", 20))
        if dist <= near_limit:
            reserve = self._adaptive_charge_steps(
                "CHARGE_NEAR_EMERGENCY_RESERVE_STEPS",
                "ADAPTIVE_CHARGE_RESERVE_MAX_SCALE",
            )
            base_ratio = float(self._cfg("CHARGE_NEAR_EMERGENCY_BATTERY_RATIO", 0.14))
        elif dist <= mid_limit:
            reserve = self._adaptive_charge_steps(
                "CHARGE_MID_EMERGENCY_RESERVE_STEPS",
                "ADAPTIVE_CHARGE_RESERVE_MAX_SCALE",
            )
            base_ratio = float(self._cfg("CHARGE_MID_EMERGENCY_BATTERY_RATIO", 0.19))
        else:
            return False
        trigger_ratio = max(
            base_ratio,
            (float(dist) + reserve) / float(battery_max),
        )
        return battery_ratio <= trigger_ratio

    def _charge_progress_stall_pressure(self, path_charger_dist=None):
        if not self.force_charge_mode:
            return False

        stall_steps = int(self._cfg("CHARGE_PROGRESS_STALL_STEPS", 6))
        early_steps = int(self._cfg("CHARGE_PROGRESS_STALL_EARLY_STEPS", 4))
        no_progress_steps = int(getattr(self, "force_charge_no_progress_steps", 0))
        if no_progress_steps < min(stall_steps, early_steps):
            return False

        if path_charger_dist is None:
            path_charger_dist = self.current_charger_path_dist
        if path_charger_dist is None:
            return no_progress_steps >= stall_steps

        dist = int(path_charger_dist)
        route_revisit_ratio = float(self._recent_route_revisit_ratio())
        if (
            early_steps < stall_steps
            and no_progress_steps >= early_steps
            and dist >= int(self._cfg("CHARGE_PROGRESS_STALL_PATH_MIN", 8))
            and dist <= int(self._cfg("CHARGE_PROGRESS_STALL_MID_PATH_MAX", 10))
            and route_revisit_ratio >= float(self._cfg("CHARGE_PROGRESS_STALL_ROUTE_REVISIT_RATIO", 0.70))
        ):
            return True

        return no_progress_steps >= stall_steps and dist >= int(self._cfg("CHARGE_PROGRESS_STALL_PATH_MIN", 8))

    def _safe_legal_actions(self, legal_action):
        danger_dist = self._cfg("NPC_DANGER_DIST", 2)
        emergency_dist = self._cfg("NPC_EMERGENCY_PROJECTED_DIST", max(1, danger_dist - 1))
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        npc_list = getattr(self.preprocessor, "npc_list", [])
        legal_actions = self._passable_legal_actions(legal_action)
        if not legal_actions:
            legal_actions = self._static_passable_legal_actions(legal_action)
        if not legal_actions:
            return self._legal_actions(legal_action)

        safe_actions = []
        fallback_actions = []
        for action in legal_actions:
            projected_dist = self._projected_npc_dist_after(action)
            if projected_dist is None or projected_dist > emergency_dist:
                safe_actions.append(action)
                continue

            npc_dist = self._nearest_npc_dist_after(action)
            if npc_dist is None or npc_dist > emergency_dist:
                fallback_actions.append(action)
        return safe_actions or fallback_actions or legal_actions

    def _strict_escape_actions(self, legal_action):
        legal_actions = self._passable_legal_actions(legal_action)
        if not legal_actions:
            legal_actions = self._static_passable_legal_actions(legal_action)
        if not legal_actions:
            return []

        danger_dist = self._cfg("NPC_DANGER_DIST", 2)
        emergency_dist = self._cfg("NPC_EMERGENCY_PROJECTED_DIST", max(1, danger_dist - 1))
        safe_actions = []
        fallback_actions = []
        for action in legal_actions:
            projected_dist = self._projected_npc_dist_after(action)
            if projected_dist is None or projected_dist > emergency_dist:
                safe_actions.append(action)
                continue

            npc_dist = self._nearest_npc_dist_after(action)
            if npc_dist is None or npc_dist > emergency_dist:
                fallback_actions.append(action)
        return safe_actions or fallback_actions

    def _update_force_charge_mode(self):
        battery = getattr(self.preprocessor, "battery", None)
        battery_max = getattr(self.preprocessor, "battery_max", None)
        if battery is None or battery_max is None or battery_max <= 0:
            return

        ratio = float(battery) / float(battery_max)
        if self._flag("ENABLE_SIMPLE_CHARGE_MODE", False):
            charge_trigger = float(self._cfg("FORCE_CHARGE_BATTERY_RATIO", 0.18))
            exit_trigger = float(self._cfg("EXIT_CHARGE_BATTERY_RATIO", 0.40))

            # v11：simple route trigger 是“提前回充”的进入条件，不应该压过退出条件。
            # v10 坏局里出现 charge_count 很高、remaining_charge 接近满电，说明机器人
            # 经常在高电量仍被动态 route_trigger 牵着回充/绕桩。已经进入 force_charge
            # 且电量达到 exit_trigger 时，先允许退出，把时间还给清扫。
            if self.force_charge_mode and ratio >= exit_trigger:
                self.force_charge_mode = False
                self.last_force_charge_exit_step = int(getattr(self.preprocessor, "step_no", 0))
                return

            # v12：先给 safe_charge_dist 一个默认值，下面同时用于动态触发和 unknown fallback。
            safe_charge_dist = None
            manhattan_dist = None
            path_dist = None

            # v8：simple charge 原来只在固定低电量触发，导致日志里大量 200 步
            # FAIL_BATTERY：机器人最后 45 步才开始回充，面对 10-40+ 格路径明显太晚。
            # 这里只根据“到充电区的保守距离 + reserve”提高触发线，不改变地图/模型，
            # 也不恢复复杂的未知路线硬提交。
            if self._flag("ENABLE_SIMPLE_CHARGE_ROUTE_TRIGGER", False):
                cur_pos = getattr(self.preprocessor, "cur_pos", None)
                charger_list = getattr(self.preprocessor, "charger_list", [])
                if cur_pos is not None and charger_list:
                    manhattan_dist = self._min_dist(cur_pos, charger_list)
                    route_targets = self._charger_zone_targets() or charger_list
                    # v19：direct BFS 只用于回充触发距离，不打开动作层 direct path。
                    if self._flag("ENABLE_SIMPLE_CHARGE_DIRECT_TRIGGER_DIST", False):
                        path_dist = self._charger_route_distance_direct(route_targets)
                    else:
                        path_dist = self._charger_route_distance(route_targets)
                    self.simple_charge_trigger_path_dist = int(path_dist) if path_dist is not None else 99
                    if path_dist is not None and int(path_dist) < 99:
                        safe_charge_dist = int(path_dist)
                        if manhattan_dist is not None:
                            safe_charge_dist = max(safe_charge_dist, int(manhattan_dist))
                    elif manhattan_dist is not None:
                        safe_charge_dist = int(manhattan_dist)

                if safe_charge_dist is not None:
                    min_dist = int(self._cfg("SIMPLE_CHARGE_ROUTE_MIN_DIST", 10))
                    if int(safe_charge_dist) >= min_dist:
                        reserve = int(self._cfg("SIMPLE_CHARGE_ROUTE_RESERVE_STEPS", 34))
                        cap = float(self._cfg("SIMPLE_CHARGE_ROUTE_TRIGGER_CAP", 0.52))
                        route_trigger = (float(safe_charge_dist) + float(reserve)) / float(battery_max)
                        charge_trigger = max(charge_trigger, min(cap, route_trigger))

            # v19：150电量+3桩时，前期离桩远就提前进入 force_charge，但不改动作。
            if safe_charge_dist is not None:
                step_no = int(getattr(self.preprocessor, "step_no", 0))
                early_max_step = int(self._cfg("SIMPLE_CHARGE_EARLY_SAFE_MAX_STEP", 260))
                early_min_dist = int(self._cfg("SIMPLE_CHARGE_EARLY_SAFE_MIN_DIST", 24))
                early_ratio = float(self._cfg("SIMPLE_CHARGE_EARLY_SAFE_RATIO", 0.55))
                early_floor = float(self._cfg("SIMPLE_CHARGE_EARLY_SAFE_TRIGGER_FLOOR", 0.50))
                if step_no <= early_max_step and int(safe_charge_dist) >= early_min_dist and ratio <= early_ratio:
                    charge_trigger = max(charge_trigger, early_floor)

            # v24：开局从未回血且离桩远/路径未知时，提前进入 force_charge。
            # 只提高触发线，不直接改动作；避免 v17 那种低电硬 capture 负优化。
            if self._flag("ENABLE_OPENING_NO_CHARGE_GUARD", False):
                step_no = int(getattr(self.preprocessor, "step_no", 0))
                no_charge_yet = int(getattr(self, "episode_charge_gain_steps", 0)) <= 0
                never_only = self._flag("OPENING_NO_CHARGE_NEVER_GAIN_ONLY", True)
                if (no_charge_yet or not never_only) and step_no <= int(self._cfg("OPENING_NO_CHARGE_MAX_STEP", 240)):
                    cap = float(self._cfg("OPENING_NO_CHARGE_TRIGGER_CAP", 0.70))
                    dist_ref = safe_charge_dist
                    if dist_ref is None and manhattan_dist is not None:
                        dist_ref = int(manhattan_dist)
                    path_unknown = path_dist is None or int(path_dist) >= 99
                    manhattan_ok = manhattan_dist is not None and int(manhattan_dist) >= int(self._cfg("OPENING_NO_CHARGE_MIN_MANHATTAN", 24))
                    if path_unknown and manhattan_ok:
                        charge_trigger = max(charge_trigger, min(cap, float(self._cfg("OPENING_NO_CHARGE_UNKNOWN_TRIGGER", 0.66))))
                    elif dist_ref is not None:
                        if int(dist_ref) >= int(self._cfg("OPENING_NO_CHARGE_FAR_DIST", 42)):
                            charge_trigger = max(charge_trigger, min(cap, float(self._cfg("OPENING_NO_CHARGE_FAR_TRIGGER", 0.68))))
                        elif int(dist_ref) >= int(self._cfg("OPENING_NO_CHARGE_MID_DIST", 28)):
                            charge_trigger = max(charge_trigger, min(cap, float(self._cfg("OPENING_NO_CHARGE_MID_TRIGGER", 0.60))))

            # v12：如果没有可靠路径距离，但电量已经进入中低区间，先保守进入找桩/回充；
            # 否则多 worker 里会出现 200 或 500 步左右 battery fail。
            unknown_trigger = float(self._cfg("SIMPLE_CHARGE_UNKNOWN_TRIGGER_RATIO", 0.40))
            if safe_charge_dist is None and unknown_trigger > 0.0 and ratio <= unknown_trigger:
                # v15：unknown fallback 只兜底低电，不再 0.50 过早拉入回充搜索，
                # 避免高重复训练里大量时间花在“找桩/旧路线”上。
                charge_trigger = max(charge_trigger, unknown_trigger)

            if ratio <= charge_trigger:
                step_no = int(getattr(self.preprocessor, "step_no", 0))
                cooldown_steps = int(self._cfg("SIMPLE_CHARGE_REENTER_COOLDOWN_STEPS", 0))
                reenter_min_ratio = float(self._cfg("SIMPLE_CHARGE_REENTER_MIN_RATIO", 0.28))
                in_reenter_cooldown = (step_no - int(getattr(self, "last_force_charge_exit_step", -10**9))) < max(0, cooldown_steps)
                if (not in_reenter_cooldown) or ratio <= reenter_min_ratio:
                    self.force_charge_mode = True
            elif ratio >= exit_trigger:
                self.force_charge_mode = False
                self.last_force_charge_exit_step = int(getattr(self.preprocessor, "step_no", 0))
            return

        charge_trigger = self._cfg("FORCE_CHARGE_BATTERY_RATIO", 0.18)
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        charger_list = getattr(self.preprocessor, "charger_list", [])
        step_no = int(getattr(self.preprocessor, "step_no", 0))
        dirt_cleaned = int(getattr(self.preprocessor, "dirt_cleaned", 0))
        _, _, resource_pressure = self._charge_env_pressure(self._cfg("ADAPTIVE_CHARGE_RESERVE_MAX_SCALE", 1.0))
        manhattan_charger_dist = self._min_dist(cur_pos, charger_list) if cur_pos is not None else None
        path_charger_dist = None
        route_charge_trigger = None
        if manhattan_charger_dist is not None:
            reserve = self._adaptive_charge_steps("CHARGE_ROUTE_RESERVE_STEPS", "ADAPTIVE_CHARGE_RESERVE_MAX_SCALE")
            trigger_cap = self._adaptive_charge_cap(max(float(charge_trigger), 0.28))
            charger_dist = float(manhattan_charger_dist)
            path_charger_dist = self._charger_route_distance(charger_list)
            if path_charger_dist is not None:
                charger_dist = float(max(int(path_charger_dist), int(manhattan_charger_dist)))
                detour_steps = int(self._cfg("CHARGE_ROUTE_DETOUR_STEPS", 18))
                if int(path_charger_dist) >= int(manhattan_charger_dist) + detour_steps:
                    trigger_cap = max(
                        trigger_cap,
                        self._adaptive_charge_cap(self._cfg("CHARGE_PATH_TRIGGER_CAP", trigger_cap)),
                    )
                long_route_min = int(self._cfg("CHARGE_LONG_ROUTE_MIN_DIST", 80))
                _, _, long_route_pressure = self._charge_env_pressure(self._cfg("ADAPTIVE_CHARGE_RESERVE_MAX_SCALE", 1.0))
                long_route_floor = int(self._cfg("ADAPTIVE_CHARGE_LONG_ROUTE_MIN_FLOOR", 48))
                long_route_min = max(long_route_floor, int(round(float(long_route_min) / max(1.0, long_route_pressure))))
                if int(path_charger_dist) >= long_route_min:
                    trigger_cap = max(
                        trigger_cap,
                        self._adaptive_charge_cap(self._cfg("CHARGE_LONG_ROUTE_TRIGGER_CAP", trigger_cap)),
                    )
                near_path_min = int(self._cfg("CHARGE_KNOWN_NEAR_PATH_MIN_DIST", 4))
                near_path_max = int(self._cfg("CHARGE_KNOWN_NEAR_PATH_MAX_DIST", 10))
                mid_path_min = int(self._cfg("CHARGE_KNOWN_MID_PATH_MIN_DIST", near_path_max + 1))
                mid_path_max = int(self._cfg("CHARGE_KNOWN_MID_PATH_MAX_DIST", 20))
                if near_path_min <= int(path_charger_dist) <= near_path_max:
                    reserve += self._adaptive_charge_steps(
                        "CHARGE_KNOWN_NEAR_PATH_EXTRA_RESERVE_STEPS",
                        "ADAPTIVE_CHARGE_RESERVE_MAX_SCALE",
                    )
                    trigger_cap = max(
                        trigger_cap,
                        self._adaptive_charge_cap(self._cfg("CHARGE_KNOWN_NEAR_PATH_TRIGGER_CAP", trigger_cap)),
                    )
                elif mid_path_min <= int(path_charger_dist) <= mid_path_max:
                    mid_detour_steps = int(self._cfg("CHARGE_KNOWN_MID_PATH_DETOUR_STEPS", 6))
                    # Do not treat every 11-20 known charger path as a high-risk route.
                    # Only inflate the trigger when the BFS route is already meaningfully
                    # worse than straight-line distance; otherwise the generic dynamic
                    # trigger is enough and avoids over-early force-charge.
                    if int(path_charger_dist) >= int(manhattan_charger_dist) + mid_detour_steps:
                        reserve += self._adaptive_charge_steps(
                            "CHARGE_KNOWN_MID_PATH_EXTRA_RESERVE_STEPS",
                            "ADAPTIVE_CHARGE_RESERVE_MAX_SCALE",
                        )
                        trigger_cap = max(
                            trigger_cap,
                            self._adaptive_charge_cap(self._cfg("CHARGE_KNOWN_MID_PATH_TRIGGER_CAP", trigger_cap)),
                        )
                    if resource_pressure >= float(self._cfg("CHARGE_LOW_RESOURCE_PRESSURE", 1.15)):
                        reserve += self._adaptive_charge_steps(
                            "CHARGE_LOW_RESOURCE_MID_PATH_EXTRA_RESERVE_STEPS",
                            "ADAPTIVE_CHARGE_RESERVE_MAX_SCALE",
                        )
                        trigger_cap = max(
                            trigger_cap,
                            self._adaptive_charge_cap(
                                self._cfg("CHARGE_LOW_RESOURCE_MID_PATH_TRIGGER_CAP", trigger_cap)
                            ),
                        )
            else:
                unknown_min_steps = self._adaptive_charge_steps(
                    "CHARGE_UNKNOWN_ROUTE_MIN_STEPS",
                    "ADAPTIVE_CHARGE_UNKNOWN_MIN_STEPS_MAX_SCALE",
                )
                charger_dist = float(max(
                    int(charger_dist),
                    int(round(float(unknown_min_steps))),
                ))
                reserve += self._adaptive_charge_steps(
                    "CHARGE_UNKNOWN_ROUTE_EXTRA_RESERVE_STEPS",
                    "ADAPTIVE_CHARGE_RESERVE_MAX_SCALE",
                )
                trigger_cap = max(
                    trigger_cap,
                    self._adaptive_charge_cap(self._cfg("CHARGE_UNKNOWN_ROUTE_TRIGGER_CAP", trigger_cap)),
                )

            npc_near_dist = int(self._cfg("CHARGE_NPC_NEAR_DIST", self._cfg("NPC_AVOID_DIST", 4)))
            nearest_charger = min(charger_list, key=lambda p: self._manhattan(cur_pos, p))
            cur_npc_dist = self._npc_dist(cur_pos)
            charger_npc_dist = self._npc_dist(nearest_charger)
            npc_disrupts_route = (
                (cur_npc_dist is not None and int(cur_npc_dist) <= npc_near_dist)
                or (charger_npc_dist is not None and int(charger_npc_dist) <= npc_near_dist)
            )
            if self._flag("ENABLE_CHARGE_NPC_EXTRA_RESERVE", False) and npc_disrupts_route:
                reserve += self._adaptive_charge_steps(
                    "CHARGE_NPC_EXTRA_RESERVE_STEPS",
                    "ADAPTIVE_CHARGE_RESERVE_MAX_SCALE",
                )
                trigger_cap = max(
                    trigger_cap,
                    self._adaptive_charge_cap(self._cfg("CHARGE_NPC_TRIGGER_CAP", trigger_cap)),
                )
            if self._flag("ENABLE_CHARGE_NARROW_EXTRA_RESERVE", False) and self._narrow_area_pressure():
                reserve += self._adaptive_charge_steps(
                    "CHARGE_NARROW_EXTRA_RESERVE_STEPS",
                    "ADAPTIVE_CHARGE_RESERVE_MAX_SCALE",
                )
                trigger_cap = max(
                    trigger_cap,
                    self._adaptive_charge_cap(self._cfg("CHARGE_NARROW_TRIGGER_CAP", trigger_cap)),
                )
            route_trigger = (float(charger_dist) + reserve) / float(battery_max)
            route_charge_trigger = min(trigger_cap, route_trigger)

        # Soft opening relaxation: do not hard-clamp force-charge. Instead, when the
        # episode is still in the opening and the charger route is only an ordinary
        # known medium/far path, slightly relax route reserve/cap so early cleaning
        # can establish momentum. Near-path protection and true emergency handling
        # remain unchanged.
        opening_path_min = int(self._cfg("OPENING_CHARGE_RELAX_PATH_MIN_DIST", 11))
        opening_step_cap = int(self._cfg("OPENING_CHARGE_RELAX_STEPS", 320))
        opening_clean_cap = int(self._cfg("OPENING_CHARGE_RELAX_CLEANED", 260))
        opening_long_route_min = int(self._cfg("CHARGE_LONG_ROUTE_MIN_DIST", 80))
        opening_charge_relax = (
            not self.force_charge_mode
            and path_charger_dist is not None
            and opening_path_min <= int(path_charger_dist) < opening_long_route_min
            and (
                step_no <= opening_step_cap
                or dirt_cleaned <= opening_clean_cap
            )
        )
        if opening_charge_relax and manhattan_charger_dist is not None:
            relax_scale = float(self._cfg("OPENING_CHARGE_RESERVE_SCALE", 0.55))
            relax_cap = float(self._cfg("OPENING_CHARGE_TRIGGER_CAP", 0.30))
            low_resource_pressure = float(self._cfg("OPENING_CHARGE_LOW_RESOURCE_PRESSURE", 1.15))
            full_resource_pressure = max(
                low_resource_pressure + 0.01,
                float(self._cfg("OPENING_CHARGE_LOW_RESOURCE_FULL_PRESSURE", 1.55)),
            )
            if resource_pressure > low_resource_pressure:
                resource_t = min(
                    1.0,
                    max(0.0, (float(resource_pressure) - low_resource_pressure) / (full_resource_pressure - low_resource_pressure)),
                )
                relax_scale += (1.0 - relax_scale) * resource_t
                relax_cap += (
                    float(self._cfg("OPENING_CHARGE_LOW_RESOURCE_TRIGGER_CAP", relax_cap)) - relax_cap
                ) * resource_t
            reserve *= relax_scale
            trigger_cap = min(
                trigger_cap,
                max(charge_trigger, relax_cap),
            )
            route_trigger = (float(charger_dist) + reserve) / float(battery_max)
            route_charge_trigger = min(trigger_cap, route_trigger)

        if route_charge_trigger is not None:
            charge_trigger = max(charge_trigger, route_charge_trigger)

        in_charger_zone = self._in_charger_zone(cur_pos)
        if self.force_charge_mode and in_charger_zone:
            hold_steps = int(getattr(self.preprocessor, "charge_phase_steps", 0))
            max_hold_steps = int(self._cfg("CHARGE_MAX_HOLD_STEPS", 30))
            min_exit_ratio = float(self._cfg("CHARGE_MIN_EXIT_BATTERY_RATIO", 0.22))
            exit_ratio = float(self._charge_exit_ratio())
            gain_steps = int(getattr(self, "current_charge_gain_steps", 0))
            if self._battery_gained_last_step() and self._last_phase_update_step_no != getattr(
                self.preprocessor, "step_no", None
            ):
                gain_steps += 1
            min_gain_steps = int(self._cfg("CHARGE_MIN_GAIN_STEPS_BEFORE_EXIT", 3))
            if gain_steps < min_gain_steps and ratio < exit_ratio:
                return
            if ratio >= exit_ratio or (
                hold_steps >= max_hold_steps and ratio >= min_exit_ratio
            ):
                self.force_charge_mode = False
                return

        if ratio <= charge_trigger:
            self.force_charge_mode = True
        elif not (self.force_charge_mode and in_charger_zone) and ratio >= self._cfg("EXIT_CHARGE_BATTERY_RATIO", 0.30):
            self.force_charge_mode = False

    def _nearest_clean_target_dist(self):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return None

        targets = self._visible_dirt_positions()
        if not targets:
            targets = list(getattr(self.preprocessor, "known_dirt_positions", set()))

        if not targets:
            known_cells = getattr(self.preprocessor, "known_cells", {})
            frontier = [
                pos
                for pos, cell in known_cells.items()
                if cell in (1, 2) and self._is_frontier_cell(pos)
            ]
            if frontier:
                visit_count = getattr(self.preprocessor, "visit_count", {})
                frontier.sort(
                    key=lambda p: (
                        self._npc_clean_suspect_ttl(p) > 0,
                        visit_count.get(p, 0),
                        self._manhattan(cur_pos, p),
                    )
                )
                targets = frontier[:60]

        if not targets:
            return None
        return min(self._manhattan(cur_pos, target) for target in targets)

    def _charge_exit_ratio(self):
        if self._flag("ENABLE_SIMPLE_CHARGE_MODE", False):
            return float(self._cfg("EXIT_CHARGE_BATTERY_RATIO", 0.40))

        battery_max = float(max(getattr(self.preprocessor, "battery_max", 1), 1))
        min_exit = float(self._cfg("CHARGE_MIN_EXIT_BATTERY_RATIO", 0.22))
        _, _, exit_pressure = self._charge_env_pressure(self._cfg("ADAPTIVE_CHARGE_EXIT_RESERVE_MAX_SCALE", 1.0))
        exit_extra = min(
            float(self._cfg("ADAPTIVE_CHARGE_EXIT_CAP_MAX_EXTRA", 0.0)),
            0.10 * max(0.0, float(exit_pressure) - 1.0),
        )
        exit_guard = float(self._cfg("EXIT_CHARGE_BATTERY_RATIO", 0.30))
        hard_cap = float(self._cfg("CHARGE_EXIT_HARD_CAP", exit_guard))
        cap = min(
            min(hard_cap, exit_guard + exit_extra),
            float(self._cfg("CHARGE_SHORT_EXIT_CAP", 0.27)) + exit_extra,
        )
        min_exit = min(exit_guard, min_exit + 0.5 * exit_extra)
        target_dist = self._nearest_clean_target_dist()
        if target_dist is None:
            return min_exit

        reserve = self._adaptive_charge_steps(
            "CHARGE_EXIT_CLEAN_RESERVE_STEPS",
            "ADAPTIVE_CHARGE_EXIT_RESERVE_MAX_SCALE",
        )
        route_ratio = (float(target_dist) + reserve) / battery_max
        return min(cap, max(min_exit, route_ratio))

    def _charger_zone_targets(self):
        charger_list = getattr(self.preprocessor, "charger_list", [])
        known_cells = getattr(self.preprocessor, "known_cells", {})
        near_dist = int(self._cfg("CHARGER_NEAR_DIST", 1))
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if not charger_list or not known_cells:
            return []

        avoid_dist = int(self._cfg("CHARGE_TARGET_NPC_AVOID_DIST", self._cfg("NPC_AVOID_DIST", 4)))
        safe_targets = []
        crowded_targets = []
        seen_targets = set()

        def add_target(pos):
            if pos in seen_targets:
                return

            cell = known_cells.get(pos)
            if cell not in (1, 2):
                cell = self._local_cell(pos)
            if cell not in (1, 2):
                return

            zone_dist = self._min_chebyshev_dist(pos, charger_list)
            if zone_dist is None or zone_dist > near_dist:
                return

            seen_targets.add(pos)
            dist_score = self._manhattan(cur_pos, pos) if cur_pos is not None else 0
            npc_dist = self._npc_dist(pos)
            if npc_dist is None or int(npc_dist) > avoid_dist:
                safe_targets.append((dist_score, pos))
            else:
                crowded_targets.append((dist_score + 100 + (avoid_dist - int(npc_dist) + 1) * 10, pos))

        for pos, cell in known_cells.items():
            if cell in (1, 2):
                add_target(pos)

        if not safe_targets and not crowded_targets:
            for charger in charger_list:
                for dx in range(-near_dist, near_dist + 1):
                    for dz in range(-near_dist, near_dist + 1):
                        add_target((int(charger[0]) + dx, int(charger[1]) + dz))

        targets = safe_targets if safe_targets else crowded_targets
        targets.sort(key=lambda item: item[0])
        return [pos for _, pos in targets]

    def _charge_hold_action(self, legal_action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        charger_list = getattr(self.preprocessor, "charger_list", [])
        if cur_pos is None or not charger_list or not self._in_charger_zone(cur_pos):
            return None

        # v8：即使开启 simple charge，也不要在充电区使用过于简单的“留在 near zone”逻辑。
        # 当前日志里多次 charger_path_dist=0/1 但 charge_gain_steps=0，说明只留在 near zone
        # 不能稳定转化为回血；下面的通用 near/final capture 逻辑会优先向更可能回血的位置推进。

        near_dist = int(self._cfg("CHARGER_NEAR_DIST", 1))
        capture_steps = int(self._cfg("CHARGE_ZONE_NO_GAIN_CAPTURE_STEPS", 1))
        no_gain_steps = int(getattr(self, "charge_zone_no_gain_steps", 0))
        path_charger_dist = self.current_charger_path_dist
        if path_charger_dist is None:
            path_charger_dist = self._charger_route_distance(charger_list)
        near_capture = bool(
            path_charger_dist is not None
            and int(path_charger_dist) <= int(self._cfg("CHARGE_NEAR_CAPTURE_PATH_MAX", 8))
        )
        final_capture = self._charge_final_capture_active(path_charger_dist)
        if near_capture:
            capture_steps = max(capture_steps, int(self._cfg("CHARGE_NEAR_CAPTURE_STEPS", 4)))
        if final_capture:
            capture_steps = max(capture_steps, int(self._cfg("CHARGE_FINAL_CAPTURE_STEPS", 8)))
        cur_zone_dist = self._charger_zone_dist(cur_pos)
        cur_raw_dist = min(self._manhattan(cur_pos, charger) for charger in charger_list)
        passable_actions = self._passable_legal_actions(legal_action)
        if not passable_actions:
            passable_actions = self._static_passable_legal_actions(legal_action)
        raw_actions = self._raw_legal_actions(legal_action)

        def score_candidate(action):
            nxt = self._next_pos(cur_pos, action)
            zone_dist = self._charger_zone_dist(nxt)
            if zone_dist is None or zone_dist > near_dist:
                return None

            cell = self._local_cell(nxt)
            if cell == 0:
                return None

            visit_n = float(getattr(self.preprocessor, "visit_count", {}).get(nxt, 0))
            score = 20.0 - float(zone_dist)
            next_raw_dist = min(self._manhattan(nxt, charger) for charger in charger_list)
            if visit_n <= 0:
                score += 1.0
            elif visit_n <= 1:
                score += 0.4
            if int(action) == int(self.last_action):
                score += 0.3
            if self._battery_gained_last_step() and int(action) == int(self.last_action):
                score += 1.5

            if no_gain_steps <= capture_steps and cur_zone_dist is not None:
                score += float(self._cfg("CHARGE_ZONE_NO_GAIN_PROGRESS_BONUS", 8.0)) * (
                    float(int(cur_zone_dist) - int(zone_dist))
                )
                if int(zone_dist) <= near_dist:
                    score += float(self._cfg("CHARGE_ZONE_NO_GAIN_HOLD_BONUS", 4.0))
                if int(zone_dist) > int(cur_zone_dist):
                    score -= float(self._cfg("CHARGE_ZONE_NO_GAIN_BACKTRACK_PENALTY", 10.0)) * (
                        float(int(zone_dist) - int(cur_zone_dist))
                    )
                if int(next_raw_dist) < int(cur_raw_dist):
                    score += 0.4 * float(self._cfg("CHARGE_ZONE_NO_GAIN_PROGRESS_BONUS", 8.0)) * (
                        float(int(cur_raw_dist) - int(next_raw_dist))
                    )
                elif int(next_raw_dist) > int(cur_raw_dist):
                    score -= 0.4 * float(self._cfg("CHARGE_ZONE_NO_GAIN_BACKTRACK_PENALTY", 10.0)) * (
                        float(int(next_raw_dist) - int(cur_raw_dist))
                    )
                if near_capture:
                    if int(zone_dist) <= near_dist:
                        score += float(self._cfg("CHARGE_NEAR_CAPTURE_HOLD_BONUS", 8.0))
                    if int(next_raw_dist) < int(cur_raw_dist):
                        score += float(self._cfg("CHARGE_NEAR_CAPTURE_PROGRESS_BONUS", 6.0)) * (
                            float(int(cur_raw_dist) - int(next_raw_dist))
                        )
                    elif int(next_raw_dist) > int(cur_raw_dist):
                        score -= float(self._cfg("CHARGE_NEAR_CAPTURE_BACKTRACK_PENALTY", 12.0)) * (
                            float(int(next_raw_dist) - int(cur_raw_dist))
                        )
                    if cur_zone_dist is not None and int(zone_dist) > int(cur_zone_dist):
                        score -= 0.6 * float(self._cfg("CHARGE_NEAR_CAPTURE_BACKTRACK_PENALTY", 12.0)) * (
                            float(int(zone_dist) - int(cur_zone_dist))
                        )
                if final_capture:
                    if int(zone_dist) <= near_dist:
                        score += float(self._cfg("CHARGE_FINAL_CAPTURE_HOLD_BONUS", 16.0))
                    if int(next_raw_dist) < int(cur_raw_dist):
                        score += float(self._cfg("CHARGE_FINAL_CAPTURE_PROGRESS_BONUS", 12.0)) * (
                            float(int(cur_raw_dist) - int(next_raw_dist))
                        )
                    elif int(next_raw_dist) > int(cur_raw_dist):
                        score -= float(self._cfg("CHARGE_FINAL_CAPTURE_BACKTRACK_PENALTY", 24.0)) * (
                            float(int(next_raw_dist) - int(cur_raw_dist))
                        )
                    if cur_zone_dist is not None and int(zone_dist) > int(cur_zone_dist):
                        score -= float(self._cfg("CHARGE_FINAL_CAPTURE_BACKTRACK_PENALTY", 24.0)) * (
                            float(int(zone_dist) - int(cur_zone_dist))
                        )

            npc_dist = self._nearest_npc_dist_after(action)
            if npc_dist is not None:
                score += 0.05 * float(npc_dist)
                if npc_dist <= self._cfg("NPC_DANGER_DIST", 2):
                    score -= 20.0

            score -= 0.2 * self._recent_position_penalty(nxt)

            return score

        best_action = None
        best_score = -10**9
        for action in passable_actions:
            score = score_candidate(action)
            if score is None:
                continue

            if score > best_score:
                best_score = score
                best_action = int(action)

        if best_action is not None:
            return best_action

        if self._battery_gained_last_step() and int(self.last_action) in raw_actions:
            nxt = self._next_pos(cur_pos, self.last_action)
            if nxt in charger_list:
                return int(self.last_action)

        return None

    def _charger_stuck_escape_action(self, legal_action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None or not self._in_charger_zone(cur_pos):
            return None

        charge_no_gain_steps = max(
            int(getattr(self.preprocessor, "charge_no_gain_steps", 0)),
            int(getattr(self, "charge_zone_no_gain_steps", 0)),
        )
        path_charger_dist = self.current_charger_path_dist
        if path_charger_dist is None:
            path_charger_dist = self._charger_route_distance(getattr(self.preprocessor, "charger_list", []))
        final_capture_escape_delay = int(self._cfg("CHARGE_FINAL_CAPTURE_ESCAPE_DELAY", 18))
        if self._charge_final_capture_active(path_charger_dist) and charge_no_gain_steps < final_capture_escape_delay:
            return None

        near_capture_path_max = int(self._cfg("CHARGE_NEAR_CAPTURE_PATH_MAX", 8))
        near_capture_escape_delay = int(self._cfg("CHARGE_NEAR_CAPTURE_ESCAPE_DELAY", 10))
        if (
            path_charger_dist is not None
            and int(path_charger_dist) <= near_capture_path_max
            and charge_no_gain_steps < near_capture_escape_delay
        ):
            return None

        if charge_no_gain_steps < int(self._cfg("CHARGE_STUCK_ESCAPE_STEPS", 5)):
            return None

        cur_zone_dist = self._charger_zone_dist(cur_pos)
        near_dist = int(self._cfg("CHARGER_NEAR_DIST", 1))
        best_action = None
        best_score = -10**9
        opposite = self._opposite_action(self.last_action)
        for action in self._passable_legal_actions(legal_action):
            nxt = self._next_pos(cur_pos, action)
            next_zone_dist = self._charger_zone_dist(nxt)
            if next_zone_dist is None:
                continue

            visit_n = float(getattr(self.preprocessor, "visit_count", {}).get(nxt, 0))
            score = 3.0 * float(next_zone_dist - cur_zone_dist)
            if next_zone_dist > near_dist:
                score += 6.0
            if visit_n <= 0:
                score += 2.0
            elif visit_n <= 1:
                score += 0.8
            if self._local_cell(nxt) == 2:
                score += 4.0
            if self._is_frontier_cell(nxt):
                score += 1.5
            score -= self._recent_position_penalty(nxt)
            if int(action) == int(self.last_action):
                score -= 8.0 + float(charge_no_gain_steps)
            if opposite is not None and int(action) == opposite:
                score += 2.0

            npc_dist = self._nearest_npc_dist_after(action)
            if npc_dist is not None:
                score += 0.1 * float(npc_dist)
                if npc_dist <= self._cfg("NPC_DANGER_DIST", 2):
                    score -= 30.0

            if score > best_score:
                best_score = score
                best_action = int(action)

        return best_action

    def _simple_low_battery_capture_action(self, legal_action):
        """低电且已经靠近充电区时，优先进入 charge entry/capture。

        v17：日志里出现 remaining_charge=0、charger_min_dist=2、workflow_direct_path=9，
        但 charge_gain_steps=0 的局。说明机器人已经靠近充电区，却没有稳定进到回血位置。
        这个兜底只在低电+近桩/近路径时触发，不参与普通清扫阶段。
        """
        if not self._flag("ENABLE_SIMPLE_CHARGE_MODE", False):
            return None
        # v18: v17 的 low-battery capture 在日志里造成多局 200/500 步电量死，
        # 因为它把“诊断出的路径距离”直接变成了硬接管。默认关闭，
        # 回充仍走稳定的 force_charge + charger_greedy 路径。
        if not self._flag("ENABLE_SIMPLE_CHARGE_LOW_BATTERY_CAPTURE", False):
            return None

        battery = getattr(self.preprocessor, "battery", None)
        battery_max = getattr(self.preprocessor, "battery_max", None)
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        charger_list = getattr(self.preprocessor, "charger_list", [])
        if battery is None or battery_max is None or battery_max <= 0 or cur_pos is None or not charger_list:
            return None

        battery_ratio = float(battery) / float(battery_max)
        if battery_ratio > float(self._cfg("SIMPLE_CHARGE_LOW_BATTERY_CAPTURE_RATIO", 0.30)):
            return None

        zone_dist = self._charger_zone_dist(cur_pos)
        min_dist = self._min_dist(cur_pos, charger_list)
        path_dist = self.current_charger_path_dist
        if path_dist is None:
            path_dist = self._charger_route_distance(charger_list)

        close_by_zone = zone_dist is not None and int(zone_dist) <= int(
            self._cfg("SIMPLE_CHARGE_LOW_BATTERY_CAPTURE_ZONE_DIST", 3)
        )
        close_by_path = path_dist is not None and int(path_dist) <= int(
            self._cfg("SIMPLE_CHARGE_LOW_BATTERY_CAPTURE_PATH_DIST", 12)
        )
        close_by_min = min_dist is not None and int(min_dist) <= int(
            self._cfg("SIMPLE_CHARGE_LOW_BATTERY_CAPTURE_MIN_DIST", 4)
        )
        if not (close_by_zone or close_by_path or close_by_min):
            return None

        if self._in_charger_zone(cur_pos):
            hold_action = self._charge_hold_action(legal_action)
            if hold_action is not None:
                return int(hold_action)

        entry_action = self._charge_entry_action(legal_action, charger_list)
        if entry_action is not None:
            return int(entry_action)

        targets = self._charger_zone_targets() or charger_list
        bfs_action = self._bfs_first_action(
            targets,
            legal_action,
            max_nodes=int(self._cfg("SIMPLE_CHARGE_BFS_LIMIT", 1400)),
        )
        return int(bfs_action) if bfs_action is not None else None

    def _charger_greedy_action(self, legal_action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        charger_list = getattr(self.preprocessor, "charger_list", [])
        if cur_pos is None or not charger_list:
            return None

        if self._in_charger_zone(cur_pos):
            return self._charge_hold_action(legal_action)

        if self._flag("ENABLE_SIMPLE_CHARGE_MODE", False):
            nearest_charger = min(charger_list, key=lambda p: self._manhattan(cur_pos, p))
            near_dist = int(self._cfg("CHARGER_NEAR_DIST", 1))
            bfs_targets = self._charger_zone_targets()

            # v8：simple charge 的最后几格也必须稳定“进桩/回血”。
            # 日志里有 charger_path_dist=0/1 仍然 FAIL_BATTERY 的行，说明普通 BFS 到 near zone
            # 不够，需要在临近充电区时先走 capture/entry 动作。
            path_charger_dist = self.current_charger_path_dist
            if path_charger_dist is None:
                if self._flag("ENABLE_SIMPLE_CHARGE_DIRECT_PROGRESS_DIST", False):
                    path_charger_dist = self._charger_route_distance_direct(bfs_targets or charger_list)
                else:
                    path_charger_dist = self._charger_route_distance(bfs_targets or charger_list)
            cur_zone_dist = self._charger_zone_dist(cur_pos)
            entry_path_max = int(self._cfg("SIMPLE_CHARGE_ENTRY_PATH_MAX", 8))
            if (
                (path_charger_dist is not None and int(path_charger_dist) <= entry_path_max)
                or (cur_zone_dist is not None and int(cur_zone_dist) <= near_dist + 2)
            ):
                entry_action = self._charge_entry_action(legal_action, charger_list)
                if entry_action is not None:
                    return int(entry_action)

            if bfs_targets:
                bfs_action = self._bfs_first_action(
                    bfs_targets,
                    legal_action,
                    max_nodes=int(self._cfg("SIMPLE_CHARGE_BFS_LIMIT", 1400)),
                )
                if bfs_action is not None:
                    return int(bfs_action)

            frontier_targets = bfs_targets or charger_list
            frontier_action = self._charger_open_frontier_bfs_action(legal_action, frontier_targets)
            if frontier_action is None:
                frontier_action = self._charger_frontier_bfs_action(legal_action, frontier_targets)
            if frontier_action is not None:
                return int(frontier_action)

            cur_dist = self._manhattan(cur_pos, nearest_charger)
            history = list(getattr(self.preprocessor, "pos_history", []))
            recent_window = int(self._cfg("RECENT_REVISIT_WINDOW", 10))
            visit_count = getattr(self.preprocessor, "visit_count", {})
            best_action = None
            best_score = -10**9
            for action in self._safe_legal_actions(legal_action):
                nxt = self._next_pos(cur_pos, action)
                charger_dist = self._manhattan(nxt, nearest_charger)
                zone_dist = self._charger_zone_dist(nxt)
                npc_dist = self._nearest_npc_dist_after(action)
                if npc_dist is None:
                    npc_dist = 99

                score = -float(charger_dist) + 0.05 * float(npc_dist)
                if int(charger_dist) < int(cur_dist):
                    score += 1.5
                if zone_dist is not None and int(zone_dist) <= near_dist:
                    score += 20.0
                visit_n = float(visit_count.get(nxt, 0))
                score -= 0.6 * visit_n
                if visit_n <= 0:
                    score += 1.0
                if nxt in history[-recent_window:]:
                    score -= 2.0
                if self._is_frontier_cell(nxt):
                    score += 0.8
                if score > best_score:
                    best_score = score
                    best_action = int(action)
            return best_action

        charger_targets = self._npc_safe_charger_targets(charger_list) or charger_list
        nearest_charger = min(charger_targets, key=lambda p: self._manhattan(cur_pos, p))
        path_charger_dist = self._charger_route_distance(charger_targets)
        charge_near_emergency = self._charge_near_emergency_pressure(path_charger_dist)
        charge_final_capture = self._charge_final_capture_active(path_charger_dist)
        charge_near_loop = self._charge_near_loop_pressure(path_charger_dist)
        charge_mid_route_loop = self._charge_mid_route_loop_pressure(path_charger_dist)
        charge_progress_stall = self._charge_progress_stall_pressure(path_charger_dist)
        route_repeat_pressure = self._charge_route_repeat_pressure()
        route_unknown_or_far = self._is_charge_route_unknown_or_far(path_charger_dist)
        search_stall_steps = int(self._cfg("CHARGE_UNKNOWN_ROUTE_STALL_STEPS", 12))
        bfs_targets = self._charger_zone_targets() or charger_targets
        if (
            path_charger_dist is not None
            and int(path_charger_dist) <= int(self._cfg("CHARGE_DIRECT_BFS_PATH_MAX", 15))
        ):
            bfs_action = self._bfs_first_action(bfs_targets, legal_action, max_nodes=4000)
            if bfs_action is not None:
                return int(bfs_action)

        if charge_final_capture:
            entry_action = self._charge_entry_action(legal_action, charger_targets)
            if entry_action is not None:
                return int(entry_action)
        if charge_near_loop:
            entry_action = self._charge_entry_action(legal_action, charger_targets)
            if entry_action is not None:
                return int(entry_action)
        if charge_near_emergency:
            entry_action = self._charge_entry_action(legal_action, charger_targets)
            if entry_action is not None:
                return int(entry_action)
        if self._charge_near_lock_active(path_charger_dist):
            entry_action = self._charge_entry_action(legal_action, charger_targets)
            if entry_action is not None:
                return int(entry_action)

        if charge_mid_route_loop and not self._charge_near_lock_active(path_charger_dist):
            if (
                path_charger_dist is not None
                and int(path_charger_dist) <= int(self._cfg("CHARGE_MID_ROUTE_LOOP_ENTRY_PATH_MAX", 10))
            ):
                entry_action = self._charge_entry_action(legal_action, charger_targets)
                if entry_action is not None:
                    return int(entry_action)

        if charge_progress_stall and not self._charge_near_lock_active(path_charger_dist):
            open_frontier_action = self._charger_open_frontier_bfs_action(legal_action, charger_targets)
            if open_frontier_action is not None:
                return int(open_frontier_action)
        bfs_action = self._bfs_first_action(bfs_targets, legal_action, max_nodes=4000)
        if bfs_action is not None:
            return int(bfs_action)

        if (
            route_unknown_or_far
            and int(getattr(self, "force_charge_search_steps", 0)) >= search_stall_steps
            and not self._charge_near_lock_active(path_charger_dist)
        ):
            open_frontier_action = self._charger_open_frontier_bfs_action(legal_action, charger_targets)
            if open_frontier_action is not None:
                return int(open_frontier_action)
        if route_repeat_pressure and not self._charge_near_lock_active(path_charger_dist):
            open_frontier_action = self._charger_open_frontier_bfs_action(legal_action, charger_targets)
            if open_frontier_action is not None:
                return int(open_frontier_action)

        frontier_action = self._charger_frontier_bfs_action(legal_action, charger_targets)
        if frontier_action is not None:
            return int(frontier_action)

        route_history = list(getattr(self, "recent_charge_route_cells", []))
        best_action = None
        best_score = -10**9
        for action in self._safe_legal_actions(legal_action):
            nxt = self._next_pos(cur_pos, action)
            zone_dist = self._charger_zone_dist(nxt)
            charger_dist = self._manhattan(nxt, nearest_charger)
            npc_dist = self._nearest_npc_dist_after(action)
            if npc_dist is None:
                npc_dist = 99
            charger_weight = 1.0
            if route_repeat_pressure and route_unknown_or_far:
                charger_weight = float(self._cfg("CHARGE_UNKNOWN_ROUTE_MANHATTAN_WEIGHT", 0.35))
            score = -charger_weight * float(charger_dist) + 0.05 * float(npc_dist)
            if charge_near_emergency:
                score += 3.0 * float(self._manhattan(cur_pos, nearest_charger) - charger_dist)
            if zone_dist is not None and zone_dist <= self._cfg("CHARGER_NEAR_DIST", 1):
                score += 20.0
                if charge_near_emergency:
                    score += 24.0
            if route_repeat_pressure:
                route_hits = float(route_history[-24:].count(nxt))
                visit_n = int(getattr(self.preprocessor, "visit_count", {}).get(nxt, 0))
                route_penalty = float(self._cfg("CHARGE_ROUTE_REPEAT_CELL_PENALTY", 2.0))
                frontier_bonus = float(self._cfg("CHARGE_ROUTE_REPEAT_FRONTIER_BONUS", 1.5))
                fresh_bonus = float(self._cfg("CHARGE_ROUTE_REPEAT_FRESH_BONUS", 2.0))
                open_bonus = 0.2
                if route_unknown_or_far:
                    route_penalty = max(
                        route_penalty,
                        float(self._cfg("CHARGE_UNKNOWN_ROUTE_REPEAT_CELL_PENALTY", 8.0)),
                    )
                    frontier_bonus = max(
                        frontier_bonus,
                        float(self._cfg("CHARGE_UNKNOWN_ROUTE_FRONTIER_BONUS", 6.0)),
                    )
                    fresh_bonus = max(
                        fresh_bonus,
                        float(self._cfg("CHARGE_UNKNOWN_ROUTE_FRESH_BONUS", 5.0)),
                    )
                    open_bonus = max(
                        open_bonus,
                        float(self._cfg("CHARGE_UNKNOWN_ROUTE_OPEN_BONUS", 0.8)),
                    )
                score -= route_penalty * route_hits
                score -= float(self._cfg("CHARGE_ROUTE_REPEAT_VISIT_PENALTY", 0.0)) * float(visit_n)
                if visit_n <= 0:
                    score += fresh_bonus
                if self._is_frontier_cell(nxt):
                    score += frontier_bonus
                score += open_bonus * float(self._local_open_neighbor_count(nxt))
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _leave_charger_action(self, legal_action):
        if self.force_charge_mode:
            return None

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        charger_list = getattr(self.preprocessor, "charger_list", [])
        battery = getattr(self.preprocessor, "battery", None)
        battery_max = getattr(self.preprocessor, "battery_max", None)
        if cur_pos is None or not charger_list or battery is None or battery_max is None or battery_max <= 0:
            return None

        battery_ratio = float(battery) / float(battery_max)
        exit_ratio = self._charge_exit_ratio()
        if not self._flag("ENABLE_SIMPLE_CHARGE_MODE", False):
            hold_steps = int(getattr(self.preprocessor, "charge_phase_steps", 0))
            if hold_steps >= int(self._cfg("CHARGE_MAX_HOLD_STEPS", 30)):
                exit_ratio = min(exit_ratio, float(self._cfg("CHARGE_MIN_EXIT_BATTERY_RATIO", 0.22)))
        if battery_ratio < exit_ratio:
            return None

        near_dist = int(self._cfg("CHARGER_NEAR_DIST", 1))
        cur_zone_dist = self._charger_zone_dist(cur_pos)
        if cur_zone_dist is None or cur_zone_dist > near_dist:
            return None

        best_action = None
        best_score = -10**9
        for action in self._safe_legal_actions(legal_action):
            nxt = self._next_pos(cur_pos, action)
            next_zone_dist = self._charger_zone_dist(nxt)
            if next_zone_dist is None:
                continue
            if int(next_zone_dist) < int(cur_zone_dist):
                continue

            visit_n = float(getattr(self.preprocessor, "visit_count", {}).get(nxt, 0))
            score = 2.0 * float(next_zone_dist - cur_zone_dist)
            if next_zone_dist > near_dist:
                score += 4.0
            elif int(next_zone_dist) > int(cur_zone_dist):
                score += 1.8
            if visit_n <= 0:
                score += 1.2
            elif visit_n <= 1:
                score += 0.4
            if self._local_cell(nxt) == 2:
                score += 3.0
            if self._is_frontier_cell(nxt):
                score += 1.0
            score -= self._recent_position_penalty(nxt)

            npc_dist = self._nearest_npc_dist_after(action)
            if npc_dist is not None and npc_dist <= self._cfg("NPC_AVOID_DIST", 4):
                score -= float(self._cfg("NPC_AVOID_DIST", 4) - npc_dist + 1) * 3.0

            if score > best_score:
                best_score = score
                best_action = int(action)

        return best_action

    def _avoid_npc_action(self, action, legal_action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        npc_list = getattr(self.preprocessor, "npc_list", [])
        if cur_pos is None or not npc_list:
            return int(action)

        danger_dist = self._cfg("NPC_DANGER_DIST", 2)
        avoid_dist = self._cfg("NPC_AVOID_DIST", 4)
        cur_min_npc = self._min_dist(cur_pos, npc_list)
        next_min_npc = self._nearest_npc_dist_after(action)
        if cur_min_npc is None or next_min_npc is None:
            return int(action)

        projected_next_min = max(0, int(next_min_npc) - 1)
        emergency_dist = self._cfg("NPC_EMERGENCY_PROJECTED_DIST", max(1, danger_dist - 1))
        cur_zone_dist = self._charger_zone_dist(cur_pos)
        battery = getattr(self.preprocessor, "battery", None)
        battery_max = getattr(self.preprocessor, "battery_max", None)
        battery_ratio = None
        if battery is not None and battery_max is not None and battery_max > 0:
            battery_ratio = float(battery) / float(battery_max)
        charger_list = getattr(self.preprocessor, "charger_list", [])
        path_charger_dist = None
        charge_route_repeat_pressure = False
        if self.force_charge_mode and charger_list:
            path_charger_dist = self._charger_route_distance(charger_list)
            charge_route_repeat_pressure = self._charge_route_repeat_pressure()
        charge_zone_no_gain_steps = int(getattr(self, "charge_zone_no_gain_steps", 0))
        near_capture_steps = int(self._cfg("CHARGE_NEAR_CAPTURE_STEPS", 4))
        near_capture_path_max = int(self._cfg("CHARGE_NEAR_CAPTURE_PATH_MAX", 8))
        near_capture_escape_delay = int(self._cfg("CHARGE_NEAR_CAPTURE_ESCAPE_DELAY", 10))
        final_capture = self._charge_final_capture_active(path_charger_dist)
        final_capture_escape_delay = int(self._cfg("CHARGE_FINAL_CAPTURE_ESCAPE_DELAY", 18))
        capture_hold_limit = max(near_capture_steps + 1, near_capture_escape_delay)
        if final_capture:
            capture_hold_limit = max(capture_hold_limit, final_capture_escape_delay)
        charge_hold_priority = bool(
            self.force_charge_mode
            and self._in_charger_zone(cur_pos)
            and (
                (battery_ratio is not None and battery_ratio < self._charge_exit_ratio())
                or (
                    path_charger_dist is not None
                    and int(path_charger_dist) <= near_capture_path_max
                    and charge_zone_no_gain_steps <= capture_hold_limit
                )
            )
        )
        if charge_hold_priority and projected_next_min > emergency_dist:
            next_zone_dist = self._charger_zone_dist(self._next_pos(cur_pos, action))
            if next_zone_dist is not None and int(next_zone_dist) <= int(self._cfg("CHARGER_NEAR_DIST", 1)):
                return int(action)
        near_charge_lock = self._charge_near_lock_active(path_charger_dist)
        charge_near_loop = self._charge_near_loop_pressure(path_charger_dist)
        charge_progress_stall = self._charge_progress_stall_pressure(path_charger_dist)
        charge_commit_priority = bool(
            self.force_charge_mode
            and not charge_hold_priority
            and not near_charge_lock
            and not final_capture
            and not charge_near_loop
            and not charge_route_repeat_pressure
            and not charge_progress_stall
            and path_charger_dist is not None
            and int(path_charger_dist) <= int(self._cfg("CHARGE_COMMIT_MAX_PATH_DIST", 18))
        )
        charge_near_emergency = self._charge_near_emergency_pressure(path_charger_dist)
        if (
            not charge_hold_priority
            and projected_next_min > emergency_dist
            and int(cur_min_npc) > int(danger_dist)
        ):
            if (
                near_charge_lock
                or final_capture
                or charge_near_emergency
                or charge_near_loop
            ) and self._action_advances_to_charger(action, charger_list):
                return int(action)
            if (
                not charge_commit_priority
                and not charge_near_emergency
                and not charge_near_loop
                and not near_charge_lock
                and not final_capture
            ):
                return int(action)

            if self._action_advances_to_charger(action, charger_list):
                return int(action)

        nearest_charger = None
        if self.force_charge_mode and charger_list:
            nearest_charger = min(charger_list, key=lambda p: self._manhattan(cur_pos, p))

        best_action = int(action)
        best_score = -10**9
        near_dist = int(self._cfg("CHARGER_NEAR_DIST", 1))
        for candidate in self._safe_legal_actions(legal_action):
            nxt = self._next_pos(cur_pos, candidate)
            npc_dist = self._min_dist(nxt, npc_list)
            if npc_dist is None:
                npc_dist = 99
            projected_dist = max(0, int(npc_dist) - 1)
            next_zone_dist = self._charger_zone_dist(nxt)

            score = 6.0 * float(projected_dist) + float(npc_dist)
            if projected_dist <= danger_dist:
                score -= 200.0 + float(danger_dist - projected_dist + 1) * 40.0
            elif projected_dist <= avoid_dist:
                score -= float(avoid_dist - projected_dist + 1) * 8.0
            if cur_min_npc <= avoid_dist:
                if npc_dist > cur_min_npc:
                    score += 5.0 * float(npc_dist - cur_min_npc)
                else:
                    score -= 10.0 * float(cur_min_npc - npc_dist + 1)

            cell = self._local_cell(nxt)
            if cell == 2:
                score += 2.5
            score -= self._recent_position_penalty(nxt)
            score -= 0.6 * float(getattr(self.preprocessor, "visit_count", {}).get(nxt, 0))

            if nearest_charger is not None:
                cur_charger_dist = float(self._manhattan(cur_pos, nearest_charger))
                next_charger_dist = float(self._manhattan(nxt, nearest_charger))
                score -= 0.5 * next_charger_dist
                if near_charge_lock and cur_zone_dist is not None and next_zone_dist is not None:
                    score += float(self._cfg("CHARGE_NEAR_ENTRY_PROGRESS_BONUS", 18.0)) * (
                        float(int(cur_zone_dist) - int(next_zone_dist))
                    )
                    if int(next_zone_dist) <= near_dist:
                        score += float(self._cfg("CHARGE_NEAR_ENTRY_ZONE_BONUS", 48.0))
                    elif int(next_zone_dist) > int(cur_zone_dist):
                        score -= float(self._cfg("CHARGE_NEAR_ENTRY_BACKTRACK_PENALTY", 28.0)) * (
                            float(int(next_zone_dist) - int(cur_zone_dist))
                        )
                if charge_commit_priority:
                    score += float(self._cfg("CHARGE_COMMIT_PROGRESS_BONUS", 12.0)) * (
                        cur_charger_dist - next_charger_dist
                    )
                    if next_charger_dist > cur_charger_dist:
                        score -= float(self._cfg("CHARGE_COMMIT_BACKTRACK_PENALTY", 18.0)) * (
                            next_charger_dist - cur_charger_dist
                        )
                    if cur_zone_dist is not None and next_zone_dist is not None:
                        if int(next_zone_dist) < int(cur_zone_dist):
                            score += float(self._cfg("CHARGE_COMMIT_ZONE_BONUS", 16.0)) * (
                                int(cur_zone_dist) - int(next_zone_dist)
                            )
                        elif int(next_zone_dist) > int(cur_zone_dist):
                            score -= 0.5 * float(int(next_zone_dist) - int(cur_zone_dist))
                if charge_near_emergency:
                    score += 18.0 * (cur_charger_dist - next_charger_dist)
                    if next_charger_dist > cur_charger_dist:
                        score -= 28.0 * (next_charger_dist - cur_charger_dist)
                if final_capture and cur_zone_dist is not None and next_zone_dist is not None:
                    if int(next_zone_dist) < int(cur_zone_dist):
                        score += float(self._cfg("CHARGE_FINAL_CAPTURE_PROGRESS_BONUS", 12.0))
                    elif int(next_zone_dist) > int(cur_zone_dist):
                        score -= float(self._cfg("CHARGE_FINAL_CAPTURE_BACKTRACK_PENALTY", 24.0))

            if charge_hold_priority:
                if next_zone_dist is not None and int(next_zone_dist) <= near_dist:
                    score += 90.0 + 8.0 * float(near_dist - int(next_zone_dist))
                else:
                    score -= 70.0
            elif near_charge_lock:
                if next_zone_dist is not None and int(next_zone_dist) <= near_dist:
                    score += 60.0 + 6.0 * float(near_dist - int(next_zone_dist))
                elif cur_zone_dist is not None and next_zone_dist is not None and int(next_zone_dist) > int(cur_zone_dist):
                    score -= 45.0
            elif charge_near_emergency:
                if next_zone_dist is not None and cur_zone_dist is not None:
                    if int(next_zone_dist) < int(cur_zone_dist):
                        score += 42.0
                    elif int(next_zone_dist) > int(cur_zone_dist):
                        score -= 52.0
                elif self._action_advances_to_charger(candidate, charger_list):
                    score += 20.0

            if int(candidate) == int(action):
                score += 0.1

            if score > best_score:
                best_score = score
                best_action = int(candidate)
        return best_action

    def _escape_if_stuck(self, legal_action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return None

        stuck_steps = int(getattr(self.preprocessor, "stuck_steps", 0))
        recent_loop = self._is_recent_loop()
        if stuck_steps < 2 and not recent_loop:
            return None

        legal_actions = self._strict_escape_actions(legal_action)
        if not legal_actions:
            return None

        severe_stuck = stuck_steps >= int(self._cfg("SEVERE_STUCK_ESCAPE_STEPS", 6))
        if not self.force_charge_mode and not severe_stuck:
            for rescue_action in (
                self._visible_dirt_bfs_action(legal_action),
                self._known_dirt_bfs_action(legal_action),
                self._frontier_bfs_action(legal_action),
            ):
                if rescue_action is not None and rescue_action in legal_actions:
                    return int(rescue_action)

        charger_list = getattr(self.preprocessor, "charger_list", [])
        nearest_charger = None
        if self.force_charge_mode and charger_list:
            nearest_charger = min(charger_list, key=lambda p: self._manhattan(cur_pos, p))

        history = list(getattr(self.preprocessor, "pos_history", []))
        blocked_cells = getattr(self.preprocessor, "blocked_cell_ttl", {})
        best_action = None
        best_score = -10**9
        opposite = self._opposite_action(self.last_action)
        for action in legal_actions:
            nxt = self._next_pos(cur_pos, action)
            cell = self._local_cell(nxt)
            if cell == 0:
                continue

            visit_n = float(getattr(self.preprocessor, "visit_count", {}).get(nxt, 0))
            score = -2.0 * visit_n
            open_neighbors = self._local_open_neighbor_count(nxt)
            score += 0.45 * float(open_neighbors)
            if visit_n <= 0:
                score += 1.2
            elif visit_n <= 1:
                score += 0.4
            if self._is_frontier_cell(nxt):
                score += self._cfg("EXPLORE_FRONTIER_STUCK_BONUS", 1.8)
            score -= self._recent_position_penalty(nxt)
            if severe_stuck:
                score += 0.75 * float(open_neighbors)
                score -= 2.0 * float(history[-12:].count(nxt))
                if blocked_cells.get(nxt, 0) > 0:
                    score -= 12.0
                if visit_n <= 0:
                    score += 2.0
            elif recent_loop:
                score -= 1.5 * float(history[-8:].count(nxt))
            if cell == 2:
                score += 3.0
            if int(action) == int(self.last_action):
                score -= 4.0 + float(stuck_steps)
            if opposite is not None and int(action) == opposite:
                score += 0.8 + 0.4 * float(stuck_steps)

            npc_dist = self._nearest_npc_dist_after(action)
            if npc_dist is not None:
                score += 0.2 * float(npc_dist)

            if nearest_charger is not None:
                score -= 0.35 * float(self._manhattan(nxt, nearest_charger))

            if score > best_score:
                best_score = score
                best_action = int(action)

        return best_action

    def _progress_targets(self):
        if self.force_charge_mode:
            return []

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        visible_targets = self._visible_dirt_positions()
        if visible_targets:
            return visible_targets[:40]

        known_dirt = list(getattr(self.preprocessor, "known_dirt_positions", set()))
        if cur_pos is not None and known_dirt:
            known_dirt.sort(key=lambda p: self._manhattan(cur_pos, p))
        return known_dirt[:80]

    def _progress_action_score(self, action, targets=None, legal_action=None):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return -10**9

        action = int(action)
        if not self._action_is_passable(cur_pos, action):
            return -10**9

        nxt = self._next_pos(cur_pos, action)
        history = list(getattr(self.preprocessor, "pos_history", []))
        visit_count = getattr(self.preprocessor, "visit_count", {})
        visit_n = float(visit_count.get(nxt, 0))

        score = -2.3 * visit_n
        score -= self._recent_position_penalty(nxt)
        if self._local_cell(nxt) == 2:
            score += 5.0
        if self._is_frontier_cell(nxt):
            score += 1.3
        if visit_n <= 0:
            score += 2.0
        elif visit_n <= 1:
            score += 0.7

        post_charge_pressure = self._post_charge_clean_pressure()
        post_charge_escape_pressure = self._post_charge_escape_pressure()
        clean_escape_pressure = self._clean_escape_pressure()
        if post_charge_pressure:
            if self._post_charge_route_revisit(nxt):
                score -= float(self._cfg("POST_CHARGE_ROUTE_SCORE_PENALTY", 3.0))
            cur_zone_dist = self._charger_zone_dist(cur_pos)
            next_zone_dist = self._charger_zone_dist(nxt)
            near_dist = int(self._cfg("POST_CHARGE_NEAR_CHARGER_DIST", 4))
            if (
                cur_zone_dist is not None
                and next_zone_dist is not None
                and int(next_zone_dist) <= int(cur_zone_dist)
                and int(next_zone_dist) <= near_dist
            ):
                score -= float(self._cfg("POST_CHARGE_NEAR_CHARGER_SCORE_PENALTY", 2.0))
        if post_charge_escape_pressure and self._local_cell(nxt) != 2:
            if visit_n > 0:
                score -= float(self._cfg("POST_CHARGE_ESCAPE_VISIT_SCORE_PENALTY", 1.6)) * min(2.0, visit_n)
            if self._post_charge_route_revisit(nxt):
                score -= float(self._cfg("POST_CHARGE_ESCAPE_ROUTE_SCORE_PENALTY", 4.8))
            if visit_n <= 0:
                score += 1.1
            if self._is_frontier_cell(nxt):
                score += 1.0
            if self._local_open_neighbor_count(nxt) >= int(self._cfg("OPEN_AREA_MIN_NEIGHBORS", 5)):
                score += 0.6

        if self._clean_repeat_pressure():
            score -= float(self._cfg("CLEAN_REPEAT_PROGRESS_VISIT_PENALTY", 1.0)) * visit_n
            if visit_n <= 0:
                score += float(self._cfg("CLEAN_REPEAT_PROGRESS_FRESH_BONUS", 1.2))
            if self._is_frontier_cell(nxt):
                score += float(self._cfg("CLEAN_REPEAT_FRONTIER_SCORE_BONUS", 1.0))
            if self._post_charge_route_revisit(nxt):
                score -= float(self._cfg("CLEAN_REPEAT_ROUTE_SCORE_PENALTY", 2.5))

        if clean_escape_pressure and self._local_cell(nxt) != 2:
            score -= float(self._cfg("CLEAN_ESCAPE_VISIT_SCORE_PENALTY", 2.2)) * min(2.0, visit_n)
            if visit_n <= 0:
                score += float(self._cfg("CLEAN_ESCAPE_FRESH_SCORE_BONUS", 2.4))
            if self._is_frontier_cell(nxt):
                score += float(self._cfg("CLEAN_ESCAPE_FRONTIER_SCORE_BONUS", 1.6))
            if self._local_open_neighbor_count(nxt) >= int(self._cfg("OPEN_AREA_MIN_NEIGHBORS", 5)):
                score += float(self._cfg("CLEAN_ESCAPE_OPEN_SCORE_BONUS", 0.8))
            if self._post_charge_route_revisit(nxt):
                score -= 2.4

        suspect_ttl = float(self._npc_clean_suspect_ttl(nxt))
        if suspect_ttl > 0 and self._local_cell(nxt) != 2 and not self.force_charge_mode:
            suspect_penalty = float(self._cfg("NPC_CLEAN_PROGRESS_PENALTY", 1.8))
            if clean_escape_pressure:
                suspect_penalty = max(
                    suspect_penalty,
                    float(self._cfg("NPC_CLEAN_ESCAPE_PROGRESS_PENALTY", 4.2)),
                )
            elif self._clean_repeat_pressure() or self._repeat_area_pressure(legal_action):
                suspect_penalty = max(
                    suspect_penalty,
                    float(self._cfg("NPC_CLEAN_REPEAT_PROGRESS_PENALTY", 2.8)),
                )
            if self._is_frontier_cell(nxt) and visit_n <= 0:
                suspect_penalty *= 0.55
            elif visit_n <= 0:
                suspect_penalty *= 0.75
            score -= suspect_penalty

        window = int(self._cfg("RECENT_REVISIT_WINDOW", 10))
        if nxt in history[-window:]:
            score -= 3.5
            if self._clean_repeat_pressure():
                score -= float(self._cfg("CLEAN_REPEAT_PROGRESS_RECENT_PENALTY", 0.0))

        if targets:
            cur_target_dist = min(self._manhattan(cur_pos, target) for target in targets)
            next_target_dist = min(self._manhattan(nxt, target) for target in targets)
            score += 0.55 * float(cur_target_dist - next_target_dist)

        npc_dist = self._nearest_npc_dist_after(action)
        if npc_dist is not None:
            avoid_dist = self._cfg("NPC_AVOID_DIST", 4)
            cur_min_npc = self._npc_dist(cur_pos)
            if npc_dist <= avoid_dist:
                score -= 0.75 * float(avoid_dist - npc_dist + 1)
            if cur_min_npc is not None and cur_min_npc <= avoid_dist:
                if npc_dist > cur_min_npc:
                    score += 0.8 * float(npc_dist - cur_min_npc)
                else:
                    score -= 0.8 * float(cur_min_npc - npc_dist + 1)

        projected_npc_heat = self._projected_npc_heat(nxt)
        if projected_npc_heat > 0.0:
            projected_penalty = float(self._cfg("NPC_PROJECT_SCORE_PENALTY", 2.8))
            if self.force_charge_mode and self._charge_near_emergency_pressure(self.current_charger_path_dist):
                projected_penalty *= 0.65
            score -= projected_penalty * float(projected_npc_heat)

        return score

    def _best_progress_action(self, legal_action, return_score=False):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return (None, -10**9) if return_score else None

        candidates = self._safe_legal_actions(legal_action)
        if not candidates:
            return (None, -10**9) if return_score else None

        targets = self._progress_targets()

        best_action = None
        best_score = -10**9
        for action in candidates:
            score = self._progress_action_score(action, targets, legal_action)
            if score > best_score:
                best_score = score
                best_action = int(action)

        return (best_action, best_score) if return_score else best_action

    def _should_follow_visible_dirt(self, action):
        if self._action_hits_dirt(action):
            return False

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        targets = self._visible_dirt_positions()
        if cur_pos is None or not targets:
            return False

        cur_dist = min(self._manhattan(cur_pos, target) for target in targets)
        if cur_dist > 8:
            return False

        nxt = self._next_pos(cur_pos, action)
        next_dist = min(self._manhattan(nxt, target) for target in targets)
        return next_dist >= cur_dist or self._action_revisits_recent(action)

    def _fresh_local_progress_action(self, legal_action):
        """清扫重复高时优先选择一步内的新格/边界格。

        这是很局部的反重复救援：
        - 只看当前 8 邻域，不做远距离强制导航；
        - 不在 force_charge 阶段生效；
        - 只返回模型本步可直接执行的安全动作；
        - 目标是降低 CleanRepeat，而不是把回充路线重复当坏重复处理。
        """
        if self.force_charge_mode:
            return None

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return None

        legal_actions = self._safe_legal_actions(legal_action)
        if not legal_actions:
            return None

        history = list(getattr(self.preprocessor, "pos_history", []))
        visit_count = getattr(self.preprocessor, "visit_count", {})
        best_action = None
        best_score = -10**9
        has_strong_candidate = False

        for candidate in legal_actions:
            candidate = int(candidate)
            if not self._action_is_passable(cur_pos, candidate):
                continue

            nxt = self._next_pos(cur_pos, candidate)
            cell = self._local_cell(nxt)
            if cell == 0:
                continue

            visit_n = int(visit_count.get(nxt, 0))
            is_fresh = visit_n <= 0
            is_frontier = self._is_frontier_cell(nxt)
            hits_dirt = cell == 2

            # 没有新格、没有污渍、也不是 frontier 的动作，不作为局部救援首选。
            if not (is_fresh or is_frontier or hits_dirt):
                continue

            score = self._progress_action_score(candidate, legal_action=legal_action)
            if hits_dirt:
                score += 8.0
                has_strong_candidate = True
            if is_fresh:
                score += float(self._cfg("LOCAL_FRESH_PROGRESS_BONUS", 5.0))
                has_strong_candidate = True
            if is_frontier:
                score += float(self._cfg("LOCAL_FRONTIER_PROGRESS_BONUS", 3.0))
                has_strong_candidate = True
            if nxt in history[-int(self._cfg("RECENT_REVISIT_WINDOW", 10)):]:
                score -= float(self._cfg("LOCAL_RECENT_REVISIT_PENALTY", 6.0))
            score -= 1.5 * float(max(visit_n, 0))

            npc_dist = self._nearest_npc_dist_after(candidate)
            if npc_dist is not None and int(npc_dist) <= int(self._cfg("NPC_DANGER_DIST", 2)):
                score -= 100.0

            if score > best_score:
                best_score = score
                best_action = candidate

        if has_strong_candidate:
            return best_action

        # v5：如果一格内已经没有“全新格/污渍/frontier”，不要立刻放弃。
        # 很多 CleanRepeat=0.5+ 的局不是卡墙，而是在一片已访问区域里来回转；
        # 这时选择“低访问、非近期、开阔”的邻格，比继续远距离 BFS 更温和。
        if not self._flag("ENABLE_LOCAL_LOW_VISIT_PROGRESS_RESCUE", True):
            return None

        low_visit_limit = int(self._cfg("LOCAL_LOW_VISIT_MAX_VISITS", 2))
        best_action = None
        best_score = -10**9
        recent_window = int(self._cfg("RECENT_REVISIT_WINDOW", 10))
        recent_set = set(history[-max(1, recent_window):])
        for candidate in legal_actions:
            candidate = int(candidate)
            if not self._action_is_passable(cur_pos, candidate):
                continue

            nxt = self._next_pos(cur_pos, candidate)
            cell = self._local_cell(nxt)
            if cell == 0:
                continue

            visit_n = int(visit_count.get(nxt, 0))
            if visit_n > low_visit_limit and nxt in recent_set:
                continue

            score = self._progress_action_score(candidate, legal_action=legal_action)
            score += float(self._cfg("LOCAL_LOW_VISIT_PROGRESS_BONUS", 2.4)) * max(
                0.0,
                float(low_visit_limit + 1 - min(visit_n, low_visit_limit + 1)),
            )
            score += float(self._cfg("LOCAL_LOW_VISIT_OPEN_BONUS", 0.6)) * float(
                self._local_open_neighbor_count(nxt)
            )
            if nxt in recent_set:
                score -= float(self._cfg("LOCAL_RECENT_REVISIT_PENALTY", 7.0))
            if int(candidate) == int(self.last_action):
                score += 0.15

            npc_dist = self._nearest_npc_dist_after(candidate)
            if npc_dist is not None and int(npc_dist) <= int(self._cfg("NPC_DANGER_DIST", 2)):
                score -= 100.0

            if score > best_score:
                best_score = score
                best_action = candidate

        return best_action


    def _should_nudge_progress(self, action, legal_action):
        if self.force_charge_mode:
            return None

        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return None

        action = int(action)
        self._progress_rescue_kind = "progress_repeat_rescue"
        nxt = self._next_pos(cur_pos, action)
        visit_count = getattr(self.preprocessor, "visit_count", {})
        visit_n = int(visit_count.get(nxt, 0))
        # 当前动作如果能直接踩到污渍，不要被“局部新格救援”误覆盖。
        # v4 初版这里漏定义 raw_hits_dirt，导致 predict 抛 NameError。
        raw_hits_dirt = self._action_hits_dirt(action)
        threshold = int(self._cfg("REPEAT_VISIT_THRESHOLD", 2))
        stall_steps = int(getattr(self.preprocessor, "steps_since_clean", 0))
        stall_threshold = int(self._cfg("LOW_EFFICIENCY_STALL_STEPS", 35))
        targets_available = bool(self._progress_targets())
        frontier_action = None
        if stall_steps >= stall_threshold and not targets_available:
            frontier_action = self._frontier_bfs_action(legal_action)
        repeat_pressure = self._repeat_area_pressure(legal_action)
        post_charge_pressure = self._post_charge_clean_pressure()
        post_charge_escape_pressure = self._post_charge_escape_pressure()
        clean_repeat_pressure = self._clean_repeat_pressure()
        clean_escape_pressure = self._clean_escape_pressure()

        low_progress = (
            self._action_revisits_recent(action)
            or visit_n >= threshold
            or repeat_pressure
            or (
                clean_repeat_pressure
                and (
                    visit_n > 0
                    or self._post_charge_route_revisit(nxt)
                    or self._action_revisits_recent(action)
                )
            )
            or (
                post_charge_pressure
                and (visit_n > 0 or self._post_charge_route_revisit(nxt))
            )
            or (
                post_charge_escape_pressure
                and (visit_n > 0 or self._post_charge_route_revisit(nxt) or self._action_revisits_recent(action))
            )
            or (
                clean_escape_pressure
                and (
                    visit_n > 0
                    or self._action_revisits_recent(action)
                    or self._post_charge_route_revisit(nxt)
                )
            )
            or (stall_steps >= stall_threshold and (targets_available or frontier_action is not None))
        )
        if not low_progress:
            return None

        # v4：如果当前动作明显会回到旧格，而旁边存在安全的新格/边界格，
        # 先做一步局部反重复救援；这比频繁 BFS 强制导航更轻。
        local_fresh_min_stall = int(self._cfg("LOCAL_FRESH_RESCUE_MIN_STALL_STEPS", 36))
        if (
            self._flag("ENABLE_LOCAL_FRESH_PROGRESS_RESCUE", True)
            and not raw_hits_dirt
            and stall_steps >= local_fresh_min_stall
        ):
            local_fresh_action = self._fresh_local_progress_action(legal_action)
            if local_fresh_action is not None and int(local_fresh_action) != int(action):
                cur_score = self._progress_action_score(action, legal_action=legal_action)
                fresh_score = self._progress_action_score(local_fresh_action, legal_action=legal_action)
                # 清扫重复压力下允许较小 margin，因为目标是降低 CleanRepeat，不是追远处 BFS 目标。
                local_margin = float(self._cfg("LOCAL_FRESH_PROGRESS_MARGIN", -0.25))
                if fresh_score >= cur_score + local_margin or self._action_revisits_recent(action):
                    self._progress_rescue_kind = "local_fresh_rescue"
                    return int(local_fresh_action)

        # v5：远距离 BFS 进度救援只在“真的长时间没清扫/明显循环/逃逸压力”下启用。
        # 之前每次 recent revisit 都可能走 BFS，rule_override 看似救援，实际会沿旧走廊反复跑。
        route_rescue_min_stall = int(self._cfg("PROGRESS_ROUTE_RESCUE_MIN_STALL_STEPS", 24))
        allow_route_rescue = bool(
            stall_steps >= route_rescue_min_stall
            or clean_escape_pressure
            or post_charge_escape_pressure
            or (repeat_pressure and self._is_recent_loop())
            or (
                clean_repeat_pressure
                and self._clean_phase_repeat_ratio() >= float(self._cfg("CLEAN_REPEAT_PRESSURE_HARD_RATIO", 0.18))
                and stall_steps >= max(4, route_rescue_min_stall // 2)
            )
        )
        if not allow_route_rescue:
            return None

        targets = self._progress_targets()
        candidate_actions = []
        local_action, local_score = self._best_progress_action(legal_action, return_score=True)
        if local_action is not None:
            candidate_actions.append((int(local_action), float(local_score)))

        route_candidates = [
            self._visible_dirt_bfs_action(legal_action),
            self._known_dirt_bfs_action(legal_action),
        ]
        if stall_steps >= stall_threshold:
            route_candidates.append(frontier_action if frontier_action is not None else self._frontier_bfs_action(legal_action))
        if repeat_pressure:
            route_candidates.extend([
                self._frontier_bfs_action(legal_action),
                self._open_frontier_bfs_action(legal_action),
            ])
        if clean_repeat_pressure:
            route_candidates.extend([
                self._known_dirt_bfs_action(legal_action),
                self._frontier_bfs_action(legal_action),
                self._open_frontier_bfs_action(legal_action),
            ])
        if clean_escape_pressure:
            route_candidates.extend([
                self._frontier_bfs_action(legal_action),
                self._open_frontier_bfs_action(legal_action),
            ])
        if post_charge_pressure:
            route_candidates.extend([
                self._known_dirt_bfs_action(legal_action),
                self._frontier_bfs_action(legal_action),
                self._open_frontier_bfs_action(legal_action),
            ])
        if post_charge_escape_pressure:
            route_candidates.extend([
                self._frontier_bfs_action(legal_action),
                self._open_frontier_bfs_action(legal_action),
            ])

        for candidate in route_candidates:
            if candidate is None:
                continue
            candidate = int(candidate)
            score = self._progress_action_score(candidate, targets, legal_action)
            candidate_actions.append((candidate, float(score)))

        if not candidate_actions:
            return None

        progress_action, best_score = max(candidate_actions, key=lambda item: item[1])

        if progress_action is None or int(progress_action) == action:
            return None

        cur_score = self._progress_action_score(action, targets, legal_action)
        margin = float(self._cfg("PROGRESS_OVERRIDE_SCORE_MARGIN", 1.2))
        if repeat_pressure:
            margin = min(margin, float(self._cfg("PROGRESS_OVERRIDE_REPEAT_MARGIN", margin)))
        if clean_repeat_pressure:
            margin = min(margin, float(self._cfg("PROGRESS_OVERRIDE_CLEAN_REPEAT_MARGIN", margin)))
        if clean_escape_pressure:
            margin = min(margin, 1.0)
        if post_charge_pressure:
            margin = min(margin, float(self._cfg("PROGRESS_OVERRIDE_POST_CHARGE_MARGIN", margin)))
        if post_charge_escape_pressure:
            margin = min(margin, 0.9)
        if best_score >= cur_score + margin:
            self._progress_rescue_kind = "progress_route_rescue"
            return int(progress_action)

        return None

    def _mark_rule_override(self, current_action, new_action, reason):
        """记录规则接管原因，并返回新动作。只记录真实改变动作的第一次原因。"""
        if new_action is None:
            return int(current_action)
        new_action = int(new_action)
        current_action = int(current_action)
        if new_action != current_action and getattr(self, "_pending_rule_override_reason", "none") == "none":
            self._pending_rule_override_reason = str(reason)
        return new_action

    def _rule_adjust_action(self, action, legal_action):
        """安全兜底 + 极少量重复救援。

        这一步之前是 rule_override_rate 偏高的主要来源：
        - 低电/充电/NPC/卡死兜底保留；
        - 清扫阶段 progress override 改成“更明确压力 + 冷却时间”才触发；
        - 记录 last_rule_override_reason，方便日志判断到底是哪类规则在接管。
        """
        self._pending_rule_override_reason = "none"
        legal_actions = self._safe_legal_actions(legal_action)
        if not legal_actions:
            return int(action)

        action = int(action)
        if action not in legal_actions:
            fallback_action = self._best_progress_action(legal_action)
            new_action = int(fallback_action if fallback_action is not None else legal_actions[0])
            action = self._mark_rule_override(action, new_action, "invalid_or_unsafe_action")

        self._update_force_charge_mode()
        simple_charge = self._flag("ENABLE_SIMPLE_CHARGE_MODE", False)
        charge_escaped = False

        if simple_charge and self._flag("ENABLE_SIMPLE_CHARGE_LOW_BATTERY_CAPTURE", False):
            low_capture_action = self._simple_low_battery_capture_action(legal_action)
            if low_capture_action is not None:
                self.force_charge_mode = True
                action = self._mark_rule_override(action, low_capture_action, "low_battery_charge_capture")

        if self.force_charge_mode:
            charge_escape_action = None if simple_charge else self._charger_stuck_escape_action(legal_action)
            if charge_escape_action is not None:
                action = self._mark_rule_override(action, charge_escape_action, "charge_stuck_escape")
                charge_escaped = True
            else:
                charge_action = self._charger_greedy_action(legal_action)
                if charge_action is not None:
                    action = self._mark_rule_override(action, charge_action, "force_charge")

        if not self.force_charge_mode:
            leave_charger_action = self._leave_charger_action(legal_action)
            if leave_charger_action is not None:
                action = self._mark_rule_override(action, leave_charger_action, "leave_charger")

        in_charge_zone = self._in_charger_zone()
        near_charge_lock = False
        if self.force_charge_mode and not in_charge_zone and not simple_charge:
            near_charge_lock = self._charge_near_lock_active(
                self._charger_route_distance(getattr(self.preprocessor, "charger_list", []))
            )
        escape_action = None if (charge_escaped or (self.force_charge_mode and (in_charge_zone or near_charge_lock))) else self._escape_if_stuck(legal_action)
        escaped = escape_action is not None
        if escape_action is not None:
            action = self._mark_rule_override(action, escape_action, "stuck_or_loop_escape")

        if self._flag("ENABLE_PROGRESS_RULE_OVERRIDE", False) and not escaped and not self.force_charge_mode:
            step_no = int(getattr(self.preprocessor, "step_no", 0))
            stall_steps = int(getattr(self.preprocessor, "steps_since_clean", 0))
            min_stall_steps = int(self._cfg("PROGRESS_OVERRIDE_MIN_STALL_STEPS", 40))
            loop_min_step = int(self._cfg("PROGRESS_OVERRIDE_LOOP_MIN_STEP", 80))
            cooldown_steps = int(self._cfg("PROGRESS_OVERRIDE_COOLDOWN_STEPS", 8))
            cooldown_ok = (step_no - int(getattr(self, "last_progress_override_step", -10**9))) >= max(0, cooldown_steps)

            raw_hits_dirt = self._action_hits_dirt(action)
            recent_loop = self._is_recent_loop() and step_no >= loop_min_step
            clean_escape_pressure = self._clean_escape_pressure()
            # v6：CleanRepeat 高但最近仍在清扫时，不再立刻硬接管；
            # 普通反重复交给 _anti_repeat_prob_adjust 的 soft bias。
            hard_repeat = (
                self._action_under_clean_repeat_pressure(action)
                and not raw_hits_dirt
                and (
                    stall_steps >= min_stall_steps
                    or recent_loop
                    or self._clean_escape_pressure()
                )
            )
            recent_revisit_pressure = (
                self._action_revisits_recent(action)
                and stall_steps >= min_stall_steps
                and not raw_hits_dirt
            )
            if not bool(self._cfg("PROGRESS_OVERRIDE_RECENT_REVISIT_NEEDS_STALL", True)):
                recent_revisit_pressure = self._action_revisits_recent(action) and not raw_hits_dirt

            should_rescue_repeat = bool(
                cooldown_ok
                and (recent_loop or hard_repeat or clean_escape_pressure or recent_revisit_pressure)
            )
            progress_action = self._should_nudge_progress(action, legal_action) if should_rescue_repeat else None
            if progress_action is not None:
                progress_reason = getattr(self, "_progress_rescue_kind", "progress_repeat_rescue")
                new_action = self._mark_rule_override(action, progress_action, progress_reason)
                if int(new_action) != int(action):
                    self.last_progress_override_step = step_no
                action = new_action

        before_npc = int(action)
        npc_action = int(self._avoid_npc_action(action, legal_action))
        if npc_action != before_npc:
            action = self._mark_rule_override(before_npc, npc_action, "npc_avoid")
        else:
            action = npc_action
        return int(action)

    def _anti_repeat_prob_adjust(self, prob, legal_action):
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None:
            return self._normalize_prob(prob, legal_action)

        legal_arr = np.array(legal_action, dtype=np.float64)
        adjusted = np.asarray(prob, dtype=np.float64).copy()
        history = list(getattr(self.preprocessor, "pos_history", []))
        visit_count = getattr(self.preprocessor, "visit_count", {})
        blocked_ttl = getattr(self.preprocessor, "blocked_action_ttl", [0] * Config.ACTION_NUM)
        recent_window = int(self._cfg("RECENT_REVISIT_WINDOW", 10))
        repeat_threshold = int(self._cfg("REPEAT_VISIT_THRESHOLD", 2))
        visible_targets = [] if self.force_charge_mode else self._visible_dirt_positions()
        cur_dirt_dist = None
        if visible_targets:
            cur_dirt_dist = min(self._manhattan(cur_pos, target) for target in visible_targets)

        npc_list = getattr(self.preprocessor, "npc_list", [])
        cur_min_npc = self._min_dist(cur_pos, npc_list) if npc_list else None
        danger_dist = self._cfg("NPC_DANGER_DIST", 2)
        avoid_dist = self._cfg("NPC_AVOID_DIST", 4)
        stall_steps = int(getattr(self.preprocessor, "steps_since_clean", 0))
        stall_threshold = int(self._cfg("LOW_EFFICIENCY_STALL_STEPS", 35))
        has_fresh_adjacent = False
        for candidate in self._legal_actions(legal_action):
            candidate_nxt = self._next_pos(cur_pos, candidate)
            if self._action_is_passable(cur_pos, candidate) and int(visit_count.get(candidate_nxt, 0)) <= 0:
                has_fresh_adjacent = True
                break
        has_adjacent_dirt = any(
            self._action_is_passable(cur_pos, candidate) and self._action_hits_dirt(candidate)
            for candidate in self._legal_actions(legal_action)
        )

        npc_pressure = cur_min_npc is not None and cur_min_npc <= avoid_dist
        low_efficiency_pressure = stall_steps >= stall_threshold
        repeat_pressure = self._repeat_area_pressure(legal_action)
        post_charge_pressure = self._post_charge_clean_pressure()
        post_charge_escape_pressure = self._post_charge_escape_pressure()
        clean_repeat_pressure = self._clean_repeat_pressure()
        clean_escape_pressure = self._clean_escape_pressure()
        narrow_pressure = self._narrow_area_pressure() and not has_fresh_adjacent
        open_area_pressure = (
            npc_pressure
            or low_efficiency_pressure
            or narrow_pressure
            or repeat_pressure
            or post_charge_pressure
            or post_charge_escape_pressure
            or clean_repeat_pressure
            or clean_escape_pressure
        )
        open_area_min_neighbors = int(self._cfg("OPEN_AREA_MIN_NEIGHBORS", 5))
        known_dirt_route_action = None
        frontier_route_action = None
        open_frontier_route_action = None
        allow_route_away_from_dirt = (
            not visible_targets
            or npc_pressure
            or (
                (narrow_pressure or repeat_pressure or post_charge_pressure or clean_repeat_pressure or clean_escape_pressure)
                and not has_adjacent_dirt
            )
        )
        if not self.force_charge_mode and allow_route_away_from_dirt:
            known_dirt_route_action = self._known_dirt_bfs_action(legal_action)
            if (
                known_dirt_route_action is None
                or stall_steps >= stall_threshold
                or npc_pressure
                or narrow_pressure
                or repeat_pressure
                or clean_repeat_pressure
                or clean_escape_pressure
                or post_charge_pressure
                or post_charge_escape_pressure
            ):
                frontier_route_action = self._frontier_bfs_action(legal_action)
            if open_area_pressure:
                open_frontier_route_action = self._open_frontier_bfs_action(legal_action)

        for action in self._legal_actions(legal_action):
            nxt = self._next_pos(cur_pos, action)
            if not self._action_is_passable(cur_pos, action):
                adjusted[action] = 0.0
                continue

            npc_dist = self._nearest_npc_dist_after(action)
            if npc_dist is not None:
                projected_dist = max(0, int(npc_dist) - 1)
                emergency_dist = self._cfg("NPC_EMERGENCY_PROJECTED_DIST", max(1, danger_dist - 1))
                if projected_dist <= emergency_dist:
                    adjusted[action] *= self._cfg("NPC_EMERGENCY_PROB_MULT", 0.02)
                elif projected_dist <= danger_dist:
                    adjusted[action] *= self._cfg("NPC_DANGER_PROB_MULT", 0.10)
                elif cur_min_npc is not None and cur_min_npc <= avoid_dist:
                    if npc_dist < cur_min_npc:
                        adjusted[action] *= self._cfg("NPC_CLOSER_PROB_MULT", 0.20)
                    elif npc_dist == cur_min_npc:
                        adjusted[action] *= self._cfg("NPC_STAY_NEAR_PROB_MULT", 0.60)
                    else:
                        adjusted[action] *= self._cfg("NPC_AWAY_PROB_MULT", 1.30)

            projected_npc_heat = self._projected_npc_heat(nxt)
            if projected_npc_heat >= 0.20:
                if not (self.force_charge_mode and self._charge_near_emergency_pressure(self.current_charger_path_dist)):
                    adjusted[action] *= max(
                        0.15,
                        float(self._cfg("NPC_PROJECT_PROB_MULT", 0.60)) ** max(1.0, projected_npc_heat * 3.0),
                    )

            cell = self._local_cell(nxt)
            if cell == 2:
                adjusted[action] *= self._cfg("DIRT_CELL_PROB_MULT", 1.45)

            battery = getattr(self.preprocessor, "battery", None)
            battery_max = getattr(self.preprocessor, "battery_max", None)
            battery_ratio = 0.0
            if battery is not None and battery_max is not None and float(battery_max) > 0:
                battery_ratio = float(battery) / float(battery_max)
            if (
                self._flag("AVOID_CHARGER_ZONE_WHEN_NOT_CHARGING", True)
                and not self.force_charge_mode
                and battery_ratio >= float(self._cfg("AVOID_CHARGER_ZONE_BATTERY_RATIO", 0.55))
                and cell != 2
            ):
                near_dist = int(self._cfg("CHARGER_NEAR_DIST", 1))
                cur_zone_dist = self._charger_zone_dist(cur_pos)
                next_zone_dist = self._charger_zone_dist(nxt)
                if cur_zone_dist is not None and int(cur_zone_dist) <= near_dist:
                    if next_zone_dist is not None and int(next_zone_dist) > int(cur_zone_dist):
                        adjusted[action] *= float(self._cfg("CHARGER_ZONE_LEAVE_PROB_MULT", 2.40))
                    elif next_zone_dist is not None and int(next_zone_dist) <= near_dist:
                        adjusted[action] *= float(self._cfg("CHARGER_ZONE_AVOID_PROB_MULT", 0.08))
                elif next_zone_dist is not None and int(next_zone_dist) <= near_dist:
                    adjusted[action] *= float(self._cfg("CHARGER_ZONE_AVOID_PROB_MULT", 0.08))

            if (
                self._flag("ENABLE_CHARGE_ROUTE_ESCAPE_SOFT_BIAS", False)
                and not self.force_charge_mode
                and self._charge_route_escape_soft_blend() > 1e-6
                and cell != 2
            ):
                if self._post_charge_route_revisit(nxt):
                    adjusted[action] *= float(self._cfg("POST_CHARGE_ROUTE_REVISIT_MULT", 0.20))
                else:
                    visit_n_for_route = int(visit_count.get(nxt, 0))
                    if visit_n_for_route <= 0:
                        adjusted[action] *= float(self._cfg("POST_CHARGE_ESCAPE_FRESH_MULT", 1.55))
                    if self._is_frontier_cell(nxt):
                        adjusted[action] *= float(self._cfg("POST_CHARGE_ESCAPE_FRONTIER_MULT", 1.55))
                    if nxt in history[-recent_window:]:
                        adjusted[action] *= 0.22

            if cur_dirt_dist is not None:
                next_dirt_dist = min(self._manhattan(nxt, target) for target in visible_targets)
                if next_dirt_dist < cur_dirt_dist:
                    adjusted[action] *= self._cfg("APPROACH_VISIBLE_DIRT_PROB_MULT", 1.20)
                elif next_dirt_dist > cur_dirt_dist and cur_dirt_dist <= 8:
                    adjusted[action] *= self._cfg("LEAVE_VISIBLE_DIRT_PROB_MULT", 0.92)

            visit_n = int(visit_count.get(nxt, 0))
            if visit_n <= 0:
                adjusted[action] *= self._cfg("EXPLORE_NEW_CELL_PROB_MULT", 1.45)
            elif visit_n == 1:
                adjusted[action] *= self._cfg("EXPLORE_LOW_VISIT_PROB_MULT", 1.18)
            if self._is_frontier_cell(nxt):
                adjusted[action] *= self._cfg("EXPLORE_FRONTIER_PROB_MULT", 1.25)
            if clean_repeat_pressure and cell != 2:
                if visit_n <= 0:
                    adjusted[action] *= self._cfg("CLEAN_REPEAT_FRESH_PROB_MULT", 1.0)
                if self._is_frontier_cell(nxt):
                    adjusted[action] *= self._cfg("CLEAN_REPEAT_FRONTIER_PROB_MULT", 1.0)
                if self._local_open_neighbor_count(nxt) >= open_area_min_neighbors:
                    adjusted[action] *= self._cfg("CLEAN_REPEAT_OPEN_PROB_MULT", 1.0)
            if clean_escape_pressure and cell != 2:
                if visit_n <= 0:
                    adjusted[action] *= self._cfg("CLEAN_ESCAPE_FRESH_PROB_MULT", 1.28)
                elif visit_n > 0:
                    adjusted[action] *= self._cfg("CLEAN_ESCAPE_REVISIT_PROB_MULT", 0.18)
                if self._is_frontier_cell(nxt):
                    adjusted[action] *= self._cfg("CLEAN_ESCAPE_FRONTIER_PROB_MULT", 1.22)
                if self._local_open_neighbor_count(nxt) >= open_area_min_neighbors:
                    adjusted[action] *= self._cfg("CLEAN_ESCAPE_OPEN_PROB_MULT", 1.14)

            suspect_ttl = self._npc_clean_suspect_ttl(nxt)
            if suspect_ttl > 0 and cell != 2 and not self.force_charge_mode:
                suspect_mult = float(self._cfg("NPC_CLEAN_PROB_MULT", 0.78))
                if clean_escape_pressure:
                    suspect_mult = min(
                        suspect_mult,
                        float(self._cfg("NPC_CLEAN_ESCAPE_PROB_MULT", 0.26)),
                    )
                elif repeat_pressure or clean_repeat_pressure:
                    suspect_mult = min(
                        suspect_mult,
                        float(self._cfg("NPC_CLEAN_REPEAT_PROB_MULT", 0.48)),
                    )
                if self._is_frontier_cell(nxt) and visit_n <= 0:
                    suspect_mult = max(
                        suspect_mult,
                        float(self._cfg("NPC_CLEAN_FRONTIER_PROB_FLOOR", 0.68)),
                    )
                elif visit_n <= 0:
                    suspect_mult = max(
                        suspect_mult,
                        float(self._cfg("NPC_CLEAN_FRESH_PROB_FLOOR", 0.58)),
                    )
                adjusted[action] *= suspect_mult

            if not self.force_charge_mode and open_area_pressure and cell != 2:
                open_neighbors = self._local_open_neighbor_count(nxt)
                if open_neighbors >= open_area_min_neighbors:
                    if npc_pressure:
                        adjusted[action] *= self._cfg("OPEN_AREA_NPC_PROB_MULT", 1.20)
                    if low_efficiency_pressure:
                        adjusted[action] *= self._cfg("OPEN_AREA_STALL_PROB_MULT", 1.12)
                elif open_neighbors <= 2 and not self._action_hits_dirt(action):
                    adjusted[action] *= self._cfg("NARROW_AREA_PRESSURE_PROB_MULT", 0.85)

            if (post_charge_pressure or clean_repeat_pressure) and cell != 2:
                if self._post_charge_route_revisit(nxt):
                    revisit_mult = self._cfg("POST_CHARGE_ROUTE_REVISIT_MULT", 0.45)
                    if clean_repeat_pressure:
                        revisit_mult = min(
                            float(revisit_mult),
                            float(self._cfg("CLEAN_REPEAT_ROUTE_REVISIT_MULT", 0.55)),
                        )
                    adjusted[action] *= revisit_mult
                cur_zone_dist = self._charger_zone_dist(cur_pos)
                next_zone_dist = self._charger_zone_dist(nxt)
                near_dist = int(self._cfg("POST_CHARGE_NEAR_CHARGER_DIST", 4))
                if (
                    post_charge_pressure
                    and cur_zone_dist is not None
                    and next_zone_dist is not None
                    and int(next_zone_dist) <= int(cur_zone_dist)
                    and int(next_zone_dist) <= near_dist
                ):
                    adjusted[action] *= self._cfg("POST_CHARGE_NEAR_CHARGER_MULT", 0.65)

            if post_charge_escape_pressure and cell != 2:
                if visit_n > 0:
                    adjusted[action] *= self._cfg("POST_CHARGE_ESCAPE_REVISIT_MULT", 0.32)
                if self._post_charge_route_revisit(nxt):
                    adjusted[action] *= min(
                        float(self._cfg("POST_CHARGE_ROUTE_REVISIT_MULT", 0.45)),
                        float(self._cfg("POST_CHARGE_ESCAPE_REVISIT_MULT", 0.32)),
                    )
                if visit_n <= 0:
                    adjusted[action] *= self._cfg("POST_CHARGE_ESCAPE_FRESH_MULT", 1.32)
                if self._is_frontier_cell(nxt):
                    adjusted[action] *= self._cfg("POST_CHARGE_ESCAPE_FRONTIER_MULT", 1.22)
                if self._local_open_neighbor_count(nxt) >= open_area_min_neighbors:
                    adjusted[action] *= self._cfg("POST_CHARGE_ESCAPE_OPEN_MULT", 1.14)

            if known_dirt_route_action is not None and int(action) == int(known_dirt_route_action):
                route_mult = self._cfg("EXPLORE_KNOWN_DIRT_ROUTE_PROB_MULT", 1.35)
                if post_charge_pressure:
                    route_mult *= self._cfg("POST_CHARGE_KNOWN_DIRT_ROUTE_MULT", 1.15)
                if clean_repeat_pressure:
                    route_mult *= self._cfg("EXPLORE_CLEAN_REPEAT_ROUTE_PROB_MULT", 1.25)
                if clean_repeat_pressure and has_fresh_adjacent and visit_n > 0:
                    route_mult = min(route_mult, self._cfg("EXPLORE_ROUTE_REVISIT_PROB_CAP", 1.0))
                if has_fresh_adjacent and visit_n >= repeat_threshold:
                    route_mult = min(route_mult, self._cfg("EXPLORE_ROUTE_REVISIT_PROB_CAP", 1.0))
                adjusted[action] *= route_mult

            if frontier_route_action is not None and int(action) == int(frontier_route_action):
                route_mult = self._cfg("EXPLORE_FRONTIER_ROUTE_PROB_MULT", 1.20)
                if stall_steps >= stall_threshold:
                    route_mult *= self._cfg("EXPLORE_STALL_ROUTE_PROB_MULT", 1.25)
                    if not has_adjacent_dirt:
                        route_mult *= self._cfg("STALL_BREAK_FRONTIER_ROUTE_PROB_MULT", 1.40)
                if repeat_pressure:
                    route_mult *= self._cfg("EXPLORE_REPEAT_ROUTE_PROB_MULT", 1.0)
                if clean_repeat_pressure:
                    route_mult *= self._cfg("EXPLORE_CLEAN_REPEAT_ROUTE_PROB_MULT", 1.25)
                if clean_repeat_pressure and has_fresh_adjacent and visit_n > 0:
                    route_mult = min(route_mult, self._cfg("EXPLORE_ROUTE_REVISIT_PROB_CAP", 1.0))
                if post_charge_pressure:
                    route_mult *= self._cfg("EXPLORE_POST_CHARGE_ROUTE_PROB_MULT", 1.0)
                if has_fresh_adjacent and visit_n >= repeat_threshold:
                    route_mult = min(route_mult, self._cfg("EXPLORE_ROUTE_REVISIT_PROB_CAP", 1.0))
                adjusted[action] *= route_mult

            if open_frontier_route_action is not None and int(action) == int(open_frontier_route_action):
                route_mult = self._cfg("EXPLORE_OPEN_FRONTIER_ROUTE_PROB_MULT", 1.16)
                if low_efficiency_pressure:
                    route_mult *= self._cfg("EXPLORE_OPEN_FRONTIER_STALL_MULT", 1.12)
                    if not has_adjacent_dirt:
                        route_mult *= self._cfg("STALL_BREAK_OPEN_FRONTIER_ROUTE_PROB_MULT", 1.55)
                if narrow_pressure:
                    route_mult *= self._cfg("EXPLORE_OPEN_FRONTIER_NARROW_MULT", 1.0)
                if repeat_pressure:
                    route_mult *= self._cfg("EXPLORE_OPEN_FRONTIER_REPEAT_MULT", 1.0)
                if clean_repeat_pressure:
                    route_mult *= self._cfg("EXPLORE_OPEN_FRONTIER_CLEAN_REPEAT_MULT", 1.15)
                if clean_repeat_pressure and has_fresh_adjacent and visit_n > 0:
                    route_mult = min(route_mult, 1.0)
                if post_charge_pressure:
                    route_mult *= self._cfg("EXPLORE_OPEN_FRONTIER_POST_CHARGE_MULT", 1.0)
                if post_charge_escape_pressure:
                    route_mult *= 1.10
                if npc_pressure:
                    route_mult *= self._cfg("NPC_OPEN_FRONTIER_ROUTE_PROB_MULT", 1.10)
                if has_fresh_adjacent and visit_n >= repeat_threshold:
                    route_mult = min(route_mult, 1.05)
                adjusted[action] *= route_mult

            if action < len(blocked_ttl) and blocked_ttl[action] > 0:
                adjusted[action] *= 0.05
            elif len(history) >= 2 and nxt == history[-2]:
                adjusted[action] *= self._cfg("IMMEDIATE_BACKTRACK_PROB_MULT", 0.35)
            elif nxt in history[-recent_window:]:
                adjusted[action] *= self._cfg("RECENT_REVISIT_PROB_MULT", 0.42)

            if visit_n >= repeat_threshold:
                repeat_decay = float(self._cfg("REPEAT_VISIT_DECAY", 0.58))
                min_repeat_mult = float(self._cfg("MIN_REPEAT_VISIT_PROB_MULT", 0.25))
                adjusted[action] *= max(min_repeat_mult, repeat_decay ** float(visit_n - repeat_threshold + 1))

            if not self.force_charge_mode and cell != 2:
                if has_fresh_adjacent and visit_n > 0:
                    adjusted[action] *= self._cfg("CLEAN_FRESH_REVISIT_PROB_MULT", 0.10)
                elif (
                    low_efficiency_pressure
                    and not has_adjacent_dirt
                    and visit_n > 0
                    and not self._is_frontier_cell(nxt)
                ):
                    adjusted[action] *= self._cfg("STALL_BREAK_REVISIT_PROB_MULT", 0.18)
                elif (repeat_pressure or clean_repeat_pressure or clean_escape_pressure) and visit_n > 0:
                    revisit_mult = self._cfg("CLEAN_REPEAT_PRESSURE_REVISIT_MULT", 0.55)
                    if clean_repeat_pressure:
                        revisit_mult = min(
                            float(revisit_mult),
                            float(self._cfg("CLEAN_REPEAT_ACCUM_REVISIT_MULT", 0.38)),
                        )
                    if clean_escape_pressure:
                        revisit_mult = min(float(revisit_mult), float(self._cfg("CLEAN_ESCAPE_REVISIT_PROB_MULT", 0.18)))
                    adjusted[action] *= revisit_mult
                elif stall_steps >= stall_threshold and visit_n >= repeat_threshold:
                    adjusted[action] *= self._cfg("CLEAN_STALL_REVISIT_PROB_MULT", 0.18)

        adjusted *= legal_arr
        return self._normalize_prob(adjusted, legal_action)

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData with legacy base feature plus memory augmentation.

        将原始 env_obs 转换为 ObsData（73D 特征 + 合法动作掩码）。
        """
        self.preprocessor.pb2struct(env_obs, self.last_action)
        self._update_force_charge_mode()
        self._update_charge_route_memory()
        feature, legal_action = self.preprocessor.build_base_feature()
        reward = self.preprocessor.reward_process(**self._phase_reward_kwargs())
        self.last_reward = reward
        feature = self._augment_observation_feature(feature)

        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {}
        return obs_data, remain_info

    def action_process(self, act_data, is_stochastic=True):
        """Extract int action from ActData and update last_action.

        从 ActData 中取出动作整数并更新 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def _adaptive_guided_prob_blend(self):
        """根据清扫重复压力自适应调节 soft guidance 强度。

        这是概率层软偏置，不是硬规则接管：
        - force_charge 阶段不放大，避免破坏回充；
        - 清扫阶段只有在 steps_since_clean、CleanRepeat 或 recent loop 升高时才放大；
        - 最大值受配置限制，避免再次把 PPO 的 KL / clip_frac 拉爆。
        """
        base_blend = float(np.clip(self._cfg("GUIDED_PROB_BLEND", 0.0), 0.0, 1.0))
        if not self._flag("ENABLE_ADAPTIVE_GUIDED_PROB", False):
            return base_blend
        if self.force_charge_mode:
            return 0.0

        stall_steps = int(getattr(self.preprocessor, "steps_since_clean", 0))
        clean_repeat = float(self._clean_phase_repeat_ratio())
        overall_repeat = float(self._overall_repeat_ratio())
        recent_loop = bool(self._is_recent_loop())

        pressure_blend = float(self._cfg("GUIDED_PROB_BLEND_PRESSURE", 0.10))
        hard_blend = float(self._cfg("GUIDED_PROB_BLEND_HARD_PRESSURE", 0.16))
        stall_steps_soft = int(self._cfg("GUIDED_PROB_STALL_STEPS", 20))
        stall_steps_hard = int(self._cfg("GUIDED_PROB_HARD_STALL_STEPS", 42))
        repeat_soft = float(self._cfg("GUIDED_PROB_CLEAN_REPEAT_RATIO", 0.16))
        repeat_hard = float(self._cfg("GUIDED_PROB_CLEAN_REPEAT_HARD_RATIO", 0.26))
        overall_soft = float(self._cfg("GUIDED_PROB_OVERALL_REPEAT_RATIO", 0.24))

        if recent_loop or stall_steps >= stall_steps_hard or clean_repeat >= repeat_hard:
            return float(np.clip(max(base_blend, hard_blend), 0.0, 0.35))
        if stall_steps >= stall_steps_soft or clean_repeat >= repeat_soft or overall_repeat >= overall_soft:
            return float(np.clip(max(base_blend, pressure_blend), 0.0, 0.25))
        return base_blend

    def _progress_score_soft_blend(self):
        """High CleanRepeat 时把一部分采样概率分给“局部进度评分分布”。

        这不是 hard override：它不直接改动作，只是在采样阶段提高新格、frontier、
        污渍方向、远离充电区方向的概率。old_prob 仍可保持 PPO 原分布，避免 v7 那种
        guided old_prob 造成 clip_frac 过高。
        """
        if not self._flag("ENABLE_PROGRESS_SCORE_SOFT_BIAS", False):
            return 0.0
        if self.force_charge_mode:
            return 0.0

        step_no = int(getattr(self.preprocessor, "step_no", 0))
        min_step = int(self._cfg("PROGRESS_SCORE_SOFT_BIAS_MIN_STEP", 60))
        stall_steps = int(getattr(self.preprocessor, "steps_since_clean", 0))
        clean_repeat = float(self._clean_phase_repeat_ratio())
        overall_repeat = float(self._overall_repeat_ratio())
        recent_loop = bool(self._is_recent_loop())

        repeat_soft = float(self._cfg("GUIDED_PROB_CLEAN_REPEAT_RATIO", 0.20))
        repeat_hard = float(self._cfg("GUIDED_PROB_CLEAN_REPEAT_HARD_RATIO", 0.42))
        overall_soft = float(self._cfg("GUIDED_PROB_OVERALL_REPEAT_RATIO", 0.34))
        stall_soft = int(self._cfg("GUIDED_PROB_STALL_STEPS", 8))
        stall_hard = int(self._cfg("GUIDED_PROB_HARD_STALL_STEPS", 20))

        # 高电量还停在充电区时，优先给“离开充电区”的软概率，不增加 leave_charger 接管。
        battery = getattr(self.preprocessor, "battery", None)
        battery_max = getattr(self.preprocessor, "battery_max", None)
        battery_ratio = 0.0
        if battery is not None and battery_max is not None and float(battery_max) > 0:
            battery_ratio = float(battery) / float(battery_max)
        cur_zone_dist_for_avoid = self._charger_zone_dist(getattr(self.preprocessor, "cur_pos", None))
        high_battery_avoid_dist = int(self._cfg("HIGH_BATTERY_CHARGER_AVOID_DIST", self._cfg("CHARGER_NEAR_DIST", 1)))
        high_battery_charger_loop = bool(
            self._flag("AVOID_CHARGER_ZONE_WHEN_NOT_CHARGING", True)
            and cur_zone_dist_for_avoid is not None
            and int(cur_zone_dist_for_avoid) <= high_battery_avoid_dist
            and battery_ratio >= float(self._cfg("AVOID_CHARGER_ZONE_BATTERY_RATIO", 0.55))
        )

        if not high_battery_charger_loop and step_no < min_step and clean_repeat < repeat_hard:
            return 0.0

        base_blend = float(self._cfg("PROGRESS_SCORE_SOFT_BIAS_BLEND", 0.22))
        hard_blend = float(self._cfg("PROGRESS_SCORE_SOFT_BIAS_HARD_BLEND", 0.34))

        route_escape_blend = self._charge_route_escape_soft_blend()
        clean_target_blend = self._clean_target_route_soft_blend()
        aux_blend = max(route_escape_blend, clean_target_blend)

        if high_battery_charger_loop:
            return float(np.clip(max(base_blend, aux_blend, 0.30), 0.0, 0.48))
        if recent_loop or stall_steps >= stall_hard or clean_repeat >= repeat_hard:
            return float(np.clip(max(hard_blend, aux_blend), 0.0, 0.50))
        if stall_steps >= stall_soft or clean_repeat >= repeat_soft or overall_repeat >= overall_soft:
            return float(np.clip(max(base_blend, aux_blend), 0.0, 0.42))
        if aux_blend > 1e-6:
            return float(np.clip(aux_blend, 0.0, 0.36))
        return 0.0

    def _charge_route_escape_soft_blend(self):
        """Soft probability pressure for escaping repeated charger routes after charging.

        v10 观察到 rule 接管已经下降，但 charge_route_revisit 仍然在 0.5+。
        这里不硬改动作，只在采样概率里更偏向新格/frontier，降低回到刚走过的回充路线。
        """
        if not self._flag("ENABLE_CHARGE_ROUTE_ESCAPE_SOFT_BIAS", False):
            return 0.0
        # v15：simple charge 下如果长 route memory 没开，不使用 route-escape soft bias。
        # 否则很短/空的 recent_charge_route_cells 会让“必要通道”被误判，重复率反而升高。
        if self._flag("ENABLE_SIMPLE_CHARGE_MODE", False) and not self._flag("ENABLE_SIMPLE_CHARGE_ROUTE_MEMORY", False):
            return 0.0
        if self.force_charge_mode:
            return 0.0

        simple_mode = self._flag("ENABLE_SIMPLE_CHARGE_MODE", False)
        post_charge_pressure = bool(self._post_charge_clean_pressure() or self._post_charge_escape_pressure())
        if simple_mode and not post_charge_pressure:
            return 0.0

        route_steps = len(getattr(self, "recent_charge_route_cells", []))
        min_steps = int(self._cfg("CHARGE_ROUTE_ESCAPE_MIN_ROUTE_STEPS", 40))
        if route_steps < max(1, min_steps):
            return 0.0

        revisit_ratio = float(self._recent_route_revisit_ratio())
        threshold = float(self._cfg("CHARGE_ROUTE_ESCAPE_REVISIT_RATIO", 0.34))
        clean_repeat = float(self._clean_phase_repeat_ratio())
        clean_repeat_soft = float(self._cfg("GUIDED_PROB_CLEAN_REPEAT_RATIO", 0.16))
        clean_repeat_hard = float(self._cfg("GUIDED_PROB_CLEAN_REPEAT_HARD_RATIO", 0.34))

        if not post_charge_pressure and revisit_ratio < threshold and clean_repeat < clean_repeat_soft:
            return 0.0

        base_blend = float(self._cfg("CHARGE_ROUTE_ESCAPE_BLEND", 0.18))
        hard_blend = float(self._cfg("CHARGE_ROUTE_ESCAPE_HARD_BLEND", 0.30))
        if self._post_charge_escape_pressure() or revisit_ratio >= threshold + 0.18 or clean_repeat >= clean_repeat_hard:
            return float(np.clip(hard_blend, 0.0, 0.40))
        return float(np.clip(base_blend, 0.0, 0.32))

    def _clean_target_route_soft_blend(self):
        """Soft blend for following known dirt/frontier routes when repeat is high.

        v10 能把规则接管压低，但 clean_repeat 仍在 0.4 左右，且日志里 known_dirt
        经常数百到上千。问题不是没有目标，而是局部反重复太强时，模型不愿意
        穿过已访问走廊去远处污渍/frontier。这里仅给 BFS 第一动作一部分采样概率，
        不做 hard override。
        """
        if not self._flag("ENABLE_CLEAN_TARGET_ROUTE_SOFT_BIAS", False):
            return 0.0
        if self.force_charge_mode:
            return 0.0

        known_dirt_n = len(getattr(self.preprocessor, "known_dirt_positions", set()) or [])
        clean_repeat = float(self._clean_phase_repeat_ratio())
        overall_repeat = float(self._overall_repeat_ratio())
        stall_steps = int(getattr(self.preprocessor, "steps_since_clean", 0))
        min_known = int(self._cfg("CLEAN_TARGET_ROUTE_MIN_KNOWN_DIRT", 40))
        repeat_threshold = float(self._cfg("CLEAN_TARGET_ROUTE_REPEAT_RATIO", 0.26))
        stall_threshold = int(self._cfg("CLEAN_TARGET_ROUTE_STALL_STEPS", 8))

        battery = getattr(self.preprocessor, "battery", None)
        battery_max = getattr(self.preprocessor, "battery_max", None)
        battery_ratio = 0.0
        if battery is not None and battery_max is not None and float(battery_max) > 0:
            battery_ratio = float(battery) / float(battery_max)
        high_battery_target_pressure = bool(
            battery_ratio >= float(self._cfg("CLEAN_TARGET_ROUTE_HIGH_BATTERY_RATIO", 0.55))
            and known_dirt_n >= int(self._cfg("CLEAN_TARGET_ROUTE_HIGH_KNOWN_DIRT", 600))
            and stall_steps >= int(self._cfg("CLEAN_TARGET_ROUTE_HIGH_BATTERY_STALL", 3))
        )

        target_pressure = bool(
            known_dirt_n >= min_known
            and (
                clean_repeat >= repeat_threshold
                or overall_repeat >= repeat_threshold + 0.08
                or stall_steps >= stall_threshold
                or high_battery_target_pressure
            )
        )
        if not target_pressure:
            return 0.0

        base = float(self._cfg("CLEAN_TARGET_ROUTE_SOFT_BLEND", 0.18))
        hard = float(self._cfg("CLEAN_TARGET_ROUTE_SOFT_HARD_BLEND", 0.30))
        if high_battery_target_pressure:
            base += float(self._cfg("CLEAN_TARGET_ROUTE_HIGH_BATTERY_BLEND_BONUS", 0.08))
        if clean_repeat >= repeat_threshold + 0.16 or stall_steps >= max(stall_threshold + 18, 28):
            return float(np.clip(max(hard, base), 0.0, 0.44))
        return float(np.clip(base, 0.0, 0.38))

    def _progress_score_soft_prob(self, legal_action):
        """Build a local progress-oriented probability distribution over legal actions."""
        cur_pos = getattr(self.preprocessor, "cur_pos", None)
        if cur_pos is None or self.force_charge_mode:
            return None

        legal_actions = self._safe_legal_actions(legal_action)
        if not legal_actions:
            return None

        temp = max(1e-3, float(self._cfg("PROGRESS_SCORE_SOFT_BIAS_TEMP", 5.0)))
        visit_count = getattr(self.preprocessor, "visit_count", {})
        history = list(getattr(self.preprocessor, "pos_history", []))
        recent_window = int(self._cfg("RECENT_REVISIT_WINDOW", 10))
        recent_set = set(history[-max(1, recent_window):])
        open_area_min_neighbors = int(self._cfg("OPEN_AREA_MIN_NEIGHBORS", 5))
        clean_repeat = float(self._clean_phase_repeat_ratio())
        clean_repeat_pressure = bool(self._clean_repeat_pressure() or clean_repeat >= float(self._cfg("GUIDED_PROB_CLEAN_REPEAT_RATIO", 0.20)))

        battery = getattr(self.preprocessor, "battery", None)
        battery_max = getattr(self.preprocessor, "battery_max", None)
        battery_ratio = 0.0
        if battery is not None and battery_max is not None and float(battery_max) > 0:
            battery_ratio = float(battery) / float(battery_max)
        avoid_charger_zone = bool(
            self._flag("AVOID_CHARGER_ZONE_WHEN_NOT_CHARGING", True)
            and battery_ratio >= float(self._cfg("AVOID_CHARGER_ZONE_BATTERY_RATIO", 0.55))
        )
        cur_zone_dist = self._charger_zone_dist(cur_pos)
        near_dist = int(self._cfg("CHARGER_NEAR_DIST", 1))
        high_battery_avoid_dist = int(self._cfg("HIGH_BATTERY_CHARGER_AVOID_DIST", near_dist))
        charge_route_escape_active = bool(self._charge_route_escape_soft_blend() > 1e-6)
        route_penalty = float(self._cfg("CHARGE_ROUTE_ESCAPE_ROUTE_PENALTY", 10.0))
        recent_penalty = float(self._cfg("CHARGE_ROUTE_ESCAPE_RECENT_PENALTY", 7.0))
        route_fresh_bonus = float(self._cfg("CHARGE_ROUTE_ESCAPE_FRESH_BONUS", 6.0))
        route_frontier_bonus = float(self._cfg("CHARGE_ROUTE_ESCAPE_FRONTIER_BONUS", 5.0))

        # v11：显式把“去污渍 / 去 frontier 的 BFS 第一动作”灌进 soft score。
        # 之前只靠 cur_target_dist - next_target_dist，远距离目标每步只有 0.55 左右，
        # 很容易被 visit/recent 惩罚盖掉，导致 known_dirt 很多却清不到。
        visible_dirt_route_action = self._visible_dirt_bfs_action(legal_action)
        known_dirt_route_action = self._known_dirt_bfs_action(legal_action)
        frontier_route_action = self._frontier_bfs_action(legal_action)
        open_frontier_route_action = self._open_frontier_bfs_action(legal_action)
        clean_target_route_active = bool(self._clean_target_route_soft_blend() > 1e-6)
        high_battery_known_dirt_push = bool(
            battery_ratio >= float(self._cfg("CLEAN_TARGET_ROUTE_HIGH_BATTERY_RATIO", 0.55))
            and len(getattr(self.preprocessor, "known_dirt_positions", set()) or []) >= int(self._cfg("CLEAN_TARGET_ROUTE_HIGH_KNOWN_DIRT", 600))
        )

        scores = np.full(Config.ACTION_NUM, -1e9, dtype=np.float64)
        for action in legal_actions:
            action = int(action)
            if not self._action_is_passable(cur_pos, action):
                continue

            nxt = self._next_pos(cur_pos, action)
            cell = self._local_cell(nxt)
            if cell == 0:
                continue

            visit_n = float(visit_count.get(nxt, 0))
            score = self._progress_action_score(action, legal_action=legal_action)

            is_target_route_action = False
            if visible_dirt_route_action is not None and action == int(visible_dirt_route_action):
                score += float(self._cfg("CLEAN_TARGET_VISIBLE_DIRT_BONUS", 16.0))
                is_target_route_action = True
            if known_dirt_route_action is not None and action == int(known_dirt_route_action):
                score += float(self._cfg("CLEAN_TARGET_KNOWN_DIRT_BONUS", 13.0))
                if high_battery_known_dirt_push:
                    score += float(self._cfg("CLEAN_TARGET_HIGH_BATTERY_KNOWN_DIRT_EXTRA_BONUS", 4.0))
                is_target_route_action = True
            if frontier_route_action is not None and action == int(frontier_route_action):
                score += float(self._cfg("CLEAN_TARGET_FRONTIER_BONUS", 8.0))
                if high_battery_known_dirt_push:
                    score += float(self._cfg("CLEAN_TARGET_HIGH_BATTERY_FRONTIER_EXTRA_BONUS", 2.0))
                is_target_route_action = True
            if open_frontier_route_action is not None and action == int(open_frontier_route_action):
                score += float(self._cfg("CLEAN_TARGET_OPEN_FRONTIER_BONUS", 6.0))
                if high_battery_known_dirt_push:
                    score += float(self._cfg("CLEAN_TARGET_HIGH_BATTERY_FRONTIER_EXTRA_BONUS", 2.0))
                is_target_route_action = True

            # 如果目标路线第一步必须穿过已访问走廊，不能像普通重复那样罚死；
            # 否则机器人会在局部新格附近乱绕，永远到不了远处 known_dirt/frontier。
            route_revisit_relief = 1.0
            if clean_target_route_active and is_target_route_action:
                route_revisit_relief = float(self._cfg("CLEAN_TARGET_ROUTE_REVISIT_RELIEF", 0.45))

            # v9：满步低分局的共性是 CleanRepeat 0.6~0.8。这里直接把“新格/边界/开阔”
            # 变成采样偏好，而不是等 long stall 后再 hard override。
            if cell == 2:
                score += 9.0
            if visit_n <= 0:
                score += 5.5 if clean_repeat_pressure else 3.0
            elif visit_n <= 1:
                score += 1.2
            else:
                score -= route_revisit_relief * min(10.0, 2.2 * visit_n)

            if self._is_frontier_cell(nxt):
                score += 4.2 if clean_repeat_pressure else 2.0
            if self._local_open_neighbor_count(nxt) >= open_area_min_neighbors:
                score += 1.0

            if nxt in recent_set:
                score -= route_revisit_relief * (8.0 if clean_repeat_pressure else 4.0)
                if charge_route_escape_active:
                    score -= route_revisit_relief * recent_penalty
            if self._post_charge_route_revisit(nxt):
                score -= route_revisit_relief * (route_penalty if charge_route_escape_active else 4.0)
            elif charge_route_escape_active:
                # 刚结束回充路线后，优先走没有访问过的新格/frontier，避免沿回充旧路折返。
                if visit_n <= 0:
                    score += route_fresh_bonus
                if self._is_frontier_cell(nxt):
                    score += route_frontier_bonus

            # 高电量不应在充电区附近反复横跳。若已经在充电区，奖励离开；若不在，惩罚进入。
            if avoid_charger_zone:
                next_zone_dist = self._charger_zone_dist(nxt)
                # v20：高电量/刚充完时，不只离开 near-zone(<=1)，也要软性离开充电桩周边旧区域。
                # 这不硬改动作，只提高“离开充电区、去 known_dirt/frontier”的采样概率，减少满步剩电过多。
                if cur_zone_dist is not None and int(cur_zone_dist) <= high_battery_avoid_dist:
                    if next_zone_dist is not None and int(next_zone_dist) > int(cur_zone_dist):
                        score += 8.0 + 1.5 * float(max(0, high_battery_avoid_dist - int(cur_zone_dist)))
                    elif next_zone_dist is not None and int(next_zone_dist) <= near_dist and cell != 2:
                        score -= 11.5
                    elif next_zone_dist is not None and int(next_zone_dist) <= high_battery_avoid_dist and cell != 2:
                        score -= 3.5
                elif next_zone_dist is not None and int(next_zone_dist) <= near_dist and cell != 2:
                    score -= 9.0

            npc_dist = self._nearest_npc_dist_after(action)
            if npc_dist is not None and int(npc_dist) <= int(self._cfg("NPC_DANGER_DIST", 2)):
                score -= 30.0

            scores[action] = score

        legal_mask = np.asarray(legal_action, dtype=np.float64).reshape(-1) > 0
        if not np.isfinite(scores[legal_mask]).any():
            return None

        valid_scores = scores.copy()
        max_score = np.max(valid_scores[legal_mask])
        weights = np.zeros_like(valid_scores, dtype=np.float64)
        weights[legal_mask] = np.exp(np.clip((valid_scores[legal_mask] - max_score) / temp, -20.0, 2.0))
        return self._normalize_prob(weights, legal_action)

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时推理（随机采样动作）。
        """
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        logits, value = self._run_model(feature)

        legal_arr = np.array(legal_action, dtype=np.float32)
        raw_prob = self._legal_soft_max(logits, legal_arr)
        self._update_force_charge_mode()
        self._update_charge_route_memory()
        ppo_prob = self._normalize_prob(raw_prob, legal_action)
        guide_blend = self._adaptive_guided_prob_blend()
        if guide_blend > 1e-6:
            guided_prob = self._anti_repeat_prob_adjust(raw_prob, legal_action)
            sample_prob = self._normalize_prob(
                (1.0 - guide_blend) * np.asarray(ppo_prob, dtype=np.float64)
                + guide_blend * np.asarray(guided_prob, dtype=np.float64),
                legal_action,
            )
        else:
            sample_prob = ppo_prob

        score_blend = self._progress_score_soft_blend()
        if score_blend > 1e-6:
            score_prob = self._progress_score_soft_prob(legal_action)
            if score_prob is not None:
                sample_prob = self._normalize_prob(
                    (1.0 - score_blend) * np.asarray(sample_prob, dtype=np.float64)
                    + score_blend * np.asarray(score_prob, dtype=np.float64),
                    legal_action,
                )

        action = self._legal_sample(sample_prob, use_max=False)
        d_action = self._legal_sample(sample_prob, use_max=True)
        raw_action = action
        self.last_rule_override = 0
        self.last_rule_override_reason = "none"
        action = self._rule_adjust_action(action, legal_action)
        action_rule_reason = getattr(self, "_pending_rule_override_reason", "none")

        # d_action 只用于评估/展示，不应该二次触发 progress cooldown 或改变 force_charge 状态。
        # 之前这里再次调用 _rule_adjust_action，会让“未执行的贪心动作”也消耗规则冷却。
        saved_pending_reason = getattr(self, "_pending_rule_override_reason", "none")
        saved_progress_step = int(getattr(self, "last_progress_override_step", -10**9))
        saved_force_charge = bool(getattr(self, "force_charge_mode", False))
        d_action = self._rule_adjust_action(d_action, legal_action)
        self._pending_rule_override_reason = saved_pending_reason
        self.last_progress_override_step = saved_progress_step
        self.force_charge_mode = saved_force_charge

        if action != raw_action:
            self.last_rule_override = 1
            self.last_rule_override_reason = action_rule_reason if action_rule_reason != "none" else "unknown"
        sample_prob = self._normalize_prob(sample_prob, legal_action)
        behavior_prob = self._behavior_prob_for_action(
            sample_prob,
            legal_action,
            action,
            hard_override=bool(self.last_rule_override),
        )
        old_prob = behavior_prob if self._flag("USE_GUIDED_BEHAVIOR_PROB", True) else ppo_prob

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(old_prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时推理（贪心）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        """Delegate to Algorithm for PPO update.

        委托给 Algorithm 执行训练。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        if path is None:
            path = "."
        os.makedirs(path, exist_ok=True)
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = self._find_model_file(path, id)
        if not os.path.exists(model_file_path):
            if self.logger:
                self.logger.warning(f"model file for id {id} not found, skip loading")
            return False

        loaded_state = torch.load(model_file_path, map_location=self.device)
        current_state = self.model.state_dict()
        compatible_state = {
            k: v
            for k, v in loaded_state.items()
            if k in current_state and current_state[k].shape == v.shape
        }
        current_state.update(compatible_state)
        self.model.load_state_dict(current_state)
        if self.logger:
            skipped = len(current_state) - len(compatible_state)
            self.logger.info(
                f"load model {model_file_path} successfully, "
                f"compatible_params:{len(compatible_state)}, skipped_params:{skipped}"
            )
        return True

    def _find_model_file(self, path=None, id="1"):
        """Find explicit model id, latest model, or newest bundled ckpt."""
        package_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.join(package_dir, "ckpt")
        search_dirs = []
        if path is not None:
            search_dirs.append(path)
        search_dirs.extend([".", ckpt_dir])

        candidates = []

        def add_candidate(file_path):
            if file_path and file_path not in candidates:
                candidates.append(file_path)

        def add_id(model_id):
            if not model_id:
                return
            model_id = str(model_id)
            if model_id.endswith(".pkl"):
                file_name = model_id
            elif model_id.startswith("model.ckpt-"):
                file_name = f"{model_id}.pkl"
            else:
                file_name = f"model.ckpt-{model_id}.pkl"
            for search_dir in search_dirs:
                add_candidate(os.path.join(search_dir, file_name))

        if id != "auto":
            add_id(id)
        if id in ("auto", "latest"):
            add_id("latest")

        id_list_path = os.path.join(ckpt_dir, "id_list")
        if id in ("auto", "latest") and os.path.exists(id_list_path):
            try:
                with open(id_list_path, "r", encoding="utf-8") as f:
                    lines = [x.strip() for x in f.readlines() if x.strip()]
                for line in reversed(lines):
                    if line == "all id list":
                        continue
                    add_id(line)
            except Exception:
                pass

        kaiwu_path = os.path.join(ckpt_dir, "kaiwu.json")
        if id in ("auto", "latest") and os.path.exists(kaiwu_path):
            try:
                with open(kaiwu_path, "r", encoding="utf-8") as f:
                    model_paths = json.load(f).get("model_file_path", [])
                for model_path in model_paths:
                    if os.path.isabs(model_path):
                        add_candidate(model_path)
                    else:
                        add_candidate(os.path.join(package_dir, model_path))
                        add_candidate(os.path.join(package_dir, os.path.basename(model_path)))
            except Exception:
                pass

        if id in ("auto", "latest") and os.path.isdir(ckpt_dir):
            model_names = [
                name
                for name in os.listdir(ckpt_dir)
                if name.startswith("model.ckpt-") and name.endswith(".pkl")
            ]
            model_names.sort(key=lambda name: os.path.getmtime(os.path.join(ckpt_dir, name)), reverse=True)
            for name in model_names:
                add_candidate(os.path.join(ckpt_dir, name))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0] if candidates else os.path.join(".", f"model.ckpt-{id}.pkl")

    def _run_model(self, feature):
        """Gradient-free forward pass, returns (logits_np, value_np).

        无梯度推理，返回 (logits_np, value_np)。
        """
        self.model.set_eval_mode()
        obs_tensor = (
            torch.tensor(np.array([feature], dtype=np.float32)).view(1, Config.DIM_OF_OBSERVATION).to(self.device)
        )
        with torch.no_grad():
            rst = self.model(obs_tensor, inference=True)
        logits = rst[0].cpu().numpy()[0]
        value = rst[1].cpu().numpy()[0]
        return logits, value

    def _legal_soft_max(self, logits, legal_action):
        """Softmax with legal action masking.

        合法动作掩码下的 softmax。
        """
        _w, _e = 1e20, 1e-5
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return self._normalize_prob(tmp, legal_action)

    def _normalize_prob(self, probs, legal_action=None):
        """Return a finite probability vector normalized in float64."""
        prob_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
        if prob_arr.size == 0:
            return prob_arr

        prob_arr = np.nan_to_num(prob_arr, nan=0.0, posinf=0.0, neginf=0.0)
        prob_arr = np.maximum(prob_arr, 0.0)

        legal_indices = None
        if legal_action is not None:
            legal_arr = np.asarray(legal_action, dtype=np.float64).reshape(-1)
            if legal_arr.size == prob_arr.size:
                legal_mask = legal_arr > 0
                legal_indices = np.flatnonzero(legal_mask)
                prob_arr *= legal_mask

        total = float(np.sum(prob_arr, dtype=np.float64))
        if total <= 1e-12:
            prob_arr.fill(0.0)
            if legal_indices is not None and legal_indices.size > 0:
                prob_arr[legal_indices] = 1.0 / float(legal_indices.size)
            else:
                prob_arr.fill(1.0 / float(prob_arr.size))
            return prob_arr

        return prob_arr / total

    def _behavior_prob_for_action(self, sample_prob, legal_action, action, hard_override=False):
        """Approximate the behavior policy that actually produced the executed action."""
        behavior_prob = self._normalize_prob(sample_prob, legal_action)
        if not hard_override or not self._flag("USE_GUIDED_BEHAVIOR_PROB", True):
            return behavior_prob

        action = int(action)
        if action < 0 or action >= behavior_prob.size:
            return behavior_prob

        legal_arr = np.asarray(legal_action, dtype=np.float64).reshape(-1)
        if legal_arr.size != behavior_prob.size or legal_arr[action] <= 0:
            return behavior_prob

        # v6：hard override 只是兜底，不再假设 50%+ 的行为概率，避免 PPO old_prob 过度扭曲。
        override_prob = float(np.clip(self._cfg("HARD_OVERRIDE_BEHAVIOR_PROB", 0.30), 0.05, 0.95))
        softened = behavior_prob * (1.0 - override_prob)
        softened[action] += override_prob
        return self._normalize_prob(softened, legal_action)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution (argmax if use_max=True).

        按概率分布采样动作（use_max=True 时取 argmax）。
        """
        probs = self._normalize_prob(probs)
        if probs.size == 0:
            return 0
        if use_max:
            return int(np.argmax(probs))

        cdf = np.cumsum(probs, dtype=np.float64)
        if float(cdf[-1]) <= 0.0:
            return int(np.argmax(probs))
        sample = np.random.random() * float(cdf[-1])
        return int(np.searchsorted(cdf, sample, side="right"))
