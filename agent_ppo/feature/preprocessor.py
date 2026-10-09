#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

from collections import deque

import numpy as np

from agent_ppo.conf.conf import Config


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


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    LOW_BATTERY_RATIO = 0.25
    CRITICAL_BATTERY_RATIO = 0.12
    NPC_DANGER_DIST = Config.NPC_DANGER_DIST
    NPC_AVOID_DIST = Config.NPC_AVOID_DIST
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.max_episode_steps = max(int(getattr(Config, "MAX_EPISODE_STEPS", 2000)), 1)
        self.battery = 600
        self.battery_max = 600
        self.last_battery = None
        self.battery_gain = False

        self.cur_pos = (0, 0)
        self.last_pos = None
        self.pos_history = deque(maxlen=24)
        self.visit_count = {}
        self.stuck_steps = 0
        self.blocked_action_ttl = [0] * 8
        self.blocked_cell_ttl = {}
        self.blocked_action_hits = 0
        self.is_charging = False
        self.in_charger_zone = False
        self.charge_phase_steps = 0
        self.charge_no_gain_steps = 0
        self.known_cells = {}
        self.known_dirt_positions = set()
        self.recent_npc_positions = deque(maxlen=max(int(getattr(Config, "NPC_CLEAN_RECENT_STEPS", 6)), 1))
        self.suspected_npc_cleaned_cells = {}
        self.suspected_npc_clean_events = 0
        self.dirty_disappear_transitions = 0
        self.self_dirty_clear_transitions = 0
        self.npc_dirty_clear_suspect_transitions = 0
        self.suspected_npc_clean_visible_count = 0
        self.map_values_seen = set()

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1
        self.steps_since_clean = 0

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8
        self.charger_list = []
        self.npc_list = []
        self.map_id = -1
        self.nearest_npc_velocity = (0.0, 0.0)
        self.nearest_npc_speed = 0.0
        self.last_min_charger_dist = None
        self.last_charge_zone_dist = None
        self.last_safe_charger_dist_for_reward = None
        self.reward_direct_charger_path_dist = None
        self.reward_safe_charger_dist = None
        # 低电 / 预回充 reward 专用距离历史，只用于判断“最近几步是否持续靠近充电桩”。
        # 不进入 observation，不影响动作选择。
        self.safe_charger_dist_history = deque(maxlen=16)
        # 诊断字段：便于后续 PHASE_DEBUG / GAMEOVER 打印 reward 贡献。
        self.last_pre_charge_reward = 0.0
        self.last_min_npc_dist = None

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

    def _extract_map_id(self, observation, env_info):
        for source in (env_info, observation):
            if not isinstance(source, dict):
                continue
            for key in ("map_id", "mapId", "map_idx", "map_index", "map_no", "mapNo", "current_map"):
                value = source.get(key)
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return -1

    def _valid_reward_charger_dist(self, dist):
        """低电 reward 专用距离清洗：只接受正常数字，99 视为未知。"""
        if dist is None:
            return None
        try:
            dist = int(float(dist))
        except (TypeError, ValueError):
            return None
        if dist < 0 or dist >= 99:
            return None
        return dist

    def _reward_known_passable(self, pos):
        cell = self.known_cells.get(pos)
        if cell in (1, 2):
            return True
        cell = self._local_cell(pos)
        return cell in (1, 2)

    def _reward_charger_targets(self):
        if not self.charger_list:
            return []

        near_dist = int(getattr(Config, "CHARGER_NEAR_DIST", 1))
        targets = []
        seen = set()
        for charger in self.charger_list:
            cx, cz = int(charger[0]), int(charger[1])
            for dx in range(-near_dist, near_dist + 1):
                for dz in range(-near_dist, near_dist + 1):
                    pos = (cx + dx, cz + dz)
                    if pos in seen:
                        continue
                    if self._chebyshev(pos, (cx, cz)) > near_dist:
                        continue
                    if not self._reward_known_passable(pos):
                        continue
                    seen.add(pos)
                    targets.append(pos)
        return targets or list(self.charger_list)

    def _reward_direct_charger_path_dist(self):
        if self.cur_pos is None or not self.known_cells:
            return None

        targets = set(self._reward_charger_targets())
        if not targets:
            return None
        if self.cur_pos in targets:
            return 0

        queue = deque([(self.cur_pos, 0)])
        visited = {self.cur_pos}
        nodes = 0
        max_nodes = int(getattr(Config, "CHARGE_ROUTE_BFS_LIMIT", 2600))
        while queue and nodes < max_nodes:
            pos, dist = queue.popleft()
            nodes += 1
            for action in range(int(getattr(Config, "ACTION_NUM", 8))):
                nxt = self._next_pos(pos, action)
                if nxt in visited:
                    continue
                if nxt in targets:
                    return dist + 1
                if not self._reward_known_passable(nxt):
                    continue
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def _safe_charger_dist_for_reward(self, charger_path_dist=None, charger_min_dist=None, workflow_path_dist=None):
        """
        Battery-risk reward distance only.

        Prefer valid path distance, then an optional workflow/direct path, then
        a reward-only BFS direct path. Fall back to nearest charger distance
        only when all path distances are invalid. This does not affect actions.
        """
        for dist in (charger_path_dist, workflow_path_dist):
            valid = self._valid_reward_charger_dist(dist)
            if valid is not None:
                return valid

        direct_dist = self._valid_reward_charger_dist(self._reward_direct_charger_path_dist())
        self.reward_direct_charger_path_dist = direct_dist
        if direct_dist is not None:
            return direct_dist

        return self._valid_reward_charger_dist(charger_min_dist)

    def _in_charger_zone(self, cur_pos=None):
        cur_pos = self.cur_pos if cur_pos is None else cur_pos
        charger_dist = self._min_chebyshev_dist(cur_pos, self.charger_list)
        return charger_dist is not None and charger_dist <= int(getattr(Config, "CHARGER_NEAR_DIST", 1))

    def _should_suspend_block_check(self, battery_gain):
        battery_ratio = float(self.battery) / float(max(self.battery_max, 1))
        return bool(
            battery_gain
            or (
                self._in_charger_zone(self.cur_pos)
                and battery_ratio < float(getattr(Config, "EXIT_CHARGE_BATTERY_RATIO", 0.30))
            )
        )

    def _next_pos(self, cur_pos, action):
        dx, dz = DIR_MAP.get(int(action), (0, 0))
        return (cur_pos[0] + dx, cur_pos[1] + dz)

    def _local_cell(self, pos):
        view_map = self._view_map
        if self.cur_pos is None or view_map is None:
            return None

        li = pos[1] - self.cur_pos[1] + self.VIEW_HALF
        lj = pos[0] - self.cur_pos[0] + self.VIEW_HALF
        if 0 <= li < view_map.shape[0] and 0 <= lj < view_map.shape[1]:
            return int(view_map[li, lj])
        return None

    def _action_is_local_passable(self, action):
        if not self._action_is_static_local_passable(action):
            return False

        dx, dz = DIR_MAP.get(int(action), (0, 0))
        nxt = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
        return self.blocked_cell_ttl.get(nxt, 0) <= 0

    def _action_is_static_local_passable(self, action):
        if self.cur_pos is None:
            return True

        dx, dz = DIR_MAP.get(int(action), (0, 0))
        nxt = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
        if self._local_cell(nxt) == 0:
            return False

        if getattr(Config, "AVOID_DIAGONAL_CORNER_CUT", True) and dx != 0 and dz != 0:
            side_x = (self.cur_pos[0] + dx, self.cur_pos[1])
            side_z = (self.cur_pos[0], self.cur_pos[1] + dz)
            if self._local_cell(side_x) == 0 or self._local_cell(side_z) == 0:
                return False

        return True

    def _extract_charger_list(self, frame_state):
        charger_list = []
        organs = frame_state.get("organs", [])
        if not isinstance(organs, list):
            return charger_list

        for obj in organs:
            try:
                if int(obj.get("sub_type", -1)) != 1:
                    continue
                pos = obj.get("pos", {})
                charger_list.append((int(pos.get("x", 0)), int(pos.get("z", 0))))
            except Exception:
                continue
        return charger_list

    def _extract_npc_list(self, frame_state):
        npc_list = []
        npcs = frame_state.get("npcs", [])
        if not isinstance(npcs, list):
            return npc_list

        for npc in npcs:
            try:
                pos = npc.get("pos", {})
                npc_list.append((int(pos.get("x", 0)), int(pos.get("z", 0))))
            except Exception:
                continue
        return npc_list

    def _estimate_nearest_npc_motion(self, cur_pos, npc_list):
        if cur_pos is None or not npc_list or not self.recent_npc_positions:
            return (0.0, 0.0), 0.0

        prev_positions = list(self.recent_npc_positions[-1])
        if not prev_positions:
            return (0.0, 0.0), 0.0

        nearest_now = min(npc_list, key=lambda p: self._manhattan(cur_pos, p))
        nearest_prev = min(prev_positions, key=lambda p: self._manhattan(p, nearest_now))
        vel_x = float(int(nearest_now[0]) - int(nearest_prev[0]))
        vel_z = float(int(nearest_now[1]) - int(nearest_prev[1]))
        speed = float(np.sqrt(vel_x * vel_x + vel_z * vel_z))
        return (vel_x, vel_z), speed

    def _visible_clean_patch_cells(self, center_ri, center_ci, radius=1):
        view = self._view_map
        if view is None:
            return [], 0

        hx, hz = self.cur_pos
        center = self.VIEW_HALF
        clean_cells = []
        dirt_count = 0
        for ri in range(max(0, center_ri - radius), min(view.shape[0], center_ri + radius + 1)):
            for ci in range(max(0, center_ci - radius), min(view.shape[1], center_ci + radius + 1)):
                cell = int(view[ri, ci])
                if cell == 1:
                    gx = hx + ci - center
                    gz = hz + ri - center
                    if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                        clean_cells.append((int(gx), int(gz)))
                elif cell == 2:
                    dirt_count += 1
        return clean_cells, dirt_count

    def _npc_recent_min_dist(self, pos):
        candidates = list(self.npc_list)
        for npc_positions in self.recent_npc_positions:
            candidates.extend(list(npc_positions))
        if not candidates:
            return None
        return min(self._manhattan(pos, npc_pos) for npc_pos in candidates)

    def _decay_suspected_npc_cleaned(self):
        self.suspected_npc_cleaned_cells = {
            pos: ttl - 1
            for pos, ttl in self.suspected_npc_cleaned_cells.items()
            if ttl > 1
        }

    def _update_known_map(self):
        view = self._view_map
        if view is None:
            return

        hx, hz = self.cur_pos
        center = self.VIEW_HALF
        clean_delta = int(self.dirt_cleaned) - int(self.last_dirt_cleaned)
        suspect_dist = int(getattr(Config, "NPC_CLEAN_SUSPECT_DIST", self.NPC_AVOID_DIST))
        suspect_ttl = max(int(getattr(Config, "NPC_CLEAN_MEMORY_STEPS", 220)), 1)
        visible_suspect_count = 0
        for ri in range(view.shape[0]):
            for ci in range(view.shape[1]):
                gx = hx + ci - center
                gz = hz + ri - center
                if not (0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE):
                    continue

                pos = (int(gx), int(gz))
                cell = int(view[ri, ci])
                self.map_values_seen.add(cell)
                prev_cell = self.known_cells.get(pos)
                if prev_cell == 2 and cell != 2:
                    self.dirty_disappear_transitions += 1
                    if clean_delta > 0:
                        self.self_dirty_clear_transitions += 1
                        self.suspected_npc_cleaned_cells.pop(pos, None)
                    else:
                        npc_recent_dist = self._npc_recent_min_dist(pos)
                        if npc_recent_dist is not None and int(npc_recent_dist) <= suspect_dist:
                            self.npc_dirty_clear_suspect_transitions += 1
                            self.suspected_npc_clean_events += 1
                            self.suspected_npc_cleaned_cells[pos] = max(
                                int(self.suspected_npc_cleaned_cells.get(pos, 0)),
                                suspect_ttl,
                            )
                elif prev_cell is None and cell == 1 and clean_delta <= 0:
                    npc_recent_dist = self._npc_recent_min_dist(pos)
                    if npc_recent_dist is not None and int(npc_recent_dist) <= suspect_dist:
                        patch_cells, patch_dirt = self._visible_clean_patch_cells(ri, ci, radius=1)
                        if len(patch_cells) >= 5 and len(patch_cells) >= patch_dirt + 2:
                            self.suspected_npc_clean_events += 1
                            for patch_pos in patch_cells:
                                self.suspected_npc_cleaned_cells[patch_pos] = max(
                                    int(self.suspected_npc_cleaned_cells.get(patch_pos, 0)),
                                    suspect_ttl,
                                )
                self.known_cells[pos] = cell
                if cell == 2:
                    self.known_dirt_positions.add(pos)
                    self.suspected_npc_cleaned_cells.pop(pos, None)
                else:
                    self.known_dirt_positions.discard(pos)
                    if self.suspected_npc_cleaned_cells.get(pos, 0) > 0:
                        visible_suspect_count += 1
        self.suspected_npc_clean_visible_count = visible_suspect_count

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        self.map_id = self._extract_map_id(observation, env_info)
        self.max_episode_steps = max(
            int(env_info.get("max_step", env_info.get("max_steps", self.max_episode_steps))),
            1,
        )
        hero = frame_state["heroes"]
        if isinstance(hero, list):
            hero = hero[0] if hero else {}

        self.step_no = int(observation["step_no"])
        hero_pos = hero.get("pos", {})
        cur_pos = (int(hero_pos.get("x", 0)), int(hero_pos.get("z", 0)))
        self.cur_pos = cur_pos
        self.charger_list = self._extract_charger_list(frame_state)
        self.npc_list = self._extract_npc_list(frame_state)
        self.nearest_npc_velocity, self.nearest_npc_speed = self._estimate_nearest_npc_motion(cur_pos, self.npc_list)
        self.recent_npc_positions.append(tuple(self.npc_list))
        self._decay_suspected_npc_cleaned()

        self.last_battery = self.battery
        self.battery = int(hero.get("battery", self.battery))
        self.battery_max = max(int(hero.get("battery_max", self.battery_max)), 1)
        battery_gain = self.last_battery is not None and self.battery > self.last_battery
        self.battery_gain = bool(battery_gain)
        self.in_charger_zone = self._in_charger_zone(cur_pos)
        self.is_charging = self._should_suspend_block_check(battery_gain)
        self.blocked_action_ttl = [max(0, ttl - 1) for ttl in self.blocked_action_ttl]
        self.blocked_cell_ttl = {
            pos: ttl - 1
            for pos, ttl in self.blocked_cell_ttl.items()
            if ttl > 1
        }

        if self.is_charging:
            self.stuck_steps = 0
            self.blocked_action_ttl = [0] * len(self.blocked_action_ttl)
            self.blocked_cell_ttl.clear()
            self.charge_phase_steps += 1
            self.charge_no_gain_steps = 0 if battery_gain else self.charge_no_gain_steps + 1
        elif self.last_pos is not None and cur_pos == self.last_pos:
            self.charge_phase_steps = 0
            self.charge_no_gain_steps = 0
            self.stuck_steps += 1
            if 0 <= int(last_action) < len(self.blocked_action_ttl):
                cooldown = Config.BLOCKED_ACTION_COOLDOWN
                if self.stuck_steps >= int(getattr(Config, "SEVERE_STUCK_ESCAPE_STEPS", 3)):
                    cooldown = max(cooldown, int(getattr(Config, "STUCK_BLOCKED_ACTION_COOLDOWN", cooldown)))
                self.blocked_action_ttl[int(last_action)] = max(
                    self.blocked_action_ttl[int(last_action)],
                    cooldown,
                )
                self.blocked_action_hits += 1
                blocked_pos = self._next_pos(self.last_pos, last_action)
                if 0 <= blocked_pos[0] < self.GRID_SIZE and 0 <= blocked_pos[1] < self.GRID_SIZE:
                    self.blocked_cell_ttl[blocked_pos] = max(
                        self.blocked_cell_ttl.get(blocked_pos, 0),
                        int(getattr(Config, "BLOCKED_CELL_COOLDOWN", 18)),
                    )
        else:
            self.charge_phase_steps = 0
            self.charge_no_gain_steps = 0
            self.stuck_steps = 0
            self.blocked_cell_ttl.pop(cur_pos, None)
        self.pos_history.append(cur_pos)
        self.visit_count[cur_pos] = self.visit_count.get(cur_pos, 0) + 1

        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero.get("dirt_cleaned", self.dirt_cleaned))
        if self.dirt_cleaned > self.last_dirt_cleaned:
            self.steps_since_clean = 0
        else:
            self.steps_since_clean += 1
        self.total_dirt = max(int(env_info.get("total_dirt", self.total_dirt)), 1)

        legal_src = observation.get("legal_action")
        if legal_src is None:
            legal_src = observation.get("legal_act")
        self._legal_act = [int(x) for x in (legal_src or [1] * 8)]

        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)
            self._update_known_map()

        self.last_pos = cur_pos

    def _update_passable(self, hx, hz):
        """Write local view into global passable map.

        将局部视野写入全局通行地图。
        """
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ci
                gz = hz - half + ri
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    # 0 = obstacle, 1/2 = passable
                    # 0 = 障碍, 1/2 = 可通行
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _get_local_view_feature(self):
        """Local view feature (49D): crop center 7×7 from 21×21.

        局部视野特征（49D）：从 21×21 视野中心裁剪 7×7。
        """
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        return (crop / 2.0).flatten()

    def _get_global_state_feature(self):
        """Global state feature (16D).

        全局状态特征（16D）。

        Dimensions / 维度说明：
          [0]  step_norm         step progress / 步数归一化 [0,1]
          [1]  battery_ratio     battery level / 电量比 [0,1]
          [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
          [3]  remaining_dirt    remaining dirt ratio / 剩余污渍比例 [0,1]
          [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
          [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
          [6]  ray_N_dirt        north ray distance / 向上（z-）方向最近污渍距离
          [7]  ray_E_dirt        east ray distance / 向右（x+）方向
          [8]  ray_S_dirt        south ray distance / 向下（z+）方向
          [9]  ray_W_dirt        west ray distance / 向左（x-）方向
          [10] nearest_dirt_norm nearest dirt Euclidean distance / 最近污渍欧氏距离归一化
          [11] dirt_delta        approaching dirt indicator / 是否在接近污渍（1=是, 0=否）
          [12] battery_low       low battery flag / 是否低电量
          [13] nearest_charger   nearest charger distance / 最近充电桩距离
          [14] nearest_npc       nearest official robot distance / 最近官方机器人距离
          [15] npc_danger        official robot danger flag / 官方机器人危险标记
        """
        step_norm = _norm(self.step_no, self.max_episode_steps)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 4-directional ray to find nearest dirt
        # 四方向射线找最近污渍距离
        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W
        ray_dirt = []
        max_ray = 30
        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self._view_map is not None:
                    cell = (
                        int(
                            self._view_map[
                                np.clip(z - (hz - self.VIEW_HALF), 0, 20), np.clip(x - (hx - self.VIEW_HALF), 0, 20)
                            ]
                        )
                        if (0 <= x - hx + self.VIEW_HALF < 21 and 0 <= z - hz + self.VIEW_HALF < 21)
                        else 0
                    )
                    if cell == 2:
                        found = step
                        break
            ray_dirt.append(_norm(found, max_ray))

        # Nearest dirt Euclidean distance (estimated from 7×7 crop)
        # 最近污渍欧氏距离（视野内 7×7 粗估）
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)

        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0
        battery_low = 1.0 if battery_ratio <= self.LOW_BATTERY_RATIO else 0.0

        nearest_charger = self._min_dist(self.cur_pos, self.charger_list)
        if nearest_charger is None:
            nearest_charger = 200.0
        nearest_charger_norm = _norm(nearest_charger, 40.0)

        nearest_npc = self._min_dist(self.cur_pos, self.npc_list)
        if nearest_npc is None:
            nearest_npc = 200.0
        nearest_npc_norm = _norm(nearest_npc, 40.0)
        npc_danger = 1.0 if nearest_npc <= self.NPC_DANGER_DIST else 0.0

        return np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                ray_dirt[0],
                ray_dirt[1],
                ray_dirt[2],
                ray_dirt[3],
                nearest_dirt_norm,
                dirt_delta,
                battery_low,
                nearest_charger_norm,
                nearest_npc_norm,
                npc_danger,
            ],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        legal_action = list(self._legal_act)
        if self.is_charging:
            passable_action = [
                int(ok) if self._action_is_local_passable(idx) else 0
                for idx, ok in enumerate(legal_action)
            ]
            return passable_action if any(passable_action) else legal_action

        masked_action = [
            int(ok) if self.blocked_action_ttl[idx] <= 0 and self._action_is_local_passable(idx) else 0
            for idx, ok in enumerate(legal_action)
        ]
        if any(masked_action):
            return masked_action

        if self.stuck_steps > 0:
            min_ttl = min(
                self.blocked_action_ttl[idx]
                for idx, ok in enumerate(legal_action)
                if ok > 0 and self._action_is_local_passable(idx)
            ) if any(
                ok > 0 and self._action_is_local_passable(idx)
                for idx, ok in enumerate(legal_action)
            ) else None
            if min_ttl is not None:
                least_blocked_action = [
                    int(ok)
                    if self._action_is_local_passable(idx) and self.blocked_action_ttl[idx] <= min_ttl
                    else 0
                    for idx, ok in enumerate(legal_action)
                ]
                if any(least_blocked_action):
                    return least_blocked_action

        passable_action = [
            int(ok) if self._action_is_local_passable(idx) else 0
            for idx, ok in enumerate(legal_action)
        ]
        if any(passable_action):
            return passable_action

        static_passable_action = [
            int(ok) if self._action_is_static_local_passable(idx) else 0
            for idx, ok in enumerate(legal_action)
        ]
        return static_passable_action if any(static_passable_action) else legal_action

    def build_base_feature(self):
        """Build the legacy 73D feature from current cached state."""
        local_view = self._get_local_view_feature()  # 49D
        global_state = self._get_global_state_feature()  # 16D
        legal_action = self.get_legal_action()  # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)
        feature = np.concatenate([local_view, global_state, legal_arr]).astype(np.float32)  # 73D
        return feature, legal_action

    def feature_process(self, env_obs, last_action):
        """Generate 73D feature vector, legal action mask, and scalar reward.

        生成 73D 特征向量、合法动作掩码和标量奖励。
        """
        self.pb2struct(env_obs, last_action)
        feature, legal_action = self.build_base_feature()
        reward = self.reward_process()
        return feature, legal_action, reward

    def reward_process(
        self,
        force_charge_mode=False,
        charge_route_revisit_ratio=0.0,
        clean_repeat_ratio=0.0,
        post_charge_pressure=False,
        post_charge_escape_pressure=False,
        post_charge_escape_ratio=0.0,
        charger_path_dist=None,
        last_charger_path_dist=None,
        force_charge_search_steps=0,
        charge_zone_no_gain_steps=0,
    ):
        # Cleaning reward / 清扫奖励
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        reward = Config.CLEAN_REWARD * cleaned_this_step + Config.STEP_PENALTY + Config.STEP_SURVIVAL_REWARD
        if cleaned_this_step > 0:
            reward += Config.CLEAN_STEP_BONUS

        if self.nearest_dirt_dist < self.last_nearest_dirt_dist:
            reward += Config.APPROACH_DIRT_REWARD
        elif self.last_nearest_dirt_dist < 200.0 and self.nearest_dirt_dist > self.last_nearest_dirt_dist:
            reward += Config.LEAVE_DIRT_PENALTY

        is_charging = self.last_battery is not None and self.battery > self.last_battery
        visit_n = self.visit_count.get(self.cur_pos, 0)
        if visit_n == 1:
            reward += Config.NEW_CELL_REWARD
        if visit_n > 2 and not is_charging:
            reward += Config.REVISIT_PENALTY * min(visit_n - 1, 3)
        recent_positions = list(self.pos_history)[-8:]
        recent_unique = len(set(recent_positions))
        if len(recent_positions) >= 8 and recent_unique <= 3 and not is_charging:
            reward += Config.LOOP_PENALTY
        stall_threshold = int(getattr(Config, "LOW_EFFICIENCY_STALL_STEPS", 35))
        if self.steps_since_clean >= stall_threshold and not is_charging:
            stall_level = min(4, 1 + (self.steps_since_clean - stall_threshold) // 20)
            reward += Config.NO_CLEAN_STALL_PENALTY * float(stall_level)

        battery_ratio = self.battery / max(self.battery_max, 1)

        # v9：满步低分局里出现“电量很高但在充电区附近反复停留”的情况。
        # 这不是生存问题，而是清扫效率问题；只在非 force_charge、高电量、没有清扫收益时惩罚，
        # 避免影响真正低电回充。
        if (
            bool(getattr(Config, "AVOID_CHARGER_ZONE_WHEN_NOT_CHARGING", True))
            and not bool(force_charge_mode)
            and self.in_charger_zone
            and not is_charging
            and cleaned_this_step <= 0
            and battery_ratio >= float(getattr(Config, "AVOID_CHARGER_ZONE_BATTERY_RATIO", 0.55))
        ):
            reward += float(getattr(Config, "HIGH_BATTERY_CHARGER_ZONE_PENALTY", -0.026))
            if int(charge_zone_no_gain_steps or 0) > 0 or self.battery >= self.battery_max:
                reward += float(getattr(Config, "HIGH_BATTERY_CHARGER_ZONE_NO_GAIN_PENALTY", -0.040))

        if battery_ratio < self.LOW_BATTERY_RATIO:
            reward += Config.LOW_BATTERY_PENALTY
        if battery_ratio < self.CRITICAL_BATTERY_RATIO:
            reward += Config.CRITICAL_BATTERY_PENALTY


        # ===== Step2-lite：清扫效率 shaping =====
        # 只在非回充、非充电、且电量不太低时生效，避免为了少重复而破坏生存。
        if (
            bool(getattr(Config, "ENABLE_CLEAN_EFFICIENCY_REWARD", False))
            and not bool(force_charge_mode)
            and not is_charging
            and battery_ratio >= float(getattr(Config, "CLEAN_EFF_MIN_BATTERY_RATIO", 0.30))
        ):
            local_cell = self._local_cell(self.cur_pos)
            clean_eff_reward = 0.0

            # 新格子给极小奖励，和原 NEW_CELL_REWARD 叠加很轻，只强化“向未覆盖区域推进”。
            if visit_n == 1 and cleaned_this_step <= 0:
                clean_eff_reward += float(getattr(Config, "CLEAN_EFF_NEW_CELL_REWARD", 0.003))

            # 最近窗口重复踩旧格子，且没有清扫收益时，轻罚。
            if visit_n > 1 and cleaned_this_step <= 0 and local_cell != 2:
                recent_window = max(4, int(getattr(Config, "CLEAN_EFF_RECENT_WINDOW", 10)))
                recent_hits = list(self.pos_history)[-recent_window:].count(self.cur_pos)
                if recent_hits >= 2:
                    clean_eff_reward += float(getattr(Config, "CLEAN_EFF_RECENT_REVISIT_PENALTY", -0.003)) * float(
                        min(3, recent_hits - 1)
                    )

            # 长时间没有清扫收益时，轻微推动离开旧区域。
            stall_start = int(getattr(Config, "CLEAN_EFF_STALL_START_STEPS", 12))
            long_stall = int(getattr(Config, "CLEAN_EFF_LONG_STALL_STEPS", 28))
            if cleaned_this_step <= 0 and self.steps_since_clean >= stall_start:
                clean_eff_reward += float(getattr(Config, "CLEAN_EFF_STALL_PENALTY", -0.004))
            if cleaned_this_step <= 0 and self.steps_since_clean >= long_stall:
                clean_eff_reward += float(getattr(Config, "CLEAN_EFF_LONG_STALL_PENALTY", -0.008))

            # 清扫阶段重复率已经偏高时，额外给轻惩罚，但不在刚开始几步就乱扣。
            repeat_start = float(getattr(Config, "CLEAN_EFF_REPEAT_RATIO_START", 0.16))
            if clean_repeat_ratio >= repeat_start and self.steps_since_clean >= stall_start and cleaned_this_step <= 0:
                repeat_scale = 1.0 + min(2.0, (float(clean_repeat_ratio) - repeat_start) / max(1e-6, 1.0 - repeat_start))
                clean_eff_reward += float(getattr(Config, "CLEAN_EFF_REPEAT_RATIO_PENALTY", -0.004)) * repeat_scale

            # v6：旧逻辑只在 steps_since_clean 较高时惩罚重复。
            # 最新日志里大量坏局 steps_since_clean 只有 7~24，但 CleanRepeat 已经 0.55+，
            # 所以这里增加“重复密度”信号：当前踩旧格、最近窗口很窄、且没有清扫收益时轻罚。
            if (
                bool(getattr(Config, "ENABLE_CLEAN_REPEAT_DENSITY_REWARD", False))
                and cleaned_this_step <= 0
                and local_cell != 2
                and visit_n >= int(getattr(Config, "CLEAN_REPEAT_DENSITY_VISIT_MIN", 2))
            ):
                density_start = float(getattr(Config, "CLEAN_REPEAT_DENSITY_RATIO", 0.32))
                density_hard = float(getattr(Config, "CLEAN_REPEAT_DENSITY_HARD_RATIO", 0.55))
                recent_unique_max = int(getattr(Config, "CLEAN_REPEAT_DENSITY_RECENT_UNIQUE_MAX", 6))
                if clean_repeat_ratio >= density_start and recent_unique <= recent_unique_max:
                    density_scale = 1.0 + min(2.0, (float(clean_repeat_ratio) - density_start) / max(1e-6, 1.0 - density_start))
                    density_penalty = float(getattr(Config, "CLEAN_REPEAT_DENSITY_PENALTY", -0.006))
                    if clean_repeat_ratio >= density_hard:
                        density_penalty = min(
                            density_penalty,
                            float(getattr(Config, "CLEAN_REPEAT_DENSITY_HARD_PENALTY", -0.010)),
                        )
                    clean_eff_reward += density_penalty * density_scale

            # 如果当前仍能看到/估计到污渍方向，接近污渍给一点点奖励，避免只靠惩罚导致策略保守。
            if (
                self.steps_since_clean >= stall_start
                and self.nearest_dirt_dist < self.last_nearest_dirt_dist
                and self.last_nearest_dirt_dist < 200.0
            ):
                clean_eff_reward += float(getattr(Config, "CLEAN_EFF_APPROACH_DIRT_REWARD", 0.003))

            reward += clean_eff_reward

        # ===== 视频问题修正：短周期来回走 / 小圈循环惩罚 =====
        # 只通过 reward 轻量惩罚，不改动作、不改 BFS 排序。
        # 普通清扫阶段很轻；低电和 force_charge 阶段加权，重点减少“来回走到没电”。
        if (
            bool(getattr(Config, "ENABLE_SHORT_LOOP_REWARD", False))
            and not is_charging
            and cleaned_this_step <= 0
            and len(self.pos_history) >= 3
        ):
            loop_penalty = 0.0
            pos_hist = list(self.pos_history)
            cur_hist_pos = pos_hist[-1]

            # ABAB：当前位置等于两步前，典型来回横跳。
            if len(pos_hist) >= 3 and cur_hist_pos == pos_hist[-3]:
                loop_penalty += float(getattr(Config, "SHORT_LOOP_ABAB_PENALTY", -0.006))

            # 小三角/三步循环：当前位置等于三步前。
            if len(pos_hist) >= 4 and cur_hist_pos == pos_hist[-4]:
                loop_penalty += float(getattr(Config, "SHORT_LOOP_TRIANGLE_PENALTY", -0.005))

            # 最近窗口内多次回到同一格，说明局部重复。
            short_window = max(4, int(getattr(Config, "SHORT_LOOP_WINDOW", 8)))
            recent_loop_window = pos_hist[-short_window:]
            revisit_hits = max(0, recent_loop_window.count(cur_hist_pos) - 1)
            if revisit_hits >= 2:
                loop_penalty += float(getattr(Config, "SHORT_LOOP_RECENT_REVISIT_PENALTY", -0.003)) * float(
                    min(3, revisit_hits - 1)
                )

            if loop_penalty < 0.0:
                loop_scale = 1.0
                if battery_ratio < float(getattr(Config, "SHORT_LOOP_LOW_BATTERY_RATIO", 0.45)):
                    loop_scale *= float(getattr(Config, "SHORT_LOOP_LOW_BATTERY_SCALE", 2.0))
                if bool(force_charge_mode):
                    loop_scale *= float(getattr(Config, "SHORT_LOOP_FORCE_CHARGE_SCALE", 2.5))
                reward += loop_penalty * loop_scale

        cur_min_charger_dist = self._min_dist(self.cur_pos, self.charger_list)
        cur_charge_zone_dist = self._min_chebyshev_dist(self.cur_pos, self.charger_list)
        safe_charger_dist = None
        battery_risk_enabled = bool(getattr(Config, "ENABLE_BATTERY_RISK_REWARD", False))
        start_ratio = float(getattr(Config, "BATTERY_RISK_START_RATIO", 0.50))
        strong_ratio = float(getattr(Config, "BATTERY_RISK_STRONG_RATIO", 0.34))

        # 预回充势能 shaping：在 force_charge 触发前，给模型一个轻量“靠近充电桩”的方向感。
        # 只用于 reward，不改变动作、不改变 BFS 排序、不提高强制回充阈值。
        self.last_pre_charge_reward = 0.0
        pre_charge_enabled = bool(getattr(Config, "ENABLE_PRE_CHARGE_POTENTIAL_REWARD", False))
        pre_charge_start = float(getattr(Config, "PRE_CHARGE_START_RATIO", 0.55))
        pre_charge_end = float(getattr(Config, "PRE_CHARGE_END_RATIO", 0.30))
        need_safe_dist = bool(
            (battery_risk_enabled and battery_ratio < start_ratio)
            or (pre_charge_enabled and pre_charge_end <= battery_ratio < pre_charge_start)
        )
        if need_safe_dist and not is_charging:
            safe_charger_dist = self._safe_charger_dist_for_reward(charger_path_dist, cur_min_charger_dist)
            self.reward_safe_charger_dist = safe_charger_dist

        if battery_risk_enabled and battery_ratio < start_ratio and not is_charging:
            if safe_charger_dist is None:
                safe_charger_dist = self._safe_charger_dist_for_reward(charger_path_dist, cur_min_charger_dist)
                self.reward_safe_charger_dist = safe_charger_dist
            if safe_charger_dist is not None:
                span = max(start_ratio - strong_ratio, 1e-6)
                risk_weight = 1.0 + 0.25 * min(1.0, max(0.0, (start_ratio - battery_ratio) / span))

                last_safe_dist = self.last_safe_charger_dist_for_reward
                if last_safe_dist is None:
                    last_safe_dist = self._safe_charger_dist_for_reward(
                        last_charger_path_dist,
                        self.last_min_charger_dist,
                    )
                if last_safe_dist is not None:
                    if safe_charger_dist < last_safe_dist:
                        reward += float(getattr(Config, "BATTERY_RISK_APPROACH_REWARD", 0.010)) * risk_weight
                    elif safe_charger_dist > last_safe_dist:
                        reward += float(getattr(Config, "BATTERY_RISK_BACKOFF_PENALTY", -0.018)) * risk_weight

                far_dist = int(getattr(Config, "BATTERY_RISK_FAR_DIST", 12))
                if battery_ratio <= strong_ratio and int(safe_charger_dist) >= far_dist:
                    reward += float(getattr(Config, "BATTERY_RISK_FAR_PENALTY", -0.008)) * risk_weight

                # 极低电惩罚只在“仍然离充电桩较远”时触发。
                # 避免低电阶段每一步都被强负反馈扰乱策略。
                if (
                    battery_ratio <= strong_ratio
                    and not self.battery_gain
                    and int(safe_charger_dist) >= far_dist
                ):
                    reward += float(getattr(Config, "BATTERY_RISK_CRITICAL_PENALTY", -0.010)) * risk_weight

                # 低电阶段最近几步没有让充电桩距离变短，给轻惩罚。
                # 这是势能 shaping：只告诉模型“低电时要持续变安全”，不替它选动作。
                if bool(getattr(Config, "ENABLE_BATTERY_NO_PROGRESS_REWARD", False)) and not self.battery_gain:
                    progress_window = max(3, int(getattr(Config, "BATTERY_NO_PROGRESS_WINDOW", 6)))
                    dist_hist = list(self.safe_charger_dist_history)
                    if len(dist_hist) >= progress_window:
                        recent_best_dist = min(dist_hist[-progress_window:])
                        if int(safe_charger_dist) >= int(recent_best_dist):
                            no_progress_scale = 1.0
                            if bool(force_charge_mode):
                                no_progress_scale *= float(getattr(Config, "BATTERY_NO_PROGRESS_FORCE_SCALE", 2.0))
                            reward += float(getattr(Config, "BATTERY_NO_PROGRESS_PENALTY", -0.004)) * no_progress_scale

                near_dist = int(getattr(Config, "BATTERY_RISK_NEAR_DIST", 3))
                near_zone_dist = int(getattr(Config, "CHARGER_NEAR_DIST", 1))
                near_charger = int(safe_charger_dist) <= near_dist or (
                    cur_charge_zone_dist is not None and int(cur_charge_zone_dist) <= near_zone_dist
                )
                # 近桩没回血只在真正回充/充电区上下文里惩罚，
                # 不惩罚普通清扫时路过充电桩附近。
                charge_context = bool(
                    force_charge_mode
                    or self.in_charger_zone
                    or self.is_charging
                    or int(charge_zone_no_gain_steps or 0) > 0
                )
                if (
                    battery_ratio <= strong_ratio
                    and charge_context
                    and near_charger
                    and not self.battery_gain
                ):
                    reward += float(getattr(Config, "BATTERY_RISK_NEAR_NO_GAIN_PENALTY", -0.010))
        elif safe_charger_dist is None:
            self.reward_safe_charger_dist = None

        # 预回充势能 shaping：电量中等偏低时，奖励持续接近充电桩，轻罚无进展。
        # 这不是硬规则，只是提前培养“回充意识”，针对 200 步才开始回充的早死。
        if (
            pre_charge_enabled
            and not is_charging
            and safe_charger_dist is not None
            and pre_charge_end <= battery_ratio < pre_charge_start
            and int(safe_charger_dist) >= int(getattr(Config, "PRE_CHARGE_MIN_DIST", 10))
        ):
            pre_reward = 0.0
            last_safe_dist = self.last_safe_charger_dist_for_reward
            if last_safe_dist is not None:
                if int(safe_charger_dist) < int(last_safe_dist):
                    pre_reward += float(getattr(Config, "PRE_CHARGE_APPROACH_REWARD", 0.003))
                elif int(safe_charger_dist) > int(last_safe_dist):
                    pre_reward += float(getattr(Config, "PRE_CHARGE_BACKOFF_PENALTY", -0.002))

            pre_window = max(3, int(getattr(Config, "PRE_CHARGE_NO_PROGRESS_WINDOW", 8)))
            dist_hist = list(self.safe_charger_dist_history)[-pre_window:]
            if len(dist_hist) >= pre_window and min(dist_hist) <= int(safe_charger_dist):
                pre_reward += float(getattr(Config, "PRE_CHARGE_NO_PROGRESS_PENALTY", -0.002))

            if pre_reward != 0.0:
                reward += pre_reward
                self.last_pre_charge_reward = pre_reward

        if False:
            pass
            # 只给轻量方向信号：靠近充电桩小奖，远离小罚，不做任何强制动作。
            last_safe_dist = self.last_safe_charger_dist_for_reward
            if last_safe_dist is None:
                last_safe_dist = self._safe_charger_dist_for_reward(last_charger_path_dist, self.last_min_charger_dist)
            if last_safe_dist is not None:
                if safe_charger_dist < last_safe_dist:
                    reward += float(getattr(Config, "BATTERY_RISK_APPROACH_REWARD", 0.008))
                elif safe_charger_dist > last_safe_dist:
                    reward += float(getattr(Config, "BATTERY_RISK_BACKOFF_PENALTY", -0.012))

            reserve_steps = float(getattr(Config, "BATTERY_RISK_RESERVE_STEPS", 12))
            if float(safe_charger_dist) + reserve_steps > float(max(self.battery, 1)):
                reward += float(getattr(Config, "BATTERY_RISK_FAR_PENALTY", -0.003))
                if battery_ratio <= float(getattr(Config, "BATTERY_RISK_STRONG_RATIO", 0.30)):
                    reward += float(getattr(Config, "BATTERY_RISK_CRITICAL_PENALTY", -0.012))
        if (
            not battery_risk_enabled
            and battery_ratio < self.LOW_BATTERY_RATIO
            and cur_charge_zone_dist is not None
            and self.last_charge_zone_dist is not None
        ):
            near_dist = int(getattr(Config, "CHARGER_NEAR_DIST", 1))
            was_in_charge_zone = self.last_charge_zone_dist <= near_dist
            is_in_charge_zone = cur_charge_zone_dist <= near_dist
            if is_in_charge_zone and was_in_charge_zone:
                pass
            elif cur_charge_zone_dist < self.last_charge_zone_dist:
                reward += Config.LOW_BATTERY_APPROACH_CHARGER_REWARD
            elif cur_charge_zone_dist > self.last_charge_zone_dist:
                reward += Config.LOW_BATTERY_LEAVE_CHARGER_PENALTY

        if (
            not battery_risk_enabled
            and battery_ratio < self.LOW_BATTERY_RATIO
            and not is_charging
            and cur_min_charger_dist is not None
            and self.last_min_charger_dist is not None
        ):
            low_battery_loop = len(recent_positions) >= 6 and recent_unique <= 4 and visit_n > 1
            if cur_min_charger_dist > self.last_min_charger_dist:
                reward += Config.LOW_BATTERY_CHARGER_BACKOFF_PENALTY
            elif cur_min_charger_dist == self.last_min_charger_dist and (low_battery_loop or visit_n > 2):
                reward += Config.LOW_BATTERY_CHARGER_STALL_PENALTY
            if low_battery_loop and cur_min_charger_dist >= self.last_min_charger_dist:
                reward += Config.LOW_BATTERY_CHARGE_LOOP_PENALTY

        if self.steps_since_clean >= stall_threshold and not is_charging:
            stale_repeat = (
                visit_n > 1
                and self._local_cell(self.cur_pos) != 2
                and self.nearest_dirt_dist >= self.last_nearest_dirt_dist
            )
            if stale_repeat:
                stall_level = min(4, 1 + (self.steps_since_clean - stall_threshold) // 20)
                reward += Config.CLEAN_REPEAT_STALL_PENALTY * float(stall_level)
                if len(recent_positions) >= 6 and recent_unique <= 4:
                    reward += Config.CLEAN_REPEAT_LOOP_PENALTY * min(2.0, 0.5 * float(max(visit_n - 1, 1)))

        if self.last_battery is not None:
            battery_delta = self.battery - self.last_battery
            if battery_delta > 0:
                prev_battery_ratio = float(self.last_battery) / float(max(self.battery_max, 1))
                if prev_battery_ratio <= Config.EXIT_CHARGE_BATTERY_RATIO:
                    reward += min(Config.LOW_BATTERY_CHARGE_GAIN_REWARD_MAX, 0.01 + 0.001 * float(battery_delta))
                else:
                    reward += Config.UNNEEDED_CHARGE_PENALTY

        self.last_min_charger_dist = cur_min_charger_dist
        self.last_charge_zone_dist = cur_charge_zone_dist
        if safe_charger_dist is not None:
            self.last_safe_charger_dist_for_reward = safe_charger_dist
            self.safe_charger_dist_history.append(int(safe_charger_dist))
        else:
            # 不在低电/预回充阶段时，清空距离势能历史，避免跨阶段误判“无进展”。
            self.last_safe_charger_dist_for_reward = None
            self.safe_charger_dist_history.clear()

        cur_min_npc_dist = self._min_dist(self.cur_pos, self.npc_list)
        if cur_min_npc_dist is not None:
            if cur_min_npc_dist <= 1:
                reward += getattr(Config, "NPC_CLOSE_PENALTY", -0.20)
            elif cur_min_npc_dist <= self.NPC_DANGER_DIST:
                reward += getattr(Config, "NPC_DANGER_PENALTY", -0.08)
            elif cur_min_npc_dist <= self.NPC_AVOID_DIST:
                reward += getattr(Config, "NPC_AVOID_PENALTY", -0.03)

            if cleaned_this_step > 0 and cur_min_npc_dist <= self.NPC_AVOID_DIST:
                reward += getattr(Config, "NPC_NEAR_CLEAN_PENALTY", -0.04)
                if cur_min_npc_dist <= self.NPC_DANGER_DIST:
                    reward += getattr(Config, "NPC_DANGER_CLEAN_PENALTY", -0.08)

            if self.last_min_npc_dist is not None:
                if cur_min_npc_dist > self.last_min_npc_dist and self.last_min_npc_dist <= self.NPC_AVOID_DIST:
                    reward += getattr(Config, "NPC_MOVE_AWAY_REWARD", 0.02)
                elif cur_min_npc_dist < self.last_min_npc_dist and cur_min_npc_dist <= self.NPC_AVOID_DIST:
                    reward += getattr(Config, "NPC_MOVE_CLOSER_PENALTY", -0.02)
                elif (
                    cur_min_npc_dist == self.last_min_npc_dist
                    and cur_min_npc_dist <= self.NPC_AVOID_DIST
                ):
                    reward += getattr(Config, "NPC_STAY_NEAR_PENALTY", -0.01)

        self.last_min_npc_dist = cur_min_npc_dist

        if self.stuck_steps >= 1 and not is_charging:
            reward += Config.BLOCKED_ACTION_PENALTY
        if self.stuck_steps >= 2 and not is_charging:
            reward += Config.STUCK_PENALTY

        force_charge_mode = bool(force_charge_mode)
        charge_route_revisit_ratio = float(charge_route_revisit_ratio or 0.0)
        clean_repeat_ratio = float(clean_repeat_ratio or 0.0)
        post_charge_pressure = bool(post_charge_pressure)
        post_charge_escape_pressure = bool(post_charge_escape_pressure)
        post_charge_escape_ratio = float(post_charge_escape_ratio or 0.0)
        search_steps = int(force_charge_search_steps or 0)
        zone_no_gain_steps = int(charge_zone_no_gain_steps or 0)
        stall_path_min = int(getattr(Config, "PHASE_CHARGE_STALL_PATH_MIN", 8))
        repeat_ratio_limit = float(getattr(Config, "PHASE_CHARGE_REPEAT_RATIO", 0.45))
        search_stall_steps = int(getattr(Config, "CHARGE_UNKNOWN_ROUTE_STALL_STEPS", 12))
        long_search_steps = int(getattr(Config, "PHASE_CHARGE_LONG_SEARCH_STEPS", 18))
        near_path_min = int(getattr(Config, "CHARGE_KNOWN_NEAR_PATH_MIN_DIST", 4))
        near_path_max = int(getattr(Config, "CHARGE_KNOWN_NEAR_PATH_MAX_DIST", 10))
        mid_path_min = int(getattr(Config, "CHARGE_KNOWN_MID_PATH_MIN_DIST", near_path_max + 1))
        mid_path_max = int(getattr(Config, "CHARGE_KNOWN_MID_PATH_MAX_DIST", 20))

        if force_charge_mode and not is_charging:
            no_progress = (
                charger_path_dist is None
                or last_charger_path_dist is None
                or int(charger_path_dist) >= int(last_charger_path_dist)
            )

            # ===== Step2-lite：回充路线重复效率 shaping =====
            # 只在已经进入 force_charge、搜索步数足够、且电量不是极端濒死时轻量生效。
            # 不改变动作，只让模型学会回充路线不要反复踩旧路径。
            if (
                bool(getattr(Config, "ENABLE_CHARGE_ROUTE_EFFICIENCY_REWARD", False))
                and search_steps >= int(getattr(Config, "CHARGE_ROUTE_EFF_MIN_SEARCH_STEPS", 12))
                and battery_ratio >= float(getattr(Config, "CHARGE_ROUTE_EFF_MIN_BATTERY_RATIO", 0.12))
            ):
                route_eff_reward = 0.0
                route_repeat_start = float(getattr(Config, "CHARGE_ROUTE_EFF_REPEAT_RATIO_START", 0.45))
                if charge_route_revisit_ratio >= route_repeat_start:
                    route_repeat_scale = 1.0 + min(2.0, (charge_route_revisit_ratio - route_repeat_start) / max(1e-6, 1.0 - route_repeat_start))
                    route_eff_reward += float(getattr(Config, "CHARGE_ROUTE_EFF_REVISIT_PENALTY", -0.0025)) * route_repeat_scale

                if charger_path_dist is not None and last_charger_path_dist is not None:
                    path_delta_for_eff = int(last_charger_path_dist) - int(charger_path_dist)
                    if path_delta_for_eff > 0:
                        route_eff_reward += float(getattr(Config, "CHARGE_ROUTE_EFF_PROGRESS_REWARD", 0.002)) * float(
                            min(path_delta_for_eff, 3)
                        )
                    elif no_progress:
                        route_eff_reward += float(getattr(Config, "CHARGE_ROUTE_EFF_NO_PROGRESS_PENALTY", -0.003))
                elif no_progress:
                    route_eff_reward += 0.5 * float(getattr(Config, "CHARGE_ROUTE_EFF_NO_PROGRESS_PENALTY", -0.003))

                reward += route_eff_reward

            if charger_path_dist is not None and last_charger_path_dist is not None:
                path_delta = int(last_charger_path_dist) - int(charger_path_dist)
                if path_delta > 0:
                    reward += float(getattr(Config, "PHASE_CHARGE_PROGRESS_REWARD", 0.0)) * float(min(path_delta, 4))
                elif path_delta < 0:
                    reward += float(getattr(Config, "PHASE_CHARGE_BACKOFF_PENALTY", 0.0)) * float(
                        min(-path_delta, 4)
                    )

            if charger_path_dist is not None and int(charger_path_dist) >= stall_path_min:
                if search_steps >= search_stall_steps and no_progress:
                    reward += float(getattr(Config, "PHASE_CHARGE_STALL_PENALTY", 0.0))
                if charge_route_revisit_ratio >= repeat_ratio_limit and search_steps >= max(6, search_stall_steps // 2):
                    repeat_scale = 1.0 + min(
                        2.0,
                        (charge_route_revisit_ratio - repeat_ratio_limit) / max(1e-6, 1.0 - repeat_ratio_limit),
                    )
                    reward += float(getattr(Config, "PHASE_CHARGE_REPEAT_PENALTY", 0.0)) * repeat_scale
                if search_steps >= long_search_steps and no_progress:
                    long_search_scale = 1.0 + min(
                        2.0,
                        float(max(0, search_steps - long_search_steps)) / 20.0,
                    )
                    reward += float(getattr(Config, "PHASE_CHARGE_LONG_SEARCH_PENALTY", 0.0)) * long_search_scale

            if charger_path_dist is not None:
                dist = int(charger_path_dist)
                in_near_band = near_path_min <= dist <= near_path_max
                in_mid_band = mid_path_min <= dist <= mid_path_max
                if (
                    (in_near_band or in_mid_band)
                    and search_steps >= max(6 if in_near_band else 8, search_stall_steps // 2)
                ):
                    mid_near_repeat_limit = 0.35 if in_near_band else 0.42
                    if no_progress and charge_route_revisit_ratio >= mid_near_repeat_limit:
                        near_repeat_scale = 1.0 + min(
                            2.0,
                            max(0.0, charge_route_revisit_ratio - mid_near_repeat_limit)
                            / max(1e-6, 1.0 - mid_near_repeat_limit),
                        )
                        if in_mid_band:
                            near_repeat_scale *= 0.8
                        reward += float(getattr(Config, "PHASE_CHARGE_MID_NEAR_STALL_PENALTY", 0.0)) * near_repeat_scale

            if self._in_charger_zone(self.cur_pos) and zone_no_gain_steps > 0:
                zone_scale = 1.0 + 0.5 * float(min(zone_no_gain_steps, 4) - 1)
                reward += float(getattr(Config, "PHASE_CHARGE_ZONE_NO_GAIN_PENALTY", 0.0)) * zone_scale
        elif not is_charging:
            clean_repeat_limit = float(getattr(Config, "PHASE_CLEAN_REPEAT_RATIO", 0.16))
            if clean_repeat_ratio >= clean_repeat_limit and self.steps_since_clean >= stall_threshold:
                repeat_scale = 1.0 + min(
                    2.0,
                    (clean_repeat_ratio - clean_repeat_limit) / max(1e-6, 1.0 - clean_repeat_limit),
                )
                stale_scale = 1.0 + min(2.0, float(max(0, self.steps_since_clean - stall_threshold)) / 40.0)
                reward += float(getattr(Config, "PHASE_CLEAN_STALE_REPEAT_PENALTY", 0.0)) * (
                    0.5 * repeat_scale + 0.5 * stale_scale
                )
            pure_repeat_limit = float(getattr(Config, "PHASE_CLEAN_PURE_REPEAT_RATIO", 0.22))
            pure_repeat_unique_max = int(getattr(Config, "PHASE_CLEAN_PURE_REPEAT_UNIQUE_MAX", 4))
            if (
                not force_charge_mode
                and clean_repeat_ratio >= pure_repeat_limit
                and visit_n > 0
                and self._local_cell(self.cur_pos) != 2
                and len(recent_positions) >= 8
                and recent_unique <= pure_repeat_unique_max
                and self.nearest_dirt_dist >= self.last_nearest_dirt_dist
            ):
                pure_repeat_scale = 1.0 + min(
                    2.0,
                    (clean_repeat_ratio - pure_repeat_limit) / max(1e-6, 1.0 - pure_repeat_limit),
                )
                reward += float(getattr(Config, "PHASE_CLEAN_PURE_REPEAT_PENALTY", 0.0)) * pure_repeat_scale
            if (
                post_charge_pressure
                and visit_n > 0
                and self._local_cell(self.cur_pos) != 2
            ):
                early_repeat_limit = float(getattr(Config, "PHASE_POST_CHARGE_EARLY_REPEAT_RATIO", 0.12))
                early_repeat_min_steps = int(getattr(Config, "PHASE_POST_CHARGE_EARLY_REPEAT_MIN_STEPS", 8))
                if clean_repeat_ratio >= early_repeat_limit and self.steps_since_clean >= early_repeat_min_steps:
                    early_repeat_scale = 1.0 + min(
                        2.0,
                        (clean_repeat_ratio - early_repeat_limit) / max(1e-6, 1.0 - early_repeat_limit),
                    )
                    reward += float(getattr(Config, "PHASE_POST_CHARGE_EARLY_REPEAT_PENALTY", 0.0)) * early_repeat_scale
                if self.steps_since_clean >= max(12, stall_threshold // 2):
                    stale_scale = 1.0 + min(2.0, float(self.steps_since_clean) / float(max(stall_threshold, 1)))
                    reward += float(getattr(Config, "PHASE_POST_CHARGE_STALE_PENALTY", 0.0)) * stale_scale
            if (
                post_charge_escape_pressure
                and visit_n > 0
                and self._local_cell(self.cur_pos) != 2
            ):
                route_scale = 1.0 + min(
                    2.0,
                    max(0.0, post_charge_escape_ratio - float(getattr(Config, "POST_CHARGE_ESCAPE_ROUTE_RATIO", 0.48)))
                    / max(1e-6, 1.0 - float(getattr(Config, "POST_CHARGE_ESCAPE_ROUTE_RATIO", 0.48))),
                )
                visit_scale = 1.0 + min(2.0, float(max(visit_n - 1, 0)) / 2.0)
                reward += float(getattr(Config, "PHASE_POST_CHARGE_ESCAPE_PENALTY", 0.0)) * (
                    0.5 * route_scale + 0.5 * visit_scale
                )

        return float(reward)
