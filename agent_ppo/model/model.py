#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

CNN + MLP compatible policy network for Robot Vacuum.

This model intentionally restores the memory-augmented architecture used by
checkpoint 71467/65675 instead of flattening all map features into a single
small MLP.  The legacy 73D base branch keeps the old MLP policy/value path,
while local/global map memory is processed by CNN delta branches and scalar
state is processed by a small MLP branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_ppo.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    """Create a linear layer with orthogonal initialization."""
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _make_conv(in_ch, out_ch, kernel_size=3, padding=1, gain=1.41421):
    """Create a convolution layer with orthogonal initialization."""
    layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    """Compatibility-preserving CNN+MLP policy/value network.

    Observation layout is controlled by Config.FEATURES:
      base   : 73D legacy feature vector
      local  : 7x7x12 local memory map, emitted cell-major by agent.py
      global : 16x16x14 global memory map, emitted channel-major by agent.py
      scalar : 32D phase/risk/route feature vector

    The parameter names and shapes intentionally match checkpoint 71467:
      backbone / actor_head / critic_head keep the legacy branch;
      spatial_branch / global_branch / scalar_branch / fusion / phase heads
      restore the trained memory branches instead of starting them from scratch.
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        self.base_len = int(getattr(Config, "BASE_FEATURE_LEN", 73))
        self.local_size = int(getattr(Config, "AUGMENTED_LOCAL_SIZE", 7))
        self.local_channels = int(getattr(Config, "AUGMENTED_LOCAL_CHANNELS", 12))
        self.local_len = int(getattr(Config, "AUGMENTED_LOCAL_LEN", self.local_size * self.local_size * self.local_channels))
        self.global_size = int(getattr(Config, "AUGMENTED_GLOBAL_SIZE", 16))
        self.global_channels = int(getattr(Config, "AUGMENTED_GLOBAL_CHANNELS", 14))
        self.global_len = int(getattr(Config, "AUGMENTED_GLOBAL_LEN", self.global_size * self.global_size * self.global_channels))
        self.scalar_len = int(getattr(Config, "AUGMENTED_SCALAR_LEN", 32))
        self.act_num = int(getattr(Config, "ACTION_NUM", 8))

        # Legacy 73D base branch.  Keep names/shapes to load old policy weights.
        self.backbone = nn.Sequential(
            _make_fc(self.base_len, 128),
            nn.ReLU(),
            _make_fc(128, 64),
            nn.ReLU(),
        )
        self.actor_head = _make_fc(64, self.act_num, gain=0.01)
        self.critic_head = _make_fc(64, 1, gain=0.01)

        # Local 7x7x12 memory CNN branch.
        self.spatial_branch = nn.Sequential(
            _make_conv(self.local_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            _make_conv(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            _make_fc(32 * self.local_size * self.local_size, 64),
            nn.ReLU(),
        )

        # Scalar phase/risk/route branch.
        self.scalar_branch = nn.Sequential(
            _make_fc(self.scalar_len, 32),
            nn.ReLU(),
            _make_fc(32, 32),
            nn.ReLU(),
        )

        # Global 16x16x14 memory CNN branch.
        self.global_branch = nn.Sequential(
            _make_conv(self.global_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            _make_conv(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            _make_fc(32 * self.global_size * self.global_size, 64),
            nn.ReLU(),
        )

        # Local/scalar fusion delta branch: 64 base + 64 local + 32 scalar = 160.
        self.fusion = nn.Sequential(
            _make_fc(64 + 64 + 32, 64),
            nn.ReLU(),
        )

        # Global context for phase-aware value heads: 64 fused + 64 global = 128.
        self.phase_context = nn.Sequential(
            _make_fc(64 + 64, 64),
            nn.ReLU(),
        )
        self.phase_gate = nn.Sequential(
            _make_fc(32, 16),
            nn.ReLU(),
            _make_fc(16, 3, gain=0.01),
        )

        # Delta heads restored from 71467/65675 checkpoints.
        self.actor_delta_head = _make_fc(64, self.act_num, gain=0.01)
        self.critic_delta_head = _make_fc(64, 1, gain=0.01)
        self.actor_global_delta_head = _make_fc(64, self.act_num, gain=0.01)
        self.critic_global_delta_head = _make_fc(64, 1, gain=0.01)
        self.critic_clean_delta_head = _make_fc(64, 1, gain=0.01)
        self.critic_charge_delta_head = _make_fc(64, 1, gain=0.01)
        self.critic_post_charge_delta_head = _make_fc(64, 1, gain=0.01)

    def _split_observation(self, s):
        """Split flat observation into base/local/global/scalar tensors."""
        x = s.to(torch.float32)
        expected = self.base_len + self.local_len + self.global_len + self.scalar_len
        if x.dim() == 1:
            x = x.view(1, -1)
        if x.size(1) < expected:
            pad = x.new_zeros((x.size(0), expected - x.size(1)))
            x = torch.cat([x, pad], dim=1)
        elif x.size(1) > expected:
            x = x[:, :expected]

        start = 0
        base = x[:, start : start + self.base_len]
        start += self.base_len
        local_flat = x[:, start : start + self.local_len]
        start += self.local_len
        global_flat = x[:, start : start + self.global_len]
        start += self.global_len
        scalar = x[:, start : start + self.scalar_len]

        # agent.py emits local memory cell-major: (H, W, C).  Convert to NCHW.
        local_map = local_flat.view(-1, self.local_size, self.local_size, self.local_channels)
        local_map = local_map.permute(0, 3, 1, 2).contiguous()

        # agent.py emits global memory channel-major already: (C, H, W).
        global_map = global_flat.view(-1, self.global_channels, self.global_size, self.global_size)
        return base, local_map, global_map, scalar

    def forward(self, s, inference=False):
        """Forward pass returning [action_logits, state_value]."""
        base, local_map, global_map, scalar = self._split_observation(s)

        base_h = self.backbone(base)
        local_h = self.spatial_branch(local_map)
        scalar_h = self.scalar_branch(scalar)
        global_h = self.global_branch(global_map)

        fused_h = self.fusion(torch.cat([base_h, local_h, scalar_h], dim=1))
        context_h = self.phase_context(torch.cat([fused_h, global_h], dim=1))

        # Keep the legacy policy/value path and add learned CNN/scalar deltas.
        logits = (
            self.actor_head(base_h)
            + self.actor_delta_head(fused_h)
            + self.actor_global_delta_head(global_h)
        )

        base_value = self.critic_head(base_h)
        value = base_value + self.critic_delta_head(fused_h) + self.critic_global_delta_head(global_h)

        phase_gate = F.softmax(self.phase_gate(scalar_h), dim=1)
        clean_v = self.critic_clean_delta_head(context_h)
        charge_v = self.critic_charge_delta_head(context_h)
        post_charge_v = self.critic_post_charge_delta_head(context_h)
        phase_value = (
            phase_gate[:, 0:1] * clean_v
            + phase_gate[:, 1:2] * charge_v
            + phase_gate[:, 2:3] * post_charge_v
        )
        value = value + phase_value

        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
