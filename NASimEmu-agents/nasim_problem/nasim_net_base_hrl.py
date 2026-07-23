#DHRL MODEL NEW
"""
python main.py \
../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
--test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
-device cpu -cpus 16 \
-epoch 100 -max_epochs 200 \
--no_debug \
-net_class NASimNetDHRL \
-force_continue_epochs 0 \
-use_a_t \
-episode_step_limit 400 \
-observation_format graph_v2 \
-lr 0.0007 \
-alpha_h 0.02 \
--sched_lr_rate 10000 \
--sched_lr_factor 0.8 \
--sched_lr_min 0.0003 \
--sched_alpha_h_rate 15000 \
--sched_alpha_h_factor 0.5 \
--sched_alpha_h_min 0.005
"""

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv
from torch_scatter import scatter

from config import config
from graph_nns import (
    MultiMessagePassingWithGlobalNodeAndLastGRUGlobal,
    positional_encoding,
)
from net import Net
from nasimemu.nasim.envs.host_vector import HostVector
from rl import ppo
from .net_utils import compute_v_target, segmented_sample

import wandb


def wandb_log_safe(data: dict, **kwargs):
    run = getattr(wandb, "run", None)
    if run is not None:
        try:
            wandb.log(data, **kwargs)
        except Exception:
            pass


class NASimNetDHRL(Net):
    """
    Deep hierarchical graph policy inspired by the stronger GNN-LSTM and FeudalGTM
    agents. A manager GNN with recurrent global context selects persistent subgoals,
    while a goal-conditioned worker performs node-level actions.
    """

    def __init__(self):
        super().__init__()

        obs_dim = config.node_dim + config.pos_enc_dim
        self.goal_dim = getattr(config, "dhrl_goal_dim", 128)
        self.num_subgoals = getattr(config, "dhrl_num_subgoals", 8)
        self.goal_horizon = getattr(config, "dhrl_goal_horizon", 4)

        # High-level encoder (manager)
        self.embed_node = nn.Sequential(nn.Linear(obs_dim, config.emb_dim), nn.LeakyReLU())
        self.manager_gnn = MultiMessagePassingWithGlobalNodeAndLastGRUGlobal(config.mp_iterations)
        self.manager_norm = nn.LayerNorm(config.emb_dim)

        # Subgoal selection / persistence
        self.subgoal_head = nn.Sequential(
            nn.Linear(config.emb_dim, 128), nn.LeakyReLU(), nn.Linear(128, self.num_subgoals)
        )
        self.subgoal_switch = nn.Sequential(
            nn.Linear(config.emb_dim, 64), nn.LeakyReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.goal_bank = nn.Parameter(torch.randn(self.num_subgoals, self.goal_dim) * 0.1)
        self.goal_refine = nn.Sequential(
            nn.Linear(self.goal_dim + config.emb_dim, self.goal_dim), nn.LeakyReLU()
        )

        # Worker conditioned on subgoal-aware context
        self.worker_gat = GATConv(
            config.emb_dim + self.goal_dim, config.emb_dim, heads=2, concat=False
        )
        self.worker_norm = nn.LayerNorm(config.emb_dim)
        self.action_head = nn.Linear(config.emb_dim, config.action_dim)

        # Value / termination heads
        self.value_head = nn.Sequential(
            nn.Linear(config.emb_dim + self.goal_dim + 1, config.emb_dim),
            nn.LeakyReLU(),
            nn.Linear(config.emb_dim, 1),
        )
        self.termination_head = nn.Sequential(
            nn.Linear(config.emb_dim + self.goal_dim + 1, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.termination_head[-2].bias, -1.0)

        # IDS-aware context to bias towards stealthier plans
        self.ids_projection = nn.Sequential(
            nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, config.emb_dim)
        )

        self.opt = torch.optim.AdamW(
            self.parameters(), lr=config.opt_lr, weight_decay=config.opt_l2
        )
        self.to(config.device)

        # Persistent hierarchical state
        self.current_subgoal_idx = None
        self.current_goal_vec = None
        self.subgoal_steps_remaining = None
        self.batch_ind = None
        self.force_continue = False
        self.ep_t = None
        self.ep_limit = getattr(config, "step_limit", getattr(config, "episode_step_limit", 400))

    @staticmethod
    def prepare_batch(s_batch):
        node_feats, edge_index, node_index, pos_index = zip(*s_batch)
        node_feats = [torch.tensor(x, dtype=torch.float32, device=config.device) for x in node_feats]
        edge_index = [torch.tensor(x, dtype=torch.int64, device=config.device) for x in edge_index]
        data = [Data(x=node_feats[i], edge_index=edge_index[i]) for i in range(len(s_batch))]
        data_lens = [x.num_nodes for x in data]
        batch = Batch.from_data_list(data)
        batch_ind = batch.batch.to(config.device)
        node_index = np.concatenate(node_index)
        pos_index = torch.tensor(np.concatenate(pos_index)).to(config.device)
        return data, data_lens, batch, batch_ind, node_index, pos_index

    def _aggregate_ids_features(self, batch, batch_ind):
        # Mimic FeudalGTM's IDS biasing to encourage stealthy sequences.
        device = batch.x.device
        num_graphs = batch.num_graphs if hasattr(batch, "num_graphs") else None
        if num_graphs is None or num_graphs == 0:
            num_graphs = int(batch_ind.max().item()) + 1 if batch_ind.numel() > 0 else 0

        if num_graphs == 0:
            return torch.zeros(0, 3, device=device)

        detection_level_idx = getattr(HostVector, "_detection_level_idx", None)
        detection_multiplier_idx = getattr(HostVector, "_detection_multiplier_idx", None)
        if detection_multiplier_idx is None or detection_level_idx is None:
            return torch.zeros(num_graphs, 3, device=device)

        host_mask = batch.x[:, 0] == 0
        if not host_mask.any():
            return torch.zeros(num_graphs, 3, device=device)

        detection_slice = slice(1 + detection_level_idx, 1 + detection_multiplier_idx + 1)
        host_detection = batch.x[host_mask, detection_slice]
        host_batches = batch_ind[host_mask]

        detection_sum = scatter(host_detection, host_batches, dim=0, dim_size=num_graphs)
        counts = (
            scatter(
                torch.ones(host_batches.shape[0], dtype=torch.float32, device=device),
                host_batches,
                dim=0,
                dim_size=num_graphs,
            )
            .unsqueeze(1)
            .clamp_min(1.0)
        )
        return detection_sum / counts

    def _ensure_subgoal_state(self, batch_size):
        if (
            self.current_subgoal_idx is None
            or self.current_subgoal_idx.numel() != batch_size
            or self.subgoal_steps_remaining is None
            or self.subgoal_steps_remaining.numel() != batch_size
            or self.current_goal_vec is None
            or self.current_goal_vec.size(0) != batch_size
        ):
            self.current_subgoal_idx = torch.zeros(
                batch_size, dtype=torch.long, device=config.device
            )
            self.subgoal_steps_remaining = torch.zeros(
                batch_size, dtype=torch.long, device=config.device
            )
            self.current_goal_vec = torch.zeros(
                batch_size, self.goal_dim, dtype=torch.float32, device=config.device
            )

    def forward(self, s_batch, only_v=False, force_action=None, reset_hidden=False):
        data, data_lens, batch, batch_ind, node_index, pos_index = self.prepare_batch(s_batch)
        x = batch.x
        batch_size = len(data_lens)
        ids_summary = self._aggregate_ids_features(batch, batch_ind)

        # Support PPO replay by preserving/restoring state on demand
        if reset_hidden:
            saved_state = (
                None if self.current_subgoal_idx is None else self.current_subgoal_idx.clone(),
                None
                if self.subgoal_steps_remaining is None
                else self.subgoal_steps_remaining.clone(),
                None if self.current_goal_vec is None else self.current_goal_vec.clone(),
                self.batch_ind.clone() if self.batch_ind is not None else None,
                None if self.ep_t is None else self.ep_t.clone(),
                None if self.manager_gnn.hidden is None else self.manager_gnn.hidden.clone(),
            )
            self.reset_state()

        # Positional encoding + embedding
        pos_enc = positional_encoding(pos_index, config.pos_enc_dim)
        x = torch.cat([x, pos_enc], dim=1)
        x = self.embed_node(x)

        # Manager message passing with recurrent global state
        x_mgr, x_global = self.manager_gnn(
            x, None, batch.edge_attr, batch.edge_index, batch_ind, batch.num_graphs, data_lens
        )
        x_global = self.manager_norm(x_global)
        ids_bias = self.ids_projection(ids_summary)
        x_global = x_global + ids_bias

        self._ensure_subgoal_state(batch_size)
        self.batch_ind = batch_ind

        if self.ep_t is None or self.ep_t.numel() != batch_size:
            self.ep_t = torch.zeros(batch_size, device=config.device, dtype=torch.long)
        tfrac = (self.ep_t.float() / float(self.ep_limit)).clamp(0.0, 1.0).unsqueeze(1)
        self.ep_t = self.ep_t + 1

        forced_action = None
        forced_terminate = None
        forced_subgoal = None
        if force_action is not None:
            if isinstance(force_action, (tuple, list)):
                if len(force_action) > 0:
                    forced_action = force_action[0]
                if len(force_action) > 1:
                    forced_terminate = force_action[1]
                if len(force_action) > 2:
                    forced_subgoal = force_action[2]
            else:
                forced_action = force_action

        # Subgoal sampling & persistence
        subgoal_logits = self.subgoal_head(x_global)

        subgoal_probs = torch.softmax(subgoal_logits, dim=-1)
        switch_probs = self.subgoal_switch(x_global).clamp(1e-6, 1 - 1e-6)

        need_new_subgoal = self.subgoal_steps_remaining <= 0
        selected_subgoals = self.current_subgoal_idx.clone()

        if forced_subgoal is not None:
            selected_subgoals = forced_subgoal.to(config.device).long()
            switch_mask = torch.ones_like(selected_subgoals, dtype=torch.bool, device=config.device)
            policy_switch_mask = switch_mask
        else:
            if self.training:
                sampled_switch = torch.distributions.Bernoulli(probs=switch_probs).sample().bool()
            else:
                sampled_switch = switch_probs.flatten() > 0.5
            switch_mask = torch.logical_or(need_new_subgoal, sampled_switch.flatten())
            policy_switch_mask = torch.logical_and(switch_mask, ~need_new_subgoal)
            sample_mask = torch.logical_or(switch_mask, need_new_subgoal)
            if sample_mask.any():
                if self.training:
                    subgoal_dist = torch.distributions.Categorical(subgoal_probs[sample_mask])
                    new_samples = subgoal_dist.sample()
                else:
                    new_samples = torch.argmax(subgoal_probs[sample_mask], dim=-1)
                selected_subgoals[sample_mask] = new_samples

        goal_vectors = self.goal_bank[selected_subgoals]
        goal_vectors = self.goal_refine(torch.cat([goal_vectors, x_global], dim=1))

        goal_vectors_detached = goal_vectors.detach().clone()
        self.current_goal_vec = goal_vectors_detached
        self.current_subgoal_idx = selected_subgoals.detach().clone()

        subgoal_prob_selected = torch.gather(
            subgoal_probs, 1, selected_subgoals.unsqueeze(1)
        ).clamp_min(1e-9)
        subgoal_factor = torch.ones(batch_size, 1, device=config.device)
        continue_mask = ~switch_mask
        subgoal_factor[policy_switch_mask] = (
            switch_probs[policy_switch_mask] * subgoal_prob_selected[policy_switch_mask]
        ).clamp_min(1e-3)
        subgoal_factor[continue_mask] = (1 - switch_probs[continue_mask]).clamp_min(1e-3)
        subgoal_factor[need_new_subgoal] = subgoal_prob_selected[need_new_subgoal].clamp_min(1e-3)

        decayed_steps = torch.clamp_min(self.subgoal_steps_remaining - 1, 0)
        reset_steps = torch.full_like(decayed_steps, self.goal_horizon)
        next_steps = torch.where(switch_mask, reset_steps, decayed_steps)
        self.subgoal_steps_remaining = next_steps

        goal_per_node = goal_vectors[batch_ind]

        # Worker: goal-conditioned graph policy
        worker_input = torch.cat([x_mgr, goal_per_node], dim=1)
        worker_features = self.worker_gat(worker_input, batch.edge_index)
        worker_features = self.worker_norm(worker_features)
        action_logits = self.action_head(worker_features)

        # Value and termination on global goal-aware context
        value_context = torch.cat([x_global, goal_vectors, tfrac], dim=1)
        value = self.value_head(value_context)

        # Stronger time pressure to align shorter episodes with higher value
        time_penalty = (
            0.05 * tfrac
            + 0.03 * (tfrac > 0.5).float()
            + 0.02 * (tfrac > 0.75).float()
        )
        value = value - time_penalty
        if only_v:
            if reset_hidden:
                (
                    self.current_subgoal_idx,
                    self.subgoal_steps_remaining,
                    self.current_goal_vec,
                    self.batch_ind,
                    self.ep_t,
                    self.manager_gnn.hidden,
                ) = saved_state
            return value

        # Mask subnet nodes and graphs with no valid hosts
        subnet_mask = batch.x[:, 0] == 1
        valid_counts = torch.bincount(batch_ind, (~subnet_mask).long(), minlength=batch_size)
        invalid_graphs = valid_counts == 0
        invalid_node_mask = invalid_graphs[batch_ind]
        action_logits[subnet_mask & ~invalid_node_mask] = -float("inf")
        action_logits[invalid_node_mask] = 0.0

        action_probs = torch_geometric.utils.softmax(
            action_logits.flatten(), torch.repeat_interleave(batch_ind, config.action_dim)
        )
        data_lens_a = [n_nodes * config.action_dim for n_nodes in data_lens]

        if forced_action is not None:
            action_selected = forced_action
        else:
            action_selected = segmented_sample(action_probs, data_lens_a)

        cum_lens = np.cumsum([0] + data_lens_a[:-1])
        a_index = torch.tensor(cum_lens, device=config.device) + action_selected
        a_prob = action_probs[a_index].view(-1, 1)

        n_index = a_index.cpu().numpy() // config.action_dim
        targets = node_index[n_index].reshape(-1, 2)
        a_id = (action_selected % config.action_dim).cpu().numpy()

        termination_probs = self.termination_head(value_context).clamp(1e-6, 1 - 1e-6)
        if forced_terminate is not None:
            terminate = forced_terminate.bool().to(config.device).flatten()
        else:
            if self.training:
                terminate = torch.bernoulli(termination_probs).bool().flatten()
            else:
                terminate = (termination_probs > 0.2).flatten()

        # Gate termination with progress and value to reduce random ends
        min_steps = int(0.25 * self.ep_limit)
        too_early = self.ep_t.flatten() < min_steps
        low_value = value.detach().flatten() <= 0.0
        terminate = terminate & (~too_early) & low_value

        if invalid_graphs.any():
            termination_probs = termination_probs.clone()
            termination_probs[invalid_graphs] = 1.0
            terminate = terminate.clone()
            terminate[invalid_graphs] = True

        if self.force_continue:
            terminate = torch.zeros_like(terminate, dtype=torch.bool, device=config.device)
            continue_probs = (1 - termination_probs).clamp(1e-6, 1.0)
        else:
            a_id[terminate.cpu().numpy()] = -1
            continue_probs = (1 - termination_probs).clamp(1e-6, 1.0)

        if self.subgoal_steps_remaining is not None:
            self.subgoal_steps_remaining[terminate] = 0

        total_prob = torch.empty_like(a_prob)
        cont_mask = ~terminate
        total_prob[cont_mask] = (
            a_prob[cont_mask] * continue_probs[cont_mask] * subgoal_factor[cont_mask]
        )
        total_prob[terminate] = termination_probs[terminate].clamp_min(1e-6)
        if invalid_graphs.any():
            total_prob[invalid_graphs] = 1.0

        raw_actions = (action_selected.detach(), terminate.detach(), selected_subgoals.detach())

        wandb_log_safe(
            {
                "value_mean": value.mean().item(),
                "subgoal_switch": switch_mask.float().mean().item(),
                "termination_mean": terminate.float().mean().item(),
            },
            commit=False,
        )

        if reset_hidden:
            (
                self.current_subgoal_idx,
                self.subgoal_steps_remaining,
                self.current_goal_vec,
                self.batch_ind,
                self.ep_t,
                self.manager_gnn.hidden,
            ) = saved_state

        return list(zip(targets, a_id)), value, total_prob, raw_actions

    def set_force_continue(self, force):
        self.force_continue = force

    def reset_state(self, batch_mask=None):
        if batch_mask is None:
            self.current_subgoal_idx = None
            self.subgoal_steps_remaining = None
            self.current_goal_vec = None
            self.batch_ind = None
            self.manager_gnn.reset_state(batch_mask)
        else:
            if not isinstance(batch_mask, torch.Tensor):
                batch_mask = torch.tensor(batch_mask, dtype=torch.bool, device=config.device)
            if self.ep_t is not None and batch_mask.numel() > 0:
                mask = batch_mask.to(config.device)
                if self.ep_t.numel() < mask.numel():
                    pad = torch.zeros(mask.numel() - self.ep_t.numel(), device=config.device, dtype=self.ep_t.dtype)
                    self.ep_t = torch.cat([self.ep_t, pad], dim=0)
                self.ep_t[mask] = 0
            if self.current_subgoal_idx is not None and batch_mask.numel() > 0:
                mask = batch_mask.to(config.device)
                self.current_subgoal_idx[mask] = 0
                if self.subgoal_steps_remaining is not None:
                    self.subgoal_steps_remaining[mask] = 0
                if self.current_goal_vec is not None:
                    self.current_goal_vec[mask] = 0
            self.manager_gnn.reset_state(batch_mask)

    def clone_state(self, other):
        self.manager_gnn.clone_state(other.manager_gnn)
        self.current_subgoal_idx = (
            None if other.current_subgoal_idx is None else other.current_subgoal_idx.clone().detach()
        )
        self.subgoal_steps_remaining = (
            None
            if other.subgoal_steps_remaining is None
            else other.subgoal_steps_remaining.clone().detach()
        )
        self.current_goal_vec = (
            None if other.current_goal_vec is None else other.current_goal_vec.clone().detach()
        )
        self.batch_ind = None if other.batch_ind is None else other.batch_ind.clone().detach()
        self.ep_t = None if other.ep_t is None else other.ep_t.clone().detach()

    def update(self, trace, target_net=None, hidden_s0=None):
        """
        PPO update treated as recurrent because the manager GNN holds temporal state.
        """
        sx, a, a_cnt, r, sx_, d = zip(*trace)

        s = np.empty((config.ppo_t, config.batch), dtype=object)
        s[:, :] = sx

        s_ = np.empty((config.ppo_t, config.batch), dtype=object)
        s_[:, :] = sx_

        r = np.vstack(r)
        d = np.vstack(d)

        if target_net is None:
            target_net = self

        if hidden_s0 is None:
            raise ValueError("NASimNetDHRL.update requires hidden_s0 for recurrent PPO replay")

        with torch.no_grad():
            target_net.clone_state(hidden_s0)
            for t in range(config.ppo_t):
                target_net(s[t], force_action=a[t])
                target_net.reset_state(d[t])
            v_ = target_net(s_[-1], only_v=True).detach().flatten()

        v_target = compute_v_target(r, v_, d, config.gamma, config.ppo_t, config.batch, config.use_a_t)
        v_target = v_target.flatten()
        a_cnt = torch.tensor(np.concatenate(a_cnt), dtype=torch.bool, device=self.device)

        wandb_log_safe({"v_value": v_.mean().item(), "v_target": v_target.mean().item()}, commit=False)

        return ppo(
            s,
            a,
            a_cnt,
            d,
            v_target,
            self,
            config.gamma,
            config.alpha_v,
            config.alpha_h,
            config.ppo_k,
            config.ppo_eps,
            config.use_a_t,
            config.v_range,
            True,
            hidden_s0,
            False,
        )

    def _update(self, loss):
        self.opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), config.opt_max_norm)
        self.opt.step()
        wandb_log_safe({"grad_norm": grad_norm.item()}, commit=False)
        return grad_norm


# Alias for backward compatibility with any prior references
SimpleHRL = NASimNetDHRL
