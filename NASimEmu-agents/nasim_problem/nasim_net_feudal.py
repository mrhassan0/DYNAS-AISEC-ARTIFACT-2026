#Feudal GTM MAIN 
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GATConv, global_mean_pool
from config import config
from net import Net
from nasimemu.nasim.envs.host_vector import HostVector

import torch, numpy as np
import torch_geometric

from numba import jit

from torch.nn import *
from torch_geometric.data import Data, Batch
from torch_scatter import scatter

from rl import a2c, ppo

from graph_nns import *
from .net_utils import *

import wandb


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim, heads=1)
    
    def forward(self, x, edge_index):
        return self.gat(x, edge_index)

# Simplified GraphMemoryModule without attention to avoid complexity
class GraphMemoryModule(nn.Module):
    def __init__(self, node_dim, memory_size):
        super().__init__()
        self.memory_size = memory_size
        self.input_proj = nn.Linear(node_dim, memory_size)
        self.message_gat = GATConv(memory_size, memory_size, heads=1, concat=False)
        self.gru = nn.GRUCell(memory_size, memory_size)
        
    def forward(self, state, abstract_goal, memory_state, edge_index):
        # state: [num_nodes, node_dim]
        # abstract_goal: [num_nodes, memory_size]
        # memory_state: [num_nodes, memory_size] - previous memory state
        # edge_index: graph connectivity for attention-based aggregation
        
        projected_state = self.input_proj(state)
        combined_state = projected_state + abstract_goal
        
        # Inject prior memory before neighborhood aggregation
        memory_enhanced = combined_state + memory_state
        neighbor_context = self.message_gat(memory_enhanced, edge_index)
        memory_input = memory_enhanced + neighbor_context
        
        new_memory = self.gru(memory_input, memory_state)
        return new_memory

class MetaSubgoalNetwork(nn.Module):
    def __init__(self, input_dim, num_subgoals):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_subgoals)
        )
    
    def forward(self, context):
        return self.net(context)

class FeudalGTM(Net):
    def __init__(self):
        super().__init__()
        observation_dim = config.node_dim + config.pos_enc_dim
        num_subgoals = 5
        self.num_subgoals = num_subgoals
        self.hidden_size = 64
        
        # Enhanced Manager Network with more capacity
        self.manager_gat1 = GraphAttentionLayer(observation_dim, 128)
        self.manager_gat2 = GraphAttentionLayer(128, 128)
        self.manager_lstm = nn.LSTM(128, self.hidden_size, batch_first=True)
        
        # Fixed Graph Temporal Memory
        self.gtm = GraphMemoryModule(
            node_dim=observation_dim,
            memory_size=256
        )
        
        # Enhanced Worker Network that uses subgoals
        self.worker = nn.Sequential(
            nn.Linear(256 + num_subgoals, 128),  # Now includes subgoal conditioning
            nn.LeakyReLU(),
            nn.Linear(128, config.action_dim)
        )
        
        # Subgoal Module
        self.subgoal_predictor = MetaSubgoalNetwork(256, num_subgoals)
        self.subgoal_switch_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Value Function
        self.value_function = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
        
        # Termination head to decouple from value
        self.termination_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Goal projection layer
        self.manager_goal_projection = nn.Linear(64, 256)
        
        # IDS summary projection
        self.ids_projection = nn.Sequential(
            nn.Linear(3, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256)
        )

        self.opt = torch.optim.AdamW(
            self.parameters(),
            lr=config.opt_lr,
            weight_decay=config.opt_l2,
        )
        self.to(config.device)
        
        # State tracking
        self.memory_state = None
        self.lstm_hidden = None
        self.lstm_cell = None
        self.force_continue = False
        self.current_subgoals = None  # Track current subgoals (one-hot)
        self.current_subgoal_indices = None
        self.subgoal_steps_remaining = None
        self.subgoal_horizon = getattr(config, "subgoal_horizon", 4)
        self.batch_ind = None
    
    def _init_lstm_state(self, batch_size):
        """Ensure LSTM hidden/cell states match current batch size."""
        self.lstm_hidden = torch.zeros(1, batch_size, self.hidden_size, device=config.device)
        self.lstm_cell = torch.zeros(1, batch_size, self.hidden_size, device=config.device)
    
    def _init_subgoals(self, batch_size):
        self.current_subgoals = torch.zeros(batch_size, self.num_subgoals, device=config.device)
        self.current_subgoal_indices = torch.zeros(batch_size, dtype=torch.long, device=config.device)
        self.subgoal_steps_remaining = torch.zeros(batch_size, dtype=torch.long, device=config.device)

    @staticmethod
    def prepare_batch(s_batch):
        # Identical to NASimNetGNN_MAct's implementation
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
        device = batch.x.device
        num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else None
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

        detection_slice = slice(
            1 + detection_level_idx,
            1 + detection_multiplier_idx + 1
        )

        host_detection = batch.x[host_mask, detection_slice]
        host_batches = batch_ind[host_mask]

        detection_sum = scatter(
            host_detection, host_batches, dim=0, dim_size=num_graphs
        )

        counts = scatter(
            torch.ones(host_batches.shape[0], dtype=torch.float32, device=device),
            host_batches,
            dim=0,
            dim_size=num_graphs
        ).unsqueeze(1).clamp_min(1.0)

        return detection_sum / counts

    def reset_state(self, batch_mask=None):
        if batch_mask is None:
            self.memory_state = None
            self.lstm_hidden = None
            self.lstm_cell = None
            self.current_subgoals = None
            self.current_subgoal_indices = None
            self.subgoal_steps_remaining = None
            self.batch_ind = None
        else:
            # Vectorized reset for finished episodes
            if not isinstance(batch_mask, torch.Tensor):
                batch_mask = torch.tensor(batch_mask, dtype=torch.bool, device=config.device)
            # Safety: ensure mask aligns with current batch before zeroing node-level memory
            if batch_mask.numel() <= 0:
                return
            
            # Reset memory_state for finished episodes
            if self.memory_state is not None and self.batch_ind is not None:
                if batch_mask.numel() > self.batch_ind.max():
                    reset_mask = batch_mask[self.batch_ind]  # Expand to node-level
                    self.memory_state[reset_mask] = 0
            
            # Reset LSTM states for finished episodes
            if self.lstm_hidden is not None:
                self.lstm_hidden[:, batch_mask, :] = 0
            if self.lstm_cell is not None:
                self.lstm_cell[:, batch_mask, :] = 0
            
            # Reset subgoals for finished episodes
            if self.current_subgoals is not None:
                self.current_subgoals[batch_mask] = 0
            if self.current_subgoal_indices is not None:
                self.current_subgoal_indices[batch_mask] = 0
            if self.subgoal_steps_remaining is not None:
                self.subgoal_steps_remaining[batch_mask] = 0

    def forward(self, s_batch, only_v=False, force_action=None, reset_hidden=False):
        data, data_lens, batch, batch_ind, node_index, pos_index = self.prepare_batch(s_batch)
        x = batch.x
        detection_summary = self._aggregate_ids_features(batch, batch_ind)
        track_recurrence = force_action is not None
        
        forced_action = None
        forced_terminate = None
        forced_subgoal = None
        if force_action is not None:
            if isinstance(force_action, (tuple, list)):
                forced_action = force_action[0]
                forced_terminate = force_action[1]
                if len(force_action) > 2:
                    forced_subgoal = force_action[2]
            else:
                forced_action = force_action

        # Save current state if we're in replay mode
        if reset_hidden:
            def _clone_state(t):
                return t.clone().detach() if t is not None else None
            saved_lstm_hidden = _clone_state(self.lstm_hidden)
            saved_lstm_cell = _clone_state(self.lstm_cell)
            saved_memory_state = _clone_state(self.memory_state)
            saved_subgoals = _clone_state(self.current_subgoals)
            saved_subgoal_indices = _clone_state(self.current_subgoal_indices)
            saved_subgoal_steps = _clone_state(self.subgoal_steps_remaining)
            saved_batch_ind = _clone_state(self.batch_ind)
        
        # Positional encoding
        pos_enc = positional_encoding(pos_index, config.pos_enc_dim)
        x = torch.cat([x, pos_enc], dim=1)
        
        # Enhanced Manager processing with 2 GAT layers
        x_gat = self.manager_gat1(x, batch.edge_index)
        x_gat = self.manager_gat2(x_gat, batch.edge_index)
        x_pooled = global_mean_pool(x_gat, batch_ind)
        
        # Handle hidden state reset for replay
        if reset_hidden:
            self.lstm_hidden = None
            self.lstm_cell = None
            self.memory_state = None
            self.current_subgoals = None
            self.current_subgoal_indices = None
            self.subgoal_steps_remaining = None
            self.batch_ind = None
        
        # Initialize LSTM state if needed
        batch_size = len(data_lens)
        if self.lstm_hidden is None or self.lstm_hidden.size(1) != batch_size:
            self._init_lstm_state(batch_size)
        
        # LSTM processing
        lstm_input = x_pooled.unsqueeze(1)
        lstm_out, (self.lstm_hidden, self.lstm_cell) = self.manager_lstm(
            lstm_input, 
            (self.lstm_hidden, self.lstm_cell)
        )
        if not track_recurrence:
            self.lstm_hidden = self.lstm_hidden.detach()
            self.lstm_cell = self.lstm_cell.detach()
        abstract_goal = lstm_out.squeeze(1)
        
        # Project and expand manager goal
        projected_goal = self.manager_goal_projection(abstract_goal)
        goal_per_node = projected_goal[batch_ind]
        
        # Initialize memory if needed - must match current batch size
        if self.memory_state is None or self.memory_state.size(0) != x.size(0):
            self.memory_state = torch.zeros(x.size(0), 256, device=config.device)
        
        # Store current batch_ind for proper reset_state
        self.batch_ind = batch_ind
        
        # Fixed Graph Temporal Memory with proper recurrence
        context = self.gtm(x, goal_per_node, self.memory_state, batch.edge_index)
        if track_recurrence:
            self.memory_state = context
        else:
            self.memory_state = context.detach()
        ids_bias = self.ids_projection(detection_summary)
        context = context + ids_bias[batch_ind]
        
        # Value function
        context_pooled = global_mean_pool(context, batch_ind)
        value = self.value_function(context_pooled)
        
        # Subgoal prediction and usage
        if (
            self.current_subgoals is None
            or self.current_subgoals.size(0) != batch_size
            or self.current_subgoal_indices is None
            or self.subgoal_steps_remaining is None
        ):
            self._init_subgoals(batch_size)
        
        subgoal_logits = self.subgoal_predictor(context_pooled)
        subgoal_probs = torch.softmax(subgoal_logits, dim=-1)
        switch_probs = self.subgoal_switch_head(context_pooled).clamp(1e-6, 1 - 1e-6)

        need_new_subgoal = self.subgoal_steps_remaining <= 0
        prev_indices = self.current_subgoal_indices.clone()
        prev_steps = self.subgoal_steps_remaining.clone()
        selected_subgoals = prev_indices.clone()

        sample_mask = None
        if forced_subgoal is not None:
            selected_subgoals = forced_subgoal.to(config.device).long()
            switch_mask = torch.logical_or(need_new_subgoal, selected_subgoals != prev_indices)
            policy_switch_mask = torch.logical_and(switch_mask, ~need_new_subgoal)
        else:
            if self.training:
                switch_dist = torch.distributions.Bernoulli(probs=switch_probs)
                sampled_switch = switch_dist.sample().bool().flatten()
            else:
                sampled_switch = (switch_probs > 0.5).bool().flatten()
            switch_mask = torch.logical_or(need_new_subgoal, sampled_switch)
            policy_switch_mask = torch.logical_and(switch_mask, ~need_new_subgoal)
            sample_mask = torch.logical_or(policy_switch_mask, need_new_subgoal)
            if sample_mask.any():
                if self.training:
                    subgoal_dist = torch.distributions.Categorical(subgoal_probs[sample_mask])
                    new_samples = subgoal_dist.sample()
                else:
                    new_samples = torch.argmax(subgoal_probs[sample_mask], dim=-1)
                selected_subgoals[sample_mask] = new_samples

        current_subgoal_vec = torch.eye(self.num_subgoals, device=config.device)[selected_subgoals]
        self.current_subgoals = current_subgoal_vec.detach().clone()
        self.current_subgoal_indices = selected_subgoals.detach().clone()
        subgoal_per_node = self.current_subgoals[batch_ind]

        subgoal_prob_selected = torch.gather(
            subgoal_probs, 1, selected_subgoals.unsqueeze(1)
        ).clamp_min(1e-9)

        # Keep hierarchical credit assignment but avoid extremely small joint probs that explode PPO ratios
        subgoal_factor = torch.ones_like(subgoal_prob_selected)
        continue_mask = ~switch_mask
        subgoal_factor[policy_switch_mask] = (
            switch_probs[policy_switch_mask] * subgoal_prob_selected[policy_switch_mask]
        ).clamp_min(1e-3)
        subgoal_factor[continue_mask] = (1 - switch_probs[continue_mask]).clamp_min(1e-3)
        subgoal_factor[need_new_subgoal] = subgoal_prob_selected[need_new_subgoal].clamp_min(1e-3)
        decayed = torch.clamp_min(prev_steps - 1, 0)
        reset_steps = torch.full_like(prev_steps, self.subgoal_horizon)
        next_steps = torch.where(switch_mask, reset_steps, decayed)
        self.subgoal_steps_remaining = next_steps.detach().clone()
        
        # Restore original state if we were in replay mode
        if reset_hidden:
            self.lstm_hidden = saved_lstm_hidden
            self.lstm_cell = saved_lstm_cell
            self.memory_state = saved_memory_state
            self.current_subgoals = saved_subgoals
            self.current_subgoal_indices = saved_subgoal_indices
            self.subgoal_steps_remaining = saved_subgoal_steps
            self.batch_ind = saved_batch_ind
        
        if only_v:
            return value
        
        # Enhanced action selection with subgoal conditioning
        worker_input = torch.cat([context, subgoal_per_node], dim=1)
        action_logits = self.worker(worker_input)
        
        # Identify graphs with no valid (non-subnet) nodes to avoid softmax over -inf
        # Mask subnet nodes
        subnet_mask = batch.x[:, 0] == 1
        valid_counts = torch.bincount(batch_ind, (~subnet_mask).long(), minlength=batch_size)
        invalid_graphs = valid_counts == 0
        invalid_node_mask = invalid_graphs[batch_ind]
        action_logits[subnet_mask & ~invalid_node_mask] = -float("inf")
        action_logits[invalid_node_mask] = 0.0  # safe fallback for all-subnet graphs
        
        # Sampling (identical to NASimNetGNN_MAct)
        action_probs = torch_geometric.utils.softmax(
            action_logits.flatten(), 
            torch.repeat_interleave(batch_ind, config.action_dim))
        data_lens_a = [n_nodes * config.action_dim for n_nodes in data_lens]
        
        if forced_action is not None:
            action_selected = forced_action
        else:
            action_selected = segmented_sample(action_probs, data_lens_a)
        
        cum_lens = np.cumsum([0] + data_lens_a[:-1])
        a_index = torch.tensor(cum_lens, device=config.device) + action_selected
        a_prob = action_probs[a_index].view(-1, 1)
        
        # Map to environment actions
        n_index = a_index.cpu().numpy() // config.action_dim
        targets = node_index[n_index].reshape(-1, 2)
        a_id = (action_selected % config.action_dim).cpu().numpy()
        
        # Improved termination using learned termination head with PPO-aligned probs
        terminate = None
        termination_probs = self.termination_head(context_pooled).clamp(1e-6, 1 - 1e-6)
        # Bias termination upward: base bias + stronger push when value is low, then sharpen via temperature
        base_term_bias = getattr(config, "termination_base_bias", 0.05)
        value_bias_scale = getattr(config, "termination_value_bias", 0.5)
        temp = getattr(config, "termination_temp", 0.7)
        term_bias = torch.sigmoid(-value.detach()) * value_bias_scale
        termination_probs = (termination_probs + base_term_bias + term_bias).clamp(1e-5, 1 - 1e-5)
        term_logits = torch.logit(termination_probs, eps=1e-5) / max(temp, 1e-3)
        termination_probs = torch.sigmoid(term_logits).clamp(1e-5, 1 - 1e-5)
        if forced_terminate is not None:
            terminate = forced_terminate.bool().to(config.device).flatten()
        else:
            if self.training:
                terminate = torch.bernoulli(termination_probs).bool().flatten()
            else:
                terminate = (termination_probs > 0.5).flatten()
        if invalid_graphs.any():
            # Hard-stop graphs that have no valid nodes; avoid training on forced choices
            termination_probs = termination_probs.clone()
            termination_probs[invalid_graphs] = 1.0
            terminate = terminate.clone()
            terminate[invalid_graphs] = True
        
        if self.force_continue:
            # Keep training the termination head by using its probabilities, but always continue in the env
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
            a_prob[cont_mask]
            * continue_probs[cont_mask]
            * subgoal_factor[cont_mask]
        )
        total_prob[terminate] = termination_probs[terminate]
        if invalid_graphs.any():
            # Forced terminations should not produce unstable training signals
            total_prob[invalid_graphs] = 1.0
        if not torch.isfinite(total_prob).all() or (total_prob <= 0).any():
            raise ValueError("Invalid probability encountered in FeudalGTM forward pass")
        
        return (
            list(zip(targets, a_id)),
            value,
            total_prob,
            (action_selected.detach(), terminate.detach(), selected_subgoals.detach()),
        )

    def set_force_continue(self, force):
        self.force_continue = force
    
    def clone_state(self, other):
        def clone_tensor(t):
            return t.clone().detach() if t is not None else None
        self.memory_state = clone_tensor(other.memory_state)
        self.lstm_hidden = clone_tensor(other.lstm_hidden)
        self.lstm_cell = clone_tensor(other.lstm_cell)
        self.current_subgoals = clone_tensor(other.current_subgoals)
        self.current_subgoal_indices = clone_tensor(other.current_subgoal_indices)
        self.subgoal_steps_remaining = clone_tensor(other.subgoal_steps_remaining)
        self.batch_ind = clone_tensor(other.batch_ind)

    def update(self, trace, target_net=None, hidden_s0=None):
        """PPO update method for FeudalGTM network"""
        sx, a, a_cnt, r, sx_, d = zip(*trace)
        
        # Prepare states and next states
        s = np.empty((config.ppo_t, config.batch), dtype=object)
        s[:, :] = sx
        s_ = np.empty((config.ppo_t, config.batch), dtype=object)
        s_[:, :] = sx_
        
        # Convert rewards and dones to arrays
        r = np.vstack(r)
        d = np.vstack(d)
        
        # Use current network as target if none provided
        if target_net is None:
            target_net = self
        
        if hidden_s0 is None:
            raise ValueError("FeudalGTM.update requires hidden_s0 for recurrent PPO replay")

        # Rebuild recurrent state on the target net to bootstrap v_ consistently
        with torch.no_grad():
            target_net.clone_state(hidden_s0)
            for t in range(config.ppo_t):
                target_net(s[t], force_action=a[t])
                target_net.reset_state(d[t])
            v_ = target_net(s_[-1], only_v=True).detach().flatten()
        
        # Compute value target
        v_target = compute_v_target(r, v_, d, config.gamma, config.ppo_t, 
                                   config.batch, config.use_a_t)
        
        # Flatten value targets (states remain sequences for recurrent PPO)
        v_target = v_target.flatten()
        a_cnt = torch.tensor(np.concatenate(a_cnt), dtype=torch.bool, device=self.device)
        
        # Log metrics
        wandb.log({
            "v_value": v_.mean().item(),
            "v_target": v_target.mean().item()
        }, commit=False)
        
        # Perform PPO update treating FeudalGTM as a recurrent policy
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
            False
        )

    def _update(self, loss):
        """Perform a single gradient update step for FeudalGTM"""
        self.opt.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), config.opt_max_norm
        )
        
        # Update weights
        self.opt.step()
        
        # Log gradient norm
        wandb.log({"grad_norm": grad_norm.item()}, commit=False)
        return grad_norm
