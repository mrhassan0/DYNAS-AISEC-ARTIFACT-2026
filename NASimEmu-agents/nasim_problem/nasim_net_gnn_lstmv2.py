import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric

from net import Net
from config import config
from graph_nns import MultiMessagePassingWithGlobalNodeAndLastGRUGlobal  # reuse existing GNN if present
from .net_utils import positional_encoding, segmented_sample, compute_v_target, ppo  # assume available
import wandb


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden=None, dropout=0.0):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.dropout(self.net(x)))


class NASimNetGNN_LSTM_v2(Net):
    """
    Improved GNN+Per-node-GRU model compatible with NASimEmu training loop.
    Designed as a drop-in replacement for NASimNetGNN_LSTM.
    """

    def __init__(self):
        super().__init__()

        emb_dim = getattr(config, "emb_dim", 64)
        pos_enc_dim = getattr(config, "pos_enc_dim", 8)
        node_in = config.node_dim + pos_enc_dim

        # Embedding for node features + positional encoding
        self.embed_node = nn.Sequential(
            nn.Linear(node_in, emb_dim),
            nn.GELU(),
            nn.LayerNorm(emb_dim),
        )

        # Residual blocks to enrich representation before MP
        self.res1 = ResidualMLP(emb_dim, hidden=emb_dim*2, dropout=0.05)
        self.res2 = ResidualMLP(emb_dim, hidden=emb_dim*2, dropout=0.05)

        # Message passing module (reuse your previously tested implementation if available)
        # This module is expected to return (node_embeddings, pooled_graph_emb)
        self.gnn = MultiMessagePassingWithGlobalNodeAndLastGRUGlobal(mp_iterations=getattr(config, "mp_iterations", 2),
                                                                     hidden_dim=emb_dim)

        # Optional additional GRU memory per node (wraps GNN outputs)
        self.node_gru = nn.GRUCell(emb_dim, emb_dim)

        # Action & node heads
        self.node_select = nn.Linear(emb_dim, 1)  # logits for node selection before segmented softmax
        self.action_select = nn.Linear(emb_dim, config.action_dim)  # action logits per node

        # Value head: combine pooled graph embedding and a pooled node summary
        self.value_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 1)
        )

        # small auxiliary head for diagnostics (optional)
        self.aux = nn.Linear(emb_dim, 1)

        # optimizer
        self.opt = torch.optim.AdamW(self.parameters(), lr=config.opt_lr, weight_decay=getattr(config, "opt_l2", 0.0))
        self.to(self.device)

        # internal state for per-node memory inside GNN (if gnn keeps its own)
        self._h_nodes = None

        # hyperparams
        self.force_continue = False
        self.dropout = nn.Dropout(p=0.05)

    # -------------------------
    # Batch preparation (same contract)
    # -------------------------
    @staticmethod
    def prepare_batch(s_batch):
        node_feats, edge_index, node_index, pos_index = zip(*s_batch)

        node_feats = [torch.tensor(x, dtype=torch.float32, device=config.device) for x in node_feats]
        edge_index = [torch.tensor(x, dtype=torch.int64, device=config.device) for x in edge_index]

        data = [Data(x=node_feats[i], edge_index=edge_index[i]) for i in range(len(s_batch))]
        data_lens = [d.num_nodes for d in data]
        batch = Batch.from_data_list(data)
        batch_ind = batch.batch.to(config.device)
        node_index = np.concatenate(node_index)
        pos_index = torch.tensor(np.concatenate(pos_index), dtype=torch.long).to(config.device)

        return data, data_lens, batch, batch_ind, node_index, pos_index

    # -------------------------
    # Forward: matches old Net API
    # -------------------------
    def forward(self, s_batch, only_v=False, complete=False, force_action=None):
        """
        s_batch: list of tuples (node_feats, edge_index, node_index, pos_index)
        If only_v: return value only
        If complete: return node_probs, action_probs, value for debug
        """
        data, data_lens, batch, batch_ind, node_index, pos_index = self.prepare_batch(s_batch)
        x = batch.x  # [total_nodes, node_dim]

        # positional encoding
        pos_enc = positional_encoding(pos_index, dim=getattr(config, "pos_enc_dim", 8))
        x = torch.cat([x, pos_enc], dim=1)

        # embed
        x = self.embed_node(x)
        x = self.res1(x)
        x = self.res2(x)

        # message passing -> expects (x, pooled_graph)
        # If your GNN implementation expects edge_attr, pass None
        x, x_pooled = self.gnn(x, None, batch.edge_attr if hasattr(batch, "edge_attr") else None,
                                batch.edge_index, batch_ind, batch.num_graphs, data_lens)

        # per-node GRU update (explicit)
        # initialize _h_nodes if None or different size
        if self._h_nodes is None or self._h_nodes.size(0) != x.size(0):
            self._h_nodes = torch.zeros_like(x).to(self.device)
        # update node memories
        h_new = self.node_gru(x, self._h_nodes)
        self._h_nodes = h_new.detach()  # store for next time (detach to avoid backprop across episodes)

        # compute value using pooled graph embedding + global summary of nodes
        # pooled nodes by mean pooling (per-graph pooling already provided as x_pooled)
        # x_pooled shape should be [batch_size, emb_dim]
        # compute extra node summary: mean of top-k node embeddings per graph (approx via mean)
        # For simplicity: use x_pooled and mean of nodes aggregated by batch
        # If x_pooled already good enough, combine twice
        global_summary = x_pooled
        value_in = torch.cat([global_summary, global_summary], dim=1) if global_summary is not None else torch.cat([global_summary, global_summary], dim=1)
        value = self.value_proj(value_in)

        if only_v:
            return value

        # Node selection logits -> segmented softmax
        node_activation = self.node_select(h_new)  # shape [total_nodes, 1]
        node_activation = node_activation.flatten()

        # mask subnet nodes (same convention in repo: batch.x[:,0] == 1)
        subnet_nodes = batch.x[:, 0] == 1
        if subnet_nodes.any():
            node_activation[subnet_nodes] = float("-inf")

        node_softmax = torch_geometric.utils.softmax(node_activation, batch_ind)

        # Action logits for chosen nodes
        # sample node per graph
        node_selected = segmented_sample(node_softmax, data_lens)  # local indices per graph
        # compute index into flattened node list
        data_starts = np.concatenate(([0], np.cumsum(data_lens)[:-1]))
        data_starts = torch.tensor(data_starts, device=self.device, dtype=torch.int64)
        n_index = data_starts + torch.tensor(node_selected, device=self.device, dtype=torch.int64)
        n_index = n_index.to(self.device)

        # node embedding of selected nodes
        node_embed_selected = h_new[n_index]
        out_action = self.action_select(node_embed_selected)  # [B, action_dim]
        action_dist = torch.distributions.Categorical(logits=out_action)
        action_selected = action_dist.sample()
        a_prob = action_dist.log_prob(action_selected).exp().view(-1, 1)  # prob

        # node prob
        n_prob = node_softmax[n_index].view(-1, 1)

        # termination gate: when value <= 0 we issue -1
        env_actions = action_selected.clone().cpu().numpy()
        if not self.force_continue:
            terminate = (value.detach().flatten() <= 0.)
            env_actions[terminate.cpu().numpy()] = -1
            n_prob[terminate] = 0.5
            a_prob[terminate] = 0.5

        tot_prob = a_prob * n_prob

        # build actions list of ((subnet,host), action_id)
        targets = node_index[n_index.cpu().numpy()].reshape(-1, 2)
        actions = list(zip(targets, env_actions.tolist()))

        raw_actions = (torch.tensor(node_selected, device=self.device),
                       action_selected.detach())

        # If complete: return full distributions for debug
        if complete:
            action_softmax = torch.softmax(out_action, dim=1)
            return node_softmax, action_softmax, value, None

        return actions, value, tot_prob, raw_actions

    # -------------------------
    # PPO-style update wrapper compatible with repo
    # -------------------------
    def update(self, trace, target_net=None, hidden_s0=None):
        """
        trace: sequence of transitions shaped (T, batch)
        Reuse existing repo helper ppo(...) which expects this Net interface.
        """
        sx, a, a_cnt, r, sx_, d = zip(*trace)
        s = np.empty((config.ppo_t, config.batch), dtype=object)
        s[:] = sx
        s_ = np.empty((config.ppo_t, config.batch), dtype=object)
        s_[:] = sx_
        r = np.vstack(r)
        d = np.vstack(d)

        # target_net fallback
        if target_net is None:
            target_net = self

        v_ = target_net(s_[-1], only_v=True)
        v_ = v_.detach().flatten()
        v_target = compute_v_target(r, v_, d, config.gamma, config.ppo_t, config.batch, config.use_a_t)
        v_target = v_target.flatten()
        a_cnt = torch.tensor(np.concatenate(a_cnt), dtype=torch.bool, device=self.device)

        # log metrics to wandb if needed
        try:
            wandb.log(dict(v=v_.mean(), v_t=v_target.mean()), commit=False)
        except Exception:
            pass

        return ppo(s, a, a_cnt, d, v_target, self, config.gamma, config.alpha_v, getattr(self, "alpha_h", 0.0),
                   config.ppo_k, config.ppo_eps, config.use_a_t, config.v_range, lstm=True, hidden_s0=hidden_s0)

    # gradient step helper
    def _update(self, loss):
        self.opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), config.opt_max_norm)
        self.opt.step()
        return norm

    # API: reset per node hidden states (called between episodes)
    def reset_state(self, batch_mask=None):
        # reset stored node-level GRU states
        self._h_nodes = None
        try:
            self.gnn.reset_state(batch_mask)
        except Exception:
            pass

    def clone_state(self, other):
        # clone gnn state if exists
        try:
            self.gnn.clone_state(other.gnn)
        except Exception:
            pass
        # clone node hidden memory if shapes match
        if hasattr(other, "_h_nodes") and other._h_nodes is not None:
            self._h_nodes = other._h_nodes.clone().detach()