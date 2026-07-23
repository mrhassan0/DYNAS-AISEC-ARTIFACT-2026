import os

import torch, numpy as np
from torch.nn import *
from config import config

class Net(Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device(config.device)
        self.lr = config.opt_lr
        self.alpha_h = config.alpha_h

    def save(self, file='model.pt', training_state=None):
        """Save model weights and optional trainer state.

        ``training_state`` deliberately lives outside the module state dict:
        old checkpoints remain loadable, while ``main.py`` can attach the
        optimizer, target-network, RNG, schedule, and progress metadata needed
        for a step-aware restart.
        """
        payload = {'state_dict': self.state_dict()}
        if training_state is not None:
            payload['training_state'] = training_state
        # A power loss during torch.save must not destroy the previous usable
        # checkpoint. Write beside it and atomically replace only after the
        # complete payload has reached the filesystem interface.
        tmp_file = f'{file}.tmp'
        torch.save(payload, tmp_file)
        os.replace(tmp_file, file)

    def load(self, file='model.pt'):
        loaded = torch.load(file, map_location=self.device)

        if isinstance(loaded, dict) and 'state_dict' in loaded:
            self.load_state_dict(loaded['state_dict'])
            return loaded.get('training_state')

        # Backward compatibility with earlier weights-only checkpoints.
        self.load_state_dict(loaded)
        return None

    def copy_weights(self, other, rho):
        params_other = list(other.parameters())
        params_self  = list(self.parameters())

        for i in range( len(params_other) ):
            val_self  = params_self[i].data
            val_other = params_other[i].data
            val_new   = rho * val_other + (1-rho) * val_self

            params_self[i].data.copy_(val_new)

    def set_lr(self, lr):
        self.lr = lr

        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def set_alpha_h(self, alpha_h):
        self.alpha_h = alpha_h

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def reset_state(self, batch_mask=None):
        pass

    def clone_state(self, other):
        pass
