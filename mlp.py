# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pydash as ps
import torch
import torch.nn as nn

from convlab.agent.net import net_util
from convlab.agent.net.base import Net
from convlab.lib import math_util, util


class MLPNetJo(Net, nn.Module):

    def __init__(self, net_spec, in_dim, out_dim):
        nn.Module.__init__(self)
        super().__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            out_layer_activation=None,
            init_fn=None,
            clip_grad_val=None,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_scheduler_spec=None,
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'shared',
            'hid_layers',
            'hid_layers_activation',
            'out_layer_activation',
            'init_fn',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])

        dims = [self.in_dim] + self.hid_layers
        self.model = net_util.build_fc_model(dims, self.hid_layers_activation)

        if not ps.is_list(self.out_layer_activation):
            self.out_layer_activation = [self.out_layer_activation] * len(out_dim)
        assert len(self.out_layer_activation) == len(self.out_dim)
        tails = []
        for out_d, out_activ in zip(self.out_dim, self.out_layer_activation):
            tail = net_util.build_fc_model([dims[-1], out_d], out_activ)
            tails.append(tail)
        self.model_tails = nn.ModuleList(tails)

        net_util.init_layers(self, self.init_fn)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)
        self.train()

    def forward(self, x):
        '''The feedforward step'''
        x = self.model(x)
        outs = []
        for model_tail in self.model_tails:
            outs.append(model_tail(x))
        return outs


