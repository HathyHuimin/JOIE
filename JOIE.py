import numpy as np
import copy, os, json
import torch, random
from torch.distributions import Categorical, Bernoulli
from gym import spaces
from math import exp
import pydash as ps
import torch.nn as nn
import torch.nn.functional as F
from convlab.agent2.net.base import Net

from convlab.agent2 import memory
from convlab.agent2 import net
from convlab.agent2.algorithm import policy_util
from convlab.agent2.net import net_util
from convlab.lib import logger, util


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class BaseNet(nn.Module):
    def __init__(self, state_dim, hid_layers=[100], gate=F.relu):
        super(BaseNet, self).__init__()
        dims = [state_dim] + hid_layers
        self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class Argmax(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None):
        new_logits = torch.full_like(logits, -1e8, dtype=torch.float)
        new_logits[logits == logits.max(dim=-1, keepdim=True)[0]] = 1.0
        logits = new_logits
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)


class baseAgent:
    def __init__(self, agent, global_nets=None):
        self.agent = agent
        self.algorithm_spec = agent.agent_spec['algorithm']
        self.name = self.algorithm_spec['name']
        self.net_spec = agent.agent_spec.get('net', None)
        self.memory_spec = agent.agent_spec['memory']
        self.body = self.agent.body
        self.device = 'cuda' if self.net_spec["gpu"] else "cpu"

        self.state_dim = self.body.state_dim
        self.option_num = 9
        self.action_num = 364
        self.option_space = spaces.Discrete(self.option_num)
        self.action_space = spaces.Discrete(self.action_num)

    def init_algorithm_params(self):
        '''Initialize other algorithm parameters.'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='Argmax',
            action_policy='epsilon_greedy',
            explore_var_spec=None,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            'explore_var_spec',
            'gamma',
            'training_batch_iter',
            'training_iter',
            'training_frequency',
            'training_start_step',
        ])

        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
        self.body.explore_var = self.explore_var_scheduler.start_val
        self.actor_batch = \
            {'state': [], 'option': [], 'logp': [],
             'entropy': [], 'done': [], 'reward': [], 'next_state': []}

    def sample_action(self, pdparam):
        action_pd = Argmax(logits=pdparam)
        action = action_pd.sample()
        return action

    def act(self, state):
        body = self.body
        action = self.action_policy(state, self, body, [])
        if isinstance(action, (list, int)):
            return action
        return int(action.cpu().squeeze())  # squeeze to handle scalar

    def torch_state(self, state=None, option=None):
        if torch.is_tensor(state):
            if len(list(state.size())) != 2:
                state = state.unsqueeze(dim=0)
            if state.size()[1] == self.state_dim + self.option_num:
                if option is None:
                    return state
                else:
                    idx = torch.arange(state.size()[0]).long()
                    state = state[idx, :torch.tensor(self.state_dim).long()]

        assert state is not None
        if not torch.is_tensor(state):
            state = torch.from_numpy(state.astype(np.float32))
        if len(list(state.size())) != 2:
            state = state.unsqueeze(dim=0)
        state = state.to(self.device)

        a1v = np.zeros((state.size()[0], self.option_num))
        a1v = torch.from_numpy(a1v.astype(np.float32))

        if option is None:
            return torch.cat([state, a1v], 1).to(self.device)

        assert option is not None
        if not torch.is_tensor(option):
            option = torch.tensor(option)
        idx = torch.arange(state.size()[0]).long()
        a1v = np.zeros((state.size()[0], self.option_num))
        a1v[idx, option.long()] = 1
        a1v = torch.from_numpy(a1v.astype(np.float32))
        # state_a1 = torch.cat((state, a1v), 1).to(self.device)
        return torch.cat([state, a1v], 1).to(self.device)

    def update(self):
        '''Updates self.target_net and the explore variables'''
        if util.frame_mod(self.body.env.clock.frame, self.net.update_frequency, self.body.env.num_envs):
            net_util.copy(self.net, self.target_net)

        self.body.explore_var = self.explore_var_scheduler.update(self, self.body.env.clock)
        return self.body.explore_var

    def post_init_nets(self):
        assert hasattr(self, 'net_names')
        if util.in_eval_lab_modes():
            self.load()

    def save(self, ckpt=None):
        '''Save net models for algorithm given the required property self.net_names'''
        if not hasattr(self, 'net_names'):
            logger.info('No net declared in self.net_names in init_nets(); no models to save.')
        else:
            net_util.save_algorithm(self, ckpt=ckpt)

    def load(self):
        '''Load net models for algorithm given the required property self.net_names'''
        if not hasattr(self, 'net_names'):
            logger.info('No net declared in self.net_names in init_nets(); no models to load.')
        else:
            net_util.load_algorithm(self)
        # set decayable variables to final values
        for k, v in vars(self).items():
            if k.endswith('_scheduler'):
                var_name = k.replace('_scheduler', '')
                if hasattr(v, 'end_val'):
                    setattr(self.body, var_name, v.end_val)


class WDQN_Co3_Jo(baseAgent):
    def __init__(self, agent, global_nets=None):
        self.agent = agent
        self.algorithm_spec = agent.agent_spec['algorithm']
        self.name = self.algorithm_spec['name']
        self.net_spec = agent.agent_spec.get('net', None)
        self.memory_spec = agent.agent_spec['memory']

        self.body = self.agent.body
        self.device = 'cuda' if self.net_spec["gpu"] else "cpu"
        self.state_dim = self.body.state_dim

        self.a1_len = 9
        self.a2_len = 13
        self.a3_len = 28  # len(self.actionSpace.vocab_sub)
        self.a1_space = spaces.Discrete(self.a1_len)
        self.a2_space = spaces.Discrete(self.a2_len)
        self.a3_space = spaces.Discrete(self.a3_len)

        self.body = self.agent.body
        self.init_algorithm_params()
        self.init_nets(global_nets)
        self.device = 'cuda' if self.net_spec["gpu"] else "cpu"
        logger.info(util.self_desc(self))

        util.set_attr(self, self.algorithm_spec, ['warmup_epi', ])
        MemoryClass = getattr(memory, self.memory_spec['warmup_name'])
        self.body.warmup_memory = MemoryClass(self.memory_spec, self.body)

        self.epsln = self.body.explore_var
        self.termination_reg = 0.01
        self.entropy_reg = 0.01

    def init_nets(self, global_nets=None):
        self.in_dim = self.body.state_dim

        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, self.in_dim + self.a1_len + self.a2_len, [self.a1_len,self.a2_len,self.a3_len])
        self.target_net = NetClass(self.net_spec, self.in_dim + self.a1_len + self.a2_len, [self.a1_len,self.a2_len,self.a3_len])
        self.target_net.load_state_dict(self.net.state_dict())

        self.net_names = ['net', 'target_net']
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec)

        net_util.set_global_nets(self, global_nets)
        self.post_init_nets()

    def loss1(self, batch):

        states = batch['states']
        a1 = batch['a1']
        dones = batch['dones']
        rewards = batch['rewards']
        next_states = batch['next_states']

        states = self.torch_state(state=states)
        next_states = self.torch_state(state=next_states)

        # ============net1==============
        Q1 = self.net(states)[0]
        with torch.no_grad():
            Q_target1 = self.target_net(next_states)[0]
        Q_sa1 = Q1.gather(-1, a1.long().unsqueeze(-1)).squeeze(-1)
        argmaxa1_Q1_target = Q_target1.argmax(dim=-1, keepdim=True)
        Q_target1_s_max_a1 = Q_target1.gather(-1, argmaxa1_Q1_target).squeeze(-1)
        Q1_label = rewards + self.gamma * (1 - dones) * Q_target1_s_max_a1
        loss1 = self.net.loss_fn(Q_sa1, Q1_label)

        return loss1

    def loss2(self, batch):

        # ============net2==============

        states = batch['states']
        a1 = batch['a1']
        a2 = batch['a2']
        dones = batch['dones']
        rewards = batch['rewards']
        next_states = batch['next_states']
        next_a1 = batch['next_a1']

        state_a1 = self.torch_state(state=states, a1=a1)
        next_state_a1 = self.torch_state(state=next_states, a1=next_a1)

        Q2 = self.net(state_a1)[1]
        with torch.no_grad():
            Q_target2 = self.target_net(next_state_a1)[1]
        Q_sa2 = Q2.gather(-1, a2.long().unsqueeze(-1)).squeeze(-1)
        argmaxa2_Q2_target = Q_target2.argmax(dim=-1, keepdim=True)
        Q_target2_s_max_a2 = Q_target2.gather(-1, argmaxa2_Q2_target).squeeze(-1)
        Q2_labe2 = rewards + self.gamma * (1 - dones) * Q_target2_s_max_a2
        loss2 = self.net.loss_fn(Q_sa2, Q2_labe2)

        return loss2

    def loss3(self, batch):

        # ============net2==============

        states = batch['states']
        a1 = batch['a1']
        a2 = batch['a2']
        a3 = batch['a3']
        dones = batch['dones']
        rewards = batch['rewards']
        next_states = batch['next_states']
        next_a1 = batch['next_a1']
        next_a2 = batch['next_a2']

        state_a1_a2 = self.torch_state(state=states, a1=a1, a2=a2)
        next_state_a1_a2 = self.torch_state(state=next_states, a1=next_a1, a2=next_a2)

        Q3 = self.net(state_a1_a2)[2]
        with torch.no_grad():
            Q_target3 = self.target_net(next_state_a1_a2)[2]
        Q_sa3 = Q3.gather(-1, a3.long().unsqueeze(-1)).squeeze(-1)
        argmaxa3_Q3_target = Q_target3.argmax(dim=-1, keepdim=True)
        Q_target3_s_max_a3 = Q_target3.gather(-1, argmaxa3_Q3_target).squeeze(-1)
        Q3_labe3 = rewards + self.gamma * (1 - dones) * Q_target3_s_max_a3
        loss3 = self.net.loss_fn(Q_sa3, Q3_labe3)

        return loss3

    def co_loss(self, batch):
        states = batch['states']
        a1 = batch['a1']
        a2 = batch['a2']
        a3 = batch['a3']
        dones = batch['dones']
        rewards = batch['rewards']
        next_states = batch['next_states']
        next_a1 = batch['next_a1']
        next_a2 = batch['next_a2']
        state_a1_a2 = self.torch_state(state=states, a1=a1, a2=a2)
        next_state_a1_a2 = self.torch_state(state=next_states, a1=next_a1, a2=next_a2)

        state_a1 = self.torch_state(state=states, a1=a1)
        next_state_a1 = self.torch_state(state=next_states, a1=next_a1)

        states = self.torch_state(state=states)
        next_states = self.torch_state(state=next_states)


        # ============net1==============

        Q1 = self.net(states)[0]
        with torch.no_grad():
            Q_target1 = self.target_net(next_states)[0]
        Q_sa1 = Q1.gather(-1, a1.long().unsqueeze(-1)).squeeze(-1)
        argmaxa1_Q1_target = Q_target1.argmax(dim=-1, keepdim=True)
        Q_target1_s_max_a1 = Q_target1.gather(-1, argmaxa1_Q1_target).squeeze(-1)
        Q1_label = rewards + self.gamma * (1 - dones) * Q_target1_s_max_a1
        loss1 = self.net.loss_fn(Q_sa1, Q1_label)

        # ============net2==============

        Q2 = self.net(state_a1)[1]
        with torch.no_grad():
            Q_target2 = self.target_net(next_state_a1)[1]
        Q_sa2 = Q2.gather(-1, a2.long().unsqueeze(-1)).squeeze(-1)
        argmaxa2_Q2_target = Q_target2.argmax(dim=-1, keepdim=True)
        Q_target2_s_max_a2 = Q_target2.gather(-1, argmaxa2_Q2_target).squeeze(-1)
        Q2_labe2 = rewards + self.gamma * (1 - dones) * Q_target2_s_max_a2
        loss2 = self.net.loss_fn(Q_sa2, Q2_labe2)

        # ============net3==============

        Q3 = self.net(state_a1_a2)[2]
        with torch.no_grad():
            Q_target3 = self.target_net(next_state_a1_a2)[2]
        Q_sa3 = Q3.gather(-1, a3.long().unsqueeze(-1)).squeeze(-1)
        argmaxa3_Q3_target = Q_target3.argmax(dim=-1, keepdim=True)
        Q_target3_s_max_a3 = Q_target3.gather(-1, argmaxa3_Q3_target).squeeze(-1)
        Q3_labe3 = rewards + self.gamma * (1 - dones) * Q_target3_s_max_a3
        loss3 = self.net.loss_fn(Q_sa3, Q3_labe3)

        return loss1, loss2, loss3

    def train(self):
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock

        if self.to_train == 1:
            loss1, loss2, loss3 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
            for i in range(self.training_iter):
                batches = []
                batches2 = []
                batches3 = []
                if self.body.warmup_memory.size >= self.body.warmup_memory.batch_size:  # x 16
                    warmup_batch = self.body.warmup_memory.sample()
                    warmup_batch = util.to_torch_batch(warmup_batch, self.device, self.body.warmup_memory.is_episodic)
                    batches.append(warmup_batch)

                    warmup_batch2 = self.body.warmup_memory.sample()
                    warmup_batch2 = util.to_torch_batch(warmup_batch2, self.device, self.body.warmup_memory.is_episodic)
                    batches2.append(warmup_batch2)

                    warmup_batch3 = self.body.warmup_memory.sample()
                    warmup_batch3 = util.to_torch_batch(warmup_batch3, self.device, self.body.warmup_memory.is_episodic)
                    batches3.append(warmup_batch3)
                if self.body.memory.size >= self.body.memory.batch_size and self.body.env.clock.epi > self.warmup_epi:
                    batch = self.body.memory.sample()
                    batch = util.to_torch_batch(batch, self.device, self.body.warmup_memory.is_episodic)
                    batches.append(batch)

                    batch2 = self.body.memory.sample()
                    batch2 = util.to_torch_batch(batch2, self.device, self.body.warmup_memory.is_episodic)
                    batches2.append(batch2)

                    batch3 = self.body.memory.sample()
                    batch3 = util.to_torch_batch(batch3, self.device, self.body.warmup_memory.is_episodic)
                    batches3.append(batch3)

                clock.set_batch_size(sum(len(batch) for batch in batches))  # 7
                for batch, batch2, batch3 in zip(batches, batches2, batches3):
                    for _ in range(self.training_batch_iter):  # 1
                        #  loss1, loss2, loss3 = self.co_loss(batch) # option: share the state
                        loss1 = self.loss1(batch)
                        loss2 = self.loss2(batch2)
                        loss3 = self.loss3(batch3)
                        loss = loss1 + loss2 + loss3
                        self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock)
            logger.info(
                f'Trained {self.name} at epi: {clock.epi}, '
                f'l1:{loss1:g},  '
                f'l2:{loss2:g},  '
                f'l3:{loss3:g},  '
                f'all:{loss1 + loss2 + loss3:g}')

            self.to_train = 0
            return (loss1 + loss2 + loss3).item()
        else:
            return np.nan

    def calc_pdparam(self, state, net=None, eval = None):
        epsilon = self.body.explore_var
        a1 = self.a1_space.sample() if random.random() < epsilon else \
            int(self.sample_action(self.net(self.torch_state(state=state))[0]).squeeze()) #.argmax(dim=-1).item()

        epsilon = self.body.explore_var
        a2 = self.a2_space.sample() if random.random() < epsilon else \
            int(self.sample_action(self.net(self.torch_state(state=state, a1=a1))[1]).squeeze())#.argmax(dim=-1).item()

        epsilon = self.body.explore_var
        a3 = self.a3_space.sample() if random.random() < epsilon else \
            int(self.sample_action(self.net(self.torch_state(state=state, a1=a1, a2 = a2))[2]).squeeze())#.argmax(dim=-1).item()

        return [a1, a2, a3]

    def torch_state(self, state=None, a1=None, a2=None):
        if torch.is_tensor(state):
            if len(list(state.size())) != 2:
                state = state.unsqueeze(dim=0)
            if state.size()[1] == self.in_dim + self.a1_len + self.a2_len:
                if a1 is None and a2 is None:
                    return state
                else:
                    l = torch.tensor(self.in_dim).long()
                    idx = torch.arange(state.size()[0]).long()
                    state = state[idx, :l]

        assert state is not None
        if not torch.is_tensor(state):
            state = torch.from_numpy(state.astype(np.float32))
        if len(list(state.size())) != 2:
            state = state.unsqueeze(dim=0)
        state = state.to(self.device)

        a1v = np.zeros((state.size()[0], self.a1_len))
        a1v = torch.from_numpy(a1v.astype(np.float32))
        a2v = np.zeros((state.size()[0], self.a2_len))
        a2v = torch.from_numpy(a2v.astype(np.float32))

        if a1 is None and a2 is None:
            return torch.cat([state, a1v, a2v], 1).to(self.device)

        assert a1 is not None
        if not torch.is_tensor(a1):
            a1 = torch.tensor(a1)
        idx = torch.arange(state.size()[0]).long()
        a1v = np.zeros((state.size()[0], self.a1_len))
        a1v[idx, a1.long()] = 1
        a1v = torch.from_numpy(a1v.astype(np.float32))
        # state_a1 = torch.cat((state, a1v), 1).to(self.device)
        if a2 is None:
            return torch.cat([state, a1v, a2v], 1).to(self.device)

        else:
            if not torch.is_tensor(a2):
                a2 = torch.tensor(a2)
            a2v = np.zeros((state.size()[0], self.a2_len))
            a2v[idx, a2.long()] = 1
            a2v = torch.from_numpy(a2v.astype(np.float32))
            # state_a1_a2 = torch.cat((state_a1, a2v), 1).to(self.device)
            return torch.cat([state,a1v,a2v], 1).to(self.device)