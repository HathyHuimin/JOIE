# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import copy
import torch
import random
from queue import PriorityQueue
from convlab.agent.memory.base import Memory
from convlab.lib import logger, util
from convlab.lib.decorator import lab_api

logger = logger.get_logger(__name__)


def sample_next_states(head, max_size, ns_idx_offset, batch_idxs, states, ns_buffer):
    '''Method to sample next_states from states, with proper guard for next_state idx being out of bound'''
    # idxs for next state is state idxs with offset, modded
    ns_batch_idxs = (batch_idxs + ns_idx_offset) % max_size
    # if head < ns_idx <= head + ns_idx_offset, ns is stored in ns_buffer
    ns_batch_idxs = ns_batch_idxs % max_size
    buffer_ns_locs = np.argwhere(
        (head < ns_batch_idxs) & (ns_batch_idxs <= head + ns_idx_offset)).flatten()
    # find if there is any idxs to get from buffer
    to_replace = buffer_ns_locs.size != 0
    if to_replace:
        # extract the buffer_idxs first for replacement later
        # given head < ns_idx <= head + offset, and valid buffer idx is [0, offset)
        # get 0 < ns_idx - head <= offset, or equiv.
        # get -1 < ns_idx - head - 1 <= offset - 1, i.e.
        # get 0 <= ns_idx - head - 1 < offset, hence:
        buffer_idxs = ns_batch_idxs[buffer_ns_locs] - head - 1
        # set them to 0 first to allow sampling, then replace later with buffer
        ns_batch_idxs[buffer_ns_locs] = 0
    # guard all against overrun idxs from offset
    ns_batch_idxs = ns_batch_idxs % max_size
    next_states = util.batch_get(states, ns_batch_idxs)
    if to_replace:
        # now replace using buffer_idxs and ns_buffer
        buffer_ns = util.batch_get(ns_buffer, buffer_idxs)
        next_states[buffer_ns_locs] = buffer_ns
    return next_states


class Replay(Memory):
    '''
    Stores agent experiences and samples from them for agent training

    An experience consists of
        - state: representation of a state
        - action: action taken
        - reward: scalar value
        - next state: representation of next state (should be same as state)
        - done: 0 / 1 representing if the current state is the last in an episode

    The memory has a size of N. When capacity is reached, the oldest experience
    is deleted to make space for the lastest experience.
        - This is implemented as a circular buffer so that inserting experiences are O(1)
        - Each element of an experience is stored as a separate array of size N * element dim

    When a batch of experiences is requested, K experiences are sampled according to a random uniform distribution.

    If 'use_cer', sampling will add the latest experience.

    e.g. memory_spec
    "memory": {
        "name": "Replay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    '''

    def __init__(self, memory_spec, body):
        super().__init__(memory_spec, body)
        util.set_attr(self, self.memory_spec, [
            'batch_size',
            'max_size',
            'use_cer',
        ])
        self.is_episodic = False
        self.batch_idxs = None
        self.size = 0  # total experiences stored
        self.seen_size = 0  # total experiences seen cumulatively
        self.head = -1  # index of most recent experience
        # generic next_state buffer to store last next_states (allow for multiple for venv)
        # self.ns_idx_offset = self.body.env.num_envs if body.env.is_venv else 1
        # self.ns_buffer = deque(maxlen=self.ns_idx_offset)
        # declare what data keys to store
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'episode', 'idx']
        self.reset()

    def reset(self):
        '''Initializes the memory arrays, size and head pointer'''
        # set self.states, self.actions, ...
        for k in self.data_keys:
            setattr(self, k, [None] * self.max_size)
            # if k != 'next_states':  # reuse self.states
            #     # list add/sample is over 10x faster than np, also simpler to handle
            #     setattr(self, k, [None] * self.max_size)
        self.size = 0
        self.head = -1
        self.seen_size = 0
        # self.ns_buffer.clear()

    @lab_api
    def update(self, state, action, reward, next_state, done, episode=-1, idx=-1):
        '''Interface method to update memory'''
        if self.body.env.is_venv:
            for sarsd in zip(state, action, reward, next_state, done):
                self.add_experience(*sarsd)
        else:
            self.add_experience(state, action, reward, next_state, done, episode, idx)

    def clear_replay(self):
        for k in self.data_keys:
            setattr(self, k, [])
        self.size = 0
        self.head = -1
        self.seen_size = 0

    def add_experience(self, state, action, reward, next_state, done, episode, idx):
        '''Implementation for update() to add experience to memory, expanding the memory size if necessary'''
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        self.states[self.head] = copy.deepcopy(state)  # list
        self.actions[self.head] = copy.deepcopy(action)  # list
        self.rewards[self.head] = copy.deepcopy(reward)  # list
        self.next_states[self.head] = copy.deepcopy(next_state)  # list
        self.dones[self.head] = copy.deepcopy(done)  # list     self.size:int
        self.episode[self.head] = copy.deepcopy(episode)
        self.idx[self.head] = copy.deepcopy(idx)

        # Actually occupied size of memory
        if self.size < self.max_size:
            self.size += 1
        self.seen_size += 1
        # set to_train using memory counters head, seen_size instead of tick since clock will step by num_envs when on venv; to_train will be set to 0 after training step
        algorithm = self.body.agent.algorithm
        algorithm.to_train = algorithm.to_train or (
                    self.seen_size > algorithm.training_start_step and self.head % algorithm.training_frequency == 0)

    @lab_api
    def sample(self):
        '''
        Returns a batch of batch_size samples. Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are an array of the corresponding sampled elements
        e.g.
        batch = {
            'states'     : states,
            'actions'    : actions,
            'rewards'    : rewards,
            'next_states': next_states,
            'dones'      : dones}
        '''
        self.batch_idxs = self.sample_idxs(self.batch_size)
        batch = {}
        for k in self.data_keys:
            batch[k] = util.batch_get(getattr(self, k), self.batch_idxs)
            # if k == 'next_states':
            #     batch[k] = sample_next_states(self.head, self.max_size, self.ns_idx_offset, self.batch_idxs, self.states, self.ns_buffer)
            # else:
            #     batch[k] = util.batch_get(getattr(self, k), self.batch_idxs)
        return batch

    def sample_idxs(self, batch_size):
        '''Batch indices a sampled random uniformly'''
        batch_idxs = np.random.randint(self.size, size=batch_size)
        if self.use_cer:  # add the latest sample
            batch_idxs[-1] = self.head
        return batch_idxs


class ReplayHR3(Memory):
    def __init__(self, memory_spec, body):
        self.memory_spec = memory_spec
        self.body = body
        util.set_attr(self, self.memory_spec, [
            'batch_size',
            'max_size',
            'use_cer',
        ])
        self.is_episodic = False
        self.batch_idxs = None
        self.size = 0  # total experiences stored
        self.seen_size = 0  # total experiences seen cumulatively
        self.head = -1  # index of most recent experience
        self.data_keys = ['states', 'a1', 'a2', 'a3', 'rewards', 'next_states', 'next_a1', 'next_a2', 'dones']
        self.reset()


    def reset(self):
        for k in self.data_keys:
            setattr(self, k, [None] * self.max_size)
        self.size = 0
        self.head = -1
        self.pointer = self.head
        self._state = []
        self._a1 = []
        self._a2 = []
        self._a3 = []
        self._reward = []
        self._next_state = []
        self._done = []
        self.flag = False

    @lab_api
    def update(self, state, action, reward, next_state, done, episode=-1, idx=-1):
        '''Interface method to update memory'''

        if isinstance(action, int):
            action = self.body.agent.action_decoder.action_vocab.get_action(
                action)  # self.body.agent.action_decoder.current_action_num)
            assert len(action.keys()) > 0
            for domain_intent in action.keys():
                slot = action[domain_intent]
                domain, intent = domain_intent.split('-')
                break  # todo
            a1, a2, a3 = self.body.agent.action_decoder.action_vocab.get_index(a1=domain, a2=intent, a3=slot[0])

        else:
            assert len(action) == 3
            a1, a2, a3 = action[0], action[1], action[2]
        self.add_experience(state, a1, a2, a3, reward, next_state, done)

    def clear_replay(self):
        for k in self.data_keys:
            setattr(self, k, [])
        self.size = 0
        self.head = -1
        self.seen_size = -1

    def add_experience(self, state, a1, a2, a3, reward, next_state, done):
        '''Implementation for update() to add experience to memory, expanding the memory size if necessary'''
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        self.states[self.head] = copy.deepcopy(state)
        self.a1[self.head] = copy.deepcopy(a1)
        self.a2[self.head] = copy.deepcopy(a2)
        self.a3[self.head] = copy.deepcopy(a3)
        self.rewards[self.head] = copy.deepcopy(reward)
        self.next_states[self.head] = copy.deepcopy(next_state)
        self.dones[self.head] = copy.deepcopy(done)

        if self.seen_size > 0:
            if self.head > 0:
                self.next_a1[self.head - 1] = copy.deepcopy(a1)
                self.next_a2[self.head - 1] = copy.deepcopy(a2)
            else:
                self.next_a1[self.max_size - 1] = copy.deepcopy(a1)
                self.next_a2[self.max_size - 1] = copy.deepcopy(a2)
            if done:
                self.next_a1[self.head] = copy.deepcopy(a1)
                self.next_a2[self.head] = copy.deepcopy(a2)

        # Actually occupied size of memory
        if self.size < self.max_size:
            self.size += 1
        self.seen_size += 1

        algorithm = self.body.agent.algorithm
        algorithm.to_train = algorithm.to_train or (self.seen_size > algorithm.training_start_step
                                                    and self.head % algorithm.training_frequency == 0)

    @lab_api
    def sample(self):
        self.batch_idxs = self.sample_idxs(self.batch_size)
        batch = {}
        for k in self.data_keys:
            batch[k] = util.batch_get(getattr(self, k), self.batch_idxs)
        return batch

    def sample_idxs(self, batch_size):
        '''Batch indices a sampled random uniformly'''
        batch_idxs = np.random.randint(self.size, size=batch_size)
        while self.head in batch_idxs:
            batch_idxs = np.delete(batch_idxs, np.where(batch_idxs == self.head), axis=0)
            batch_idxs = np.hstack([batch_idxs, np.random.randint(self.size, size=1)])
        return batch_idxs
