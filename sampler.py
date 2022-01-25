import numpy as np
import time

import torch

from copy import deepcopy

class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

    def set_policy(self, policy):
        self.policy = policy

    def sample(self):
        raise NotImplementedError

    def batch_ready(self):
        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action, _ = self.policy(self._current_observation)
        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation)

        if terminal or self._path_length >= self._max_path_length:
            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation


class MASampler(SimpleSampler):
    def __init__(self, agent_num, joint, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)
        self.agent_num = agent_num
        self.joint = joint
        self._path_length = 0
        self._path_return = np.array([0.] * self.agent_num, dtype=np.float32)
        self._last_path_return = np.array([0.] * self.agent_num, dtype=np.float32)
        self._max_path_return = np.array([-np.inf] * self.agent_num, dtype=np.float32)
        self._n_episodes = 0
        self._total_samples = 0

        self._current_observation_n = None
        self.env = None
        self.agents = None

        self.x_dot = []
        self.y_max_r_dot = []
        self.y_mean_r_dot = []
        self.y_r_dot = []

    def set_policy(self, policies):
        for agent, policy in zip(self.agents, policies):
            agent.policy = policy

    def batch_ready(self):
        enough_samples = self.agents[0].pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self, i):
        return self.agents[i].pool.random_batch(self._batch_size)

    def initialize(self, env, agents):
        self._current_observation_n = None
        self.env = env
        self.agents = agents

    def sample(self):
        if self._current_observation_n is None:
            self._current_observation_n = self.env.reset()
        action_n = []
        for agent, current_observation in zip(self.agents, self._current_observation_n):
            action = agent._policy(torch.Tensor(current_observation).detach())
            action_n.append(np.array(action.detach()))
        next_observation_n, reward_n, done_n, info = self.env.step(action_n)
        self._path_length += 1
        self._path_return += np.array(reward_n, dtype=np.float32)
        self._total_samples += 1
        for i, agent in enumerate(self.agents):
            action = deepcopy(action_n[i])
            agent._pool.add_sample(observation=self._current_observation_n[i],
                                    action=action,
                                    reward=reward_n[i],
                                    terminal=done_n[i],
                                    next_observation=next_observation_n[i])
        #np.all?done_n?
        if np.all(done_n) or self._path_length >= self._max_path_length:
            self._current_observation_n = self.env.reset()
            self._max_path_return = np.maximum(self._max_path_return, self._path_return)
            self._mean_path_return = self._path_return / self._path_length
            self._last_path_return = self._path_return

            self._path_length = 0

            self._path_return = np.array([0.] * self.agent_num, dtype=np.float32)
            self._n_episodes += 1

            self.x_dot.append(self._n_episodes)
            
            self.y_max_r_dot.append(self._max_path_return)
            self.y_mean_r_dot.append(self._mean_path_return)
            self.y_r_dot.append(self._last_path_return)

        else:
            self._current_observation_n = next_observation_n