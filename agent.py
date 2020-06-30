#!/usr/bin/env python
# -*- coding: utf-8 -*-

import parl
import torch
import numpy as np


class Agent(parl.Agent):
    """Agent of Cartpole env.

    Args:
        algorithm(parl.Algorithm): algorithm used to solve the problem.

    """

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")



    def sample(self, obs):
        """Sample an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)
        
        Returns:
            action(int)
        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        prob = self.algorithm.predict(obs)
        prob = prob.cpu().data.numpy()
        action = np.random.choice(len(prob), 1, p=prob)[0]
        return action

    def predict(self, obs):
        """Predict an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)
        
        Returns:
            action(int)
        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        prob = self.algorithm.predict(obs)
        _, action = prob.max(-1)
        return action.item()

    def learn(self, obs, action, reward):
        """Update model with an episode data

        Args:
            obs(np.float32): shape of (batch_size, obs_dim)
            action(np.int64): shape of (batch_size)
            reward(np.float32): shape of (batch_size)
        
        Returns:
            loss(float)

        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)

        loss = self.algorithm.learn(obs, action, reward)
        return loss.item()
