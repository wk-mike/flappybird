#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import parl

# parl.Model
# nn.Module
class Model(nn.Module):
    """ Linear network to solve Cartpole problem.
    
    Args:
        obs_dim (int): Dimension of observation space.
        act_dim (int): Dimension of action space.
    """

    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        hid1_size = act_dim * 2
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid1_size)
        self.fc3 = nn.Linear(hid1_size, act_dim)

        self.tanh=nn.Tanh()

    def forward(self, x):
        # out = torch.tanh(self.fc1(x))
        out = self.fc1(x)
        out = self.tanh(out)

        out = self.fc2(out)
        out = self.tanh(out)

        prob = F.softmax(self.fc3(out), dim=-1)
        return prob
