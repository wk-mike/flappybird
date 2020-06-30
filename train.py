#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from ple.games.flappybird import FlappyBird
from ple import PLE

import parl
from parl.utils import logger

from model import Model
from agent import Agent

OBS_DIM = 8
ACT_DIM = 2
LEARNING_RATE = 1e-3


def run_episode(env, agent, train_or_test='train'):
    obs_list, action_list, reward_list = [], [], []
    env.reset_game()
    obs = env.getGameState()
    norm = np.array([512, 10, 512, 256, 256, 512, 256, 256])
    obs=np.array(list(obs.values()))/norm-0.5
    action_choose=[119,None]
    while True:
        obs_list.append(obs)
        if train_or_test == 'train':
            action = agent.sample(obs)
        else:
            action = agent.predict(obs)
        action_list.append(action)

        # obs, reward, done, _ = env.step(action)
        obs=env.getGameState()
        # print(obs)
        obs=np.array(list(obs.values()))/norm-0.5

        action = action_choose[action] # [119 None]
        reward = env.act(action)

        reward_list.append(reward)

        if env.game_over(): #check if the game is over
    #         p.reset_game()
            break
    return obs_list, action_list, reward_list


def calc_reward_to_go(reward_list,gamma = 0.99):
    for i in range(len(reward_list) - 2, -1, -1):
        reward_list[i] += gamma * reward_list[i + 1]
    return np.array(reward_list)




def main():

    import torch
    # env = gym.make('CartPole-v0')
    game = FlappyBird()
    env = PLE(game, fps=30,frame_skip=4, display_screen=True, force_fps=True,
                reward_values = {"tick": 0.00},state_preprocessor=None)
    env.init()

    model = Model(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    if torch.cuda.is_available():
        model = model.cuda()


    model.load_state_dict(torch.load('checkpoint.pt'))

    from parl.algorithms.torch import PolicyGradient
    alg = PolicyGradient(model, LEARNING_RATE)
    agent = Agent(alg)

    for i in range(10000):  # 1000 episodes
        obs_list, action_list, reward_list = run_episode(env, agent)
        if i % 100 == 0:
            logger.info("Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 1000 == 0:
            _, _, reward_list = run_episode(env, agent, train_or_test='test')
            total_reward = np.sum(reward_list)
            logger.info('Test reward: {}'.format(total_reward))

            torch.save(model.state_dict(), 'checkpoint.pt')



def watch():

    import torch

    game = FlappyBird()
    env = PLE(game, fps=30,frame_skip=4, display_screen=True, force_fps=False,
                reward_values = {"tick": 0.00},state_preprocessor=None)
    env.init()

    model = Model(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    if torch.cuda.is_available():
        model = model.cuda()


    model.load_state_dict(torch.load('checkpoint.pt'))

    from parl.algorithms.torch import PolicyGradient
    alg = PolicyGradient(model, LEARNING_RATE)
    agent = Agent(alg)

    for i in range(10000):  # 1000 episodes
        obs_list, action_list, reward_list = run_episode(env, agent)

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        # agent.learn(batch_obs, batch_action, batch_reward)

        _, _, reward_list = run_episode(env, agent, train_or_test='test')
        total_reward = np.sum(reward_list)
        logger.info('Test reward: {}'.format(total_reward))

            # torch.save(model.state_dict(), 'checkpoint.pt')


if __name__ == '__main__':
    main()
    # watch()








    # game = FlappyBird()
    # p = PLE(game, fps=30, display_screen=True, force_fps=True)
    # p.init()

    # print(p.getActionSet())
    # print(p.getGameState())

    # # print(list(p.getGameState().values()))



    # import time

    # while True:
    #     if p.game_over(): #check if the game is over
    #         p.reset_game()
    #         break

    #     # obs = p.getScreenRGB()
    #     obs=p.getGameState()
    #     action = 119 # myAgent.pickAction(reward, obs)
    #     reward = p.act(action)
    #     # time.sleep(0.02)
    #     # print(obs,action,reward)
    

