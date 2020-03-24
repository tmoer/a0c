# -*- coding: utf-8 -*-
"""
Wrapper to generate the game environments
@author: thomas
"""
import gym
import numpy as np
from rl.rewardwrapper import RewardWrapper,PILCOWrapper,NormalizeWrapper
from rl.atariwrapper import AtariWrapper,ClipRewardWrapper
from rl.envs.chain import Chain, ChainOrdered
#from rl.doom_setup import make_doom_env_with_wrappers
from gym import spaces
import os
#import gym_ple

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)
register(
    id='FrozenLakeNotSlippery-v1',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

def make_game(game):
    os.system('export LD_LIBRARY_PATH=`$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin`')

    if 'Chain' in game:
        game,n = game.split('-')
        if game == 'Chain':
            Env = Chain(int(n))
        elif game == 'ChainOrdered':
            Env = ChainOrdered(int(n))
    elif game == 'CartPole-vr' or game == 'MountainCar-vr' or game == 'Acrobot-vr' or game == 'LunarLander-vr':
        Env = RewardWrapper(game)
    elif game == 'CartPole-vp' or game == 'MountainCar-vp' or game == 'Acrobot-vp':
        Env = PILCOWrapper(game)
    elif game == 'CartPole-vn' or game == 'MountainCar-vn':
        Env = NormalizeWrapper(game)
    else:
        Env = gym.make(game)
        if type(Env) == gym.wrappers.time_limit.TimeLimit:
            Env = Env.env
    if game in ['Breakout-v0','Pong-v0','MontezumaRevenge-v0']:
        Env = AtariWrapper(Env,skip=3,k=3,ram=False)
        Env = ClipRewardWrapper(Env)
    elif 'ram' in game:
        Env = AtariWrapper(Env,skip=3,k=2,ram=True)
        Env = ClipRewardWrapper(Env)
    if 'CartPole' in game:
        Env.observation_space = gym.spaces.Box(np.array([-4.8,-10,-4.8,-10]),np.array([4.8,10,4.8,10]))        
    return Env

def check_space(space):    
    '''check the properties of the env '''
    if isinstance(space,spaces.Box):
        dim = space.shape # should the zero be here?
        discrete = False    
    elif isinstance(space,spaces.Discrete):
        dim = space.n
        discrete = True
    else:
        raise NotImplementedError
    return dim, discrete