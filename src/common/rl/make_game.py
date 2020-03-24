# -*- coding: utf-8 -*-
"""
Wrapper to generate the game environments
@author: thomas
"""
import gym
import numpy as np
from gym import spaces
import os

from .envs.chain import Chain,ChainOrdered,ChainLoop
from .wrappers.control import NormalizeWrapper,ReparametrizeWrapper,PILCOWrapper,ScaleRewardWrapper
from .wrappers.atari import ClipRewardWrapper

# Register deterministic FrozenLakes
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

def get_base_env(env):
    ''' removes all wrappers '''
    while hasattr(env,'env'):
        env = env.env
    return env

def is_atari_game(env):
    ''' Verify whether game uses the Arcade Learning Environment '''
    env = get_base_env(env)
    return hasattr(env,'ale')

def prepare_gym_env(game):
    ''' Modifications to Env '''
    print(game)
    name,version = game.rsplit('-',1)
    if len(version) > 2:
        modify = version[2:]
        game = name + '-' + version[:2]
    else:
        modify = ''
        
    env = gym.make(game)
    # remove timelimit wrapper
    if type(env) == gym.wrappers.time_limit.TimeLimit:
        env = env.env
    
    print(modify)
    # prepare control
    if 'n' in modify and type(env.observation_space) == gym.spaces.Box:
        env = NormalizeWrapper(env)        
    if 'r' in modify:
        env = ReparametrizeWrapper(env)
    if 'p' in modify:
        env = PILCOWrapper(env)
    if 's' in modify:
        env = ScaleRewardWrapper(env)
    
    if 'CartPole' in game:
        env.observation_space = gym.spaces.Box(np.array([-4.8,-10,-4.8,-10]),np.array([4.8,10,4.8,10]))        

    # prepare atari
    if is_atari_game(env):
        env = prepare_atari_env(env)
    return env

def prepare_atari_env(Env,frame_skip=3,repeat_action_prob=0.0,reward_clip=True):
    ''' Initialize an Atari environment '''
    env = get_base_env(Env)
    env.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_prob)
    env.frame_skip = frame_skip
    if reward_clip:
        Env = ClipRewardWrapper(Env)
    return Env

def prepare_chain_env(game):
    game,n = game.split('-')
    if game == 'Chain':
        Env = Chain(int(n))
    elif game == 'ChainOrdered':
        Env = ChainOrdered(int(n))
    elif game == 'ChainLoop':
        Env = ChainLoop(int(n))   
    return Env

def make_game(game):
    ''' Overall wrapper for gym.make_game '''
    os.system('export LD_LIBRARY_PATH=`$HOME/.mujoco/mjpro150/bin`') # necessary for mujoco
    if 'Chain' in game:
        Env = prepare_chain_env(game)
    else:
        Env = prepare_gym_env(game)
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