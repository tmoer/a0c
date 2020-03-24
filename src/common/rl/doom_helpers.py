# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:16:19 2018

@author: thomas
"""
import gym
import ppaquette_gym_doom
from .wrappers.doom.action_space import ToDiscrete
from .wrappers.doom.observation_space import SetResolution

def make_doom_env(game):
    ''' Makes doom environments based on simpler function '''
    if game == 'Doom-1':
        Env = gym.make('ppaquette/DoomBasic-v0')
    if game == 'Doom-2':
        Env = gym.make('ppaquette/DoomCorridor-v0')
    if game == 'Doom-3':
        Env = gym.make('ppaquette/DoomDefendCenter-v0')        
    if game == 'Doom-4':
        Env = gym.make('ppaquette/DoomDefendLine-v0')        
    if game == 'Doom-5':
        Env = gym.make('ppaquette/DoomHealthGathering-v0')   
    if game == 'Doom-6':
        Env = gym.make('ppaquette/DoomMyWayHome-v0')        
    if game == 'Doom-7':
        Env = gym.make('ppaquette/PredictPosition-v0')        
    if game == 'Doom-8':
        Env = gym.make('ppaquette/TakeCover-v0')        
    if game == 'Doom-9':
        Env = gym.make('ppaquette/Deathmatch-v0')
    if game == 'Doom-10':
        Env = gym.make('ppaquette/meta-Doom-v0')
    return Env
    
def make_doom_env_with_wrappers(game,action_config='minimal',screen_res='160x120'):
    '''
    action_config can be 'minimal', 'constant-7', 'constant-17', 'full'
    '''
    Env = make_doom_env(game)
    if action_config is not None:   
        action_wrapper = ToDiscrete(config=action_config)
        Env = action_wrapper(Env)    
    if screen_res is not None:
        obs_wrapper = SetResolution(screen_res)
        Env = obs_wrapper(Env)
    return Env