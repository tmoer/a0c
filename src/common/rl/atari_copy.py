# -*- coding: utf-8 -*-
"""
Atari helper functions
@author: thomas
"""

def get_base_env(env):
    ''' removes all wrappers '''
    while hasattr(env,'env'):
        env = env.env
    return env

def copy_atari_state(env):
    env = get_base_env(env)
    return env.clone_full_state()
#    return env.ale.cloneSystemState()

def restore_atari_state(env,snapshot):
    env = get_base_env(env)
    env.restore_full_state(snapshot)
#    env.ale.restoreSystemState(snapshot)

def is_atari_game(env):
    ''' Verify whether game uses the Arcade Learning Environment '''
    env = get_base_env(env)
    return hasattr(env,'ale')