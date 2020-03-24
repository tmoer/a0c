# -*- coding: utf-8 -*-
"""
Chain environment
@author: thomas
"""

import gym.spaces
import gym
import numpy as np

from gym import Wrapper

class NormalizeWrapper(object):
    ''' Heuristically normalizes the reward scale for CartPole and MountainCar '''
    
    def __init__(self,name):
        # n = length of chain
        if 'CartPole' in name:
            self.env = gym.make('CartPole-v0')
        elif 'MountainCar' in name:
            self.env = gym.make('MountainCar-v0')
        self.name = name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()
        
    def step(self,a):
        s,r,terminal,_ = self.env.step(a)
        r = r/50
        return s,r,terminal, _

class PILCOWrapper(object):
    ''' Wraps according to PILCO '''
    
    def __init__(self,name):
        # n = length of chain
        if 'CartPole' in name:
            self.env = gym.make('CartPole-v0')
        elif 'MountainCar' in name:
            self.env = gym.make('MountainCar-v0')
        self.name = name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()
        
    def step(self,a):
        s,r,terminal,_ = self.env.step(a)
        r = pilco_reward(s,self.name)
        return s,r,terminal, _
    
def pilco_reward(s,game='Cartpole-v0'):
    ''' use modified reward function as in Pilco '''
    from scipy.stats import multivariate_normal
    if game == 'CartPole-vp':
        target = np.array([0.0,0.0,0.0,0.0])
    elif game == 'Acrobot-vp':
        target = np.array([1.0])
        s = -np.cos(s[0]) - np.cos(s[1] + s[0])
    elif game == 'MountainCar-vp':
        target = np.array([0.5])
        s = s[0]
    elif game == 'Pendulum-vp':
        target = np.array([0.0,0.0])
    else:
        raise ValueError('no PILCO reward mofication for this game')
    r = 1 - multivariate_normal.pdf(s,mean=target)
    return r

class RewardWrapper2(Wrapper):
    env = None

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self._warn_double_wrap() 
        while True:
            if hasattr(env,'_spec'):
                self.name = env._spec.id
                break
            else:
                env = env.env
    
    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        return observation, self.reward(reward,terminal), terminal, info

    def reward(self,r,terminal):
        if 'CartPole' in self.name:
            if terminal:
                r = -1
            else:
                r = 0.005
        elif 'MountainCar' in self.name:
            if terminal:
                r = 1
            else:
                r = -0.005
        elif 'Acrobot' in self.name:
            if terminal:
                r = 1
            else:
                r = -0.005
        elif 'LunarLander' in self.name:
            r = r/250.0
        return r
        

class RewardWrapper(object):
    ''' Chain domain '''
    
    def __init__(self,name):
        # n = length of chain
        if name == 'CartPole-vr':
            self.env = gym.make('CartPole-v1')
            if type(self.env) == gym.wrappers.time_limit.TimeLimit:
                self.env = self.env.env
        elif name == 'MountainCar-vr':
            self.env = gym.make('MountainCar-v0')
            if type(self.env) == gym.wrappers.time_limit.TimeLimit:
                self.env = self.env.env
        elif name == 'Acrobot-vr':
            self.env = gym.make('Acrobot-v1')
            if type(self.env) == gym.wrappers.time_limit.TimeLimit:
                self.env = self.env.env
        elif name == 'LunarLander-vr':
            self.env = gym.make('LunarLander-v2')
            # self.env.metadata = {}
            if type(self.env) == gym.wrappers.time_limit.TimeLimit:
                self.env = self.env.env
        self.name = name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()
        
    def step(self,a):
        s,r,terminal,_ = self.env.step(a)        
        if self.name == 'CartPole-vr':
            if terminal:
                r = -1
            else:
                r = 0
        elif self.name == 'MountainCar-vr':
            if terminal:
                r = 1
            else:
                r = 0
        elif self.name == 'Acrobot-vr':
            if terminal:
                r = 1
            else:
                r = 0
        elif self.name == 'LunarLander-vr':
            r = r/250.0
        return s,r,terminal, _

    def seed(self,seed):
        self.env.seed(seed)
                
    def render(self):
        return self.env.render()
        
    def close(self):
        return self.env.close()

# Test
if __name__ == '__main__':
    for game in ['MountainCar-vr','CartPole-vr']:
        Env = RewardWrapper(game)
        s = Env.reset()
        for i in range(500): 
            a = Env.action_space.sample()
            s,r,terminal,_ = Env.step(a)
            if terminal:
                print('Died in step',i,'with reward',r,' restarting')
                s = Env.reset() 
        print('Finished')