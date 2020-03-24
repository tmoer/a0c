# -*- coding: utf-8 -*-
"""
Chain environment
@author: thomas
"""

import gym.spaces
import numpy as np

class ChainOrdered(object):
    ''' Chain domain '''
    
    def __init__(self,n=10):
        # n = length of chain
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(n+1)        
        self.n = n      
        self.state = 0
        self.correct = np.repeat(1,n)

    def reset(self):
        self.state = 0
        return self.state        

    def step(self,a):
        if a == 0:
            # move back
            self.state = 0
            r = 0
            terminal = True
        elif a == 1:
            # move forward
            self.state += 1
            if self.state == self.n:
                r = 1
                terminal = True
            else:
                r = 0
                terminal = False
        else:
            raise ValueError('Action not possible')
            
        return self.state,r,terminal, {}
    
    def seed(self,seed):
        pass # deterministic anyway

class Chain(object):
    ''' Chain domain '''
    
    def __init__(self,n=10):
        # n = length of chain
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(n+1)        
        self.n = n      
        self.state = 0
        self.correct = np.random.randint(0,2,n) # correct action in each state
        self.counts = np.zeros((self.n,2))

    def reset(self):
        self.state = 0
        return self.state        

    def step(self,a):
        self.counts[self.state,a] += 1
        if a != self.correct[self.state]:
            # move back
            self.state = 0
            r = 0
            terminal = True
        elif a == self.correct[self.state]:
            # move forward
            self.state += 1
            if self.state == self.n:
                r = 1
                terminal = True
            else:
                r = 0
                terminal = False
        else:
            raise ValueError('Action not possible')
            
        return self.state,r,terminal, {}

    def seed(self,seed):
        pass # deterministic anyway


class ChainLoop(object):
    ''' Chain domain '''
    
    def __init__(self,n=10):
        # n = length of chain
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(n+1)        
        self.n = n      
        self.state = 0
        self.correct = np.random.randint(0,2,n) # correct action in each state
        self.counts = np.zeros((self.n,2))

    def reset(self):
        self.state = 0
        return self.state        

    def step(self,a):
        self.counts[self.state,a] += 1
        if a != self.correct[self.state]:
            # move back
            self.state = 0
            r = 0
            terminal = False
        elif a == self.correct[self.state]:
            # move forward
            self.state += 1
            if self.state == self.n:
                r = 1
                terminal = True
            else:
                r = 0
                terminal = False
        else:
            raise ValueError('Action not possible')
            
        return self.state,r,terminal, {}

    def seed(self,seed):
        pass # deterministic anyway

# Test
if __name__ == '__main__':
    Env = ChainOrdered()
    s = Env.reset()
    for i in range(500): 
        a = Env.action_space.sample()
        s,r,terminal,_ = Env.step(a)
        if terminal:
            print('Died in step',i,'with reward',r,' restarting')
            s = Env.reset() 
    print('Finished')