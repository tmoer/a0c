# -*- coding: utf-8 -*-
"""
Taxi Env
@author: thomas
"""

import numpy
import random
import gym

class Taxi():
    ''' '''
    
    def __init__(self):
        self.size = numpy.array([4,4])
        self.landmarks = numpy.array([[0.0, 0.0], [0.0, 4.0], [3.0, 0.0], [4.0, 4.0]])
        self.walls = numpy.array([[1.0, 2.0], [2.0, -2.0], [3.0, 2.0]])
        self.fuel = 0
        self.fuel_loc = numpy.array([2.0, 1.0])
        self.pass_loc = 0 # Passenger location: -1 for in taxi, >= 0 for a landmark
        self.pass_dest = 0 # Passenger destination: >=0 for a landmark
        self.pos = numpy.zeros((2,))
        self.observation_space = gym.spaces.Box(0,12,(5,)) 
        self.action_space = gym.spaces.Discrete(6)

    def reset(self):
        self.pos = numpy.random.randint(0,5,(2,))
        self.fuel = numpy.random.random()*7 + 5.0
        self.lm_list = [i for i in range(len(self.landmarks))]
        random.shuffle(self.lm_list)
        self.pass_loc = self.lm_list.pop()
        self.pass_dest = random.choice(self.lm_list)
        return self.get_state()

    def get_state(self):
        return numpy.hstack([self.pos,self.fuel,self.pass_loc,self.pass_dest])

    def step(self,action):
        # move taxi
        reward = self.takeAction(action)
        terminal = 1 if self.isAtGoal() or (self.fuel_loc is not None and self.fuel) < 0 else 0	
        return self.get_state(),reward,terminal,{}

    def takeAction(self, intAction):
        reward = -1.0
        self.fuel -= 1
        prev_pos = self.pos.copy()
        sign = 0
        if intAction == 0:
            self.pos[0] += 1.0
            sign = 1
        elif intAction == 1:
            self.pos[0] -= 1.0
            sign = -1
        elif intAction == 2:
            self.pos[1] += 1.0
        elif intAction == 3:
            self.pos[1] -= 1.0
        elif intAction == 4: # Pickup
            if self.pass_loc >= 0 and self.atPoint(self.landmarks[self.pass_loc]):
                self.pass_loc = -1
            else:
                reward = -10.0
        elif intAction == 5: # Drop off
            if self.pass_loc == -1 and self.atPoint(self.landmarks[self.pass_dest]):
                self.pass_loc = self.pass_dest
                reward = 20.0
            else:
                reward = -10.0
        elif self.fuel_loc is not None and intAction == 4: # Refuel
            if self.atPoint(self.fuel_loc):
                self.fuel = 12.0
    
        self.pos = self.pos.clip([0, 0], self.size)
    
        if sign != 0 and self.hitsWall(prev_pos, self.pos, sign):
            self.pos[0] = prev_pos[0] # Only revert the x-coord, to allow noise and such in y
    
        return reward

    # helpers
    def atPoint(self, point):
        return numpy.linalg.norm(self.pos - point) < 0.1

    def isAtGoal(self):
        return self.pass_loc == self.pass_dest

    def hitsWall(self, old_pos, new_pos, sign):
        return (((self.walls[:,0]*sign >= old_pos[0]*sign) & (self.walls[:,0]*sign < new_pos[0]*sign)) \
				& ((self.walls[:,1] > old_pos[1]) | ((self.size[1]-1)+self.walls[:,1] < old_pos[1]))).any()
   
