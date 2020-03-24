# -*- coding: utf-8 -*-
"""
Atari wrapper, based on https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
@author: thomas
"""
import gym
from gym import spaces
from collections import deque
import numpy as np
from PIL import Image

class ClipRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return 0.5 * np.sign(reward)

class AtariWrapper(gym.Wrapper):
    ''' Chain domain '''
    
    def __init__(self, env, skip=4, k=4,ram=False):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # Frame skip and pooling
        self._obs_buffer = deque(maxlen=skip)
        self._skip = skip    
        self._ram = ram

        # Frame stacking
        self.k = k
        self.frames = deque([], maxlen=k)
        
        # Frame wrapping
        if not self._ram:
            self.res = 84
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.res,self.res, k))
        else:
            self.res = env.observation_space.shape[0]
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.res, k))

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=-1)  

    def _resize(self, obs):
        if not self._ram:
            frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
            frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
                resample=Image.BILINEAR), dtype=np.float32)/255.0
            return frame.reshape((self.res, self.res, 1))
        else:
            obs = obs/255
            return obs.astype('float32').reshape((self.res,1))
            
    def _reset(self):
        """Clear buffers and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        ob = self._resize(ob)
        for _ in range(self.k): self.frames.append(ob)
        self._obs_buffer.clear()
        for _ in range(self._skip): self._obs_buffer.append(ob)
        return self._observation()
        
    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            obs = self._resize(obs)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        if not self._ram:
            max_frame = np.max(np.stack(self._obs_buffer), axis=0) # max over skips
        else:
            max_frame = obs # just take the last, max has no interpretation
        self.frames.append(max_frame) # append to buffer
        return self._observation(), total_reward, done, info
