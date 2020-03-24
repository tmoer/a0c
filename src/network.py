#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural network specification
@author: thomas
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
from common.rl.make_game import check_space

from pdb import set_trace


class Model():
    
    def __init__(self,Env,lr,n_mix,clip_gradient_norm,loss_type='count',bound='tanh',temp=1.0,entropy_l=0.0):
        
        self.action_dim, self.action_discrete  = check_space(Env.action_space)
        self.state_dim, self.state_discrete  = check_space(Env.observation_space)
        
        if self.action_discrete: 
            raise ValueError('Discrete action space not implemented')
        if len(self.action_dim) > 1:
            raise ValueError('Cant handle multidimensional action spaces')
        else:
            self.action_dim = self.action_dim[0]
            self.scale = Env.action_space.high[0] # assumes a symmetric action space [-scale,scale] for all action_dim
            
        # placeholders
        if not self.state_discrete:
            self.x = x = tf.placeholder("float32", shape=np.append(None,self.state_dim),name='x') # s   
        else:
            self.x = x = tf.placeholder("int32", shape=np.append(None,1)) # s 
            x =  tf.squeeze(tf.one_hot(x,self.state_dim,axis=1),axis=2)
        
        # feedforward
        for i in range(2):
            x = slim.fully_connected(x,128,activation_fn=tf.nn.elu)
            
        # Mixture of Gaussians
        if self.action_discrete:
            raise ValueError('Only works for continuous outputs')
        #print(self.action_dim)
        n_params = n_mix *(2 * self.action_dim)
        z = slim.fully_connected(x,n_params,activation_fn=None) 
        if n_mix > 1:
            logits = slim.fully_connected(x,n_mix,activation_fn=None) 

        # params
        #self.sigma_p = sigma_p = tf.Print(sigma_p,[sigma_p],summarize=16)

        # Make distribution
        if bound == 'tanh':
            self.mu_p = mu_p = z[:,:(self.action_dim*n_mix)]
            log_sigma = z[:,(self.action_dim*n_mix):(2*self.action_dim*n_mix)] 
            self.sigma_p = sigma_p = tf.clip_by_value(tf.nn.softplus(log_sigma),0.001,10000) 
            if n_mix == 1:
                if self.action_dim == 1:
                    outdist = tf.distributions.Normal(mu_p,sigma_p)
                else:
                    outdist = tf.contrib.distributions.MultivariateNormalDiag(mu_p,sigma_p)
            else:                
                p_dist = tf.distributions.Categorical(logits=logits,validate_args=True,allow_nan_stats=False)
                n_dist = []
                for i in range(n_mix):
                    if self.action_dim == 1:
                        n_dist.append(tf.distributions.Normal(mu_p[:,i],sigma_p[:,i]))  
                    else:
                        n_dist.append(tf.contrib.distributions.MultivariateNormalDiag(loc=mu_p[:,(i*self.action_dim):((i+1)*self.action_dim)],scale_diag=sigma_p[:,(i*self.action_dim):((i+1)*self.action_dim)]))  
                outdist = tf.contrib.distributions.Mixture(cat=p_dist,components=n_dist)
        # Wrap distribution        
            outdist = BoundedDistribution(outdist,scale=self.scale)
        elif bound == 'beta':
            self.alpha = alpha = z[:,:(self.action_dim*n_mix)]
            self.beta = beta = z[:,(self.action_dim*n_mix):(2*self.action_dim*n_mix)] 
            if n_mix == 1:
                outdist = tf.contrib.distributions.BetaWithSoftplusConcentration(alpha,beta)
                outdist = BoundedDistributionBeta(outdist,scale=self.scale)
                self.entropy = outdist.entropy()
            else:
                raise ValueError('Beta bounding not implemented for n_mix >1')        
        else:
            raise ValueError('Unknown bounding type: {}'.format(bound))        
        
        # V loss            
        self.V_hat = slim.fully_connected(x,1,activation_fn=None)
        self.V = tf.placeholder("float32", shape=[None,1],name='V')
        self.V_loss = tf.losses.mean_squared_error(labels=self.V,predictions=self.V_hat)
    
        # pi loss (needs a)
        self.a = a = tf.placeholder("float32", shape=np.append(None,self.action_dim),name='a') 
        self.log_pi_a_s = outdist.log_prob(a) # shape (batch,)
        self.pi_hat = outdist.prob(a) # shape (batch,)
        if loss_type == 'count':
            self.n_a = n_a = tf.placeholder("float32", shape=np.append(None,1),name='n_a') 
            pi_loss = tf.stop_gradient(self.log_pi_a_s - tf.log(tf.squeeze(n_a,axis=1))) * self.log_pi_a_s
        elif loss_type == 'Q':
            self.n_a = n_a = tf.placeholder("float32", shape=np.append(None,1),name='Q') 
            pi_loss = tf.stop_gradient(self.log_pi_a_s - tf.squeeze((n_a*temp) - self.V_hat,axis=1)) * self.log_pi_a_s    
        self.pi_loss = tf.reduce_mean(pi_loss)
        self.sample = outdist.sample()
        self.pi_sample = outdist.prob(self.sample)
        
        # training
        self.loss = self.V_loss + self.pi_loss
        if bound == 'beta':
            self.loss -= tf.reduce_mean(entropy_l * self.entropy)
        self.lr = tf.Variable(lr,name="learning_rate",trainable=False)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        var_list = tf.trainable_variables()
        grads = tf.gradients(self.loss, var_list)
        if clip_gradient_norm > 0.0:
            clip_global = tf.Variable(clip_gradient_norm,trainable=False)
            grads,self.gradient_norm = tf.clip_by_global_norm(grads, clip_global)
        else:
            self.gradient_norm = tf.global_norm(grads)
        gvs = list(zip(grads, var_list))
        self.train_op = optimizer.apply_gradients(gvs)
    
    def train(self,D,n_epochs,lr):
        sess = self.sess
        D.reshuffle()
        gn,VL,piL,V = [],[],[],[]
        for epoch in range(n_epochs):
            for sb,Vb,ab,a_nb in D:        
                _,VL_,piL_,gn_,V_ = sess.run([self.train_op,self.V_loss,self.pi_loss,self.gradient_norm,self.V],
                                             feed_dict={self.x:sb,
                                                  self.V:Vb,
                                                  self.a:ab,
                                                  self.n_a:a_nb,
                                                  self.lr:lr
                                                  })
                gn.append(gn_)
                VL.append(VL_)
                piL.append(piL_)
                V.append(np.mean(V_))
                if np.isnan(np.mean(gn)) or np.isnan(np.mean(VL)) or np.isnan(np.mean(piL)) or np.isnan(np.mean(V)):
                    set_trace()
        t_loss = np.mean(VL)+np.mean(piL)
        return {'V_loss':np.mean(VL),'pi_loss':np.mean(piL),'gn':np.mean(gn),'total_loss':t_loss,'V':np.mean(V)}
    
    def predict_V(self,s):
        sess = self.sess
        return sess.run(self.V_hat,feed_dict={self.x:s})
        
    def predict_pi(self,s,a):
        sess = self.sess
        return sess.run(self.pi_hat,feed_dict={self.x:s,
                                               self.a:a})
    
    def log_prob(self,s,a):
        return self.sess.run([self.log_pi_a_s],feed_dict={self.x:s,
                                                     self.a:a})
        qui
    def sample_action(self,s):
        sess = self.sess
        mix_list = sess.run(self.p_dist.sample(),feed_dict={self.x:s})
        samples = np.array([sess.run(self.n_dist[mix].sample(),feed_dict={self.x:s}) for mix in mix_list])
        return samples
    
    def sample_action_and_pi(self,s):
        sess = self.sess
        return sess.run([self.sample,self.pi_sample],feed_dict={self.x:s})

class BoundedDistribution(object):
    ''' Bounded transformation of arbitrary continuous density with support on real line '''
    
    def __init__(self,dist,scale):
        self.dist = dist        
        self.scale = scale
    
    def to_u(self,a):
        return tf.atanh(tf.clip_by_value(a/self.scale,-0.999999,0.999999)) # clip what goes into atanh
    
    def to_a(self,u):
        return self.scale*tf.tanh(u)

    def sample(self):
        return self.to_a(self.dist.sample())
        
    def log_prob(self,a):
        u = self.to_u(a)
        return self.dist.log_prob(u) - tf.reduce_sum(tf.log(self.scale*(1-tf.square(
                tf.clip_by_value(tf.tanh(u),-0.999999,0.999999)))),axis=1) # clip what comes out of tanh and goes into log
    
    def prob(self,a):
        return tf.exp(self.log_prob(a))      

class BoundedDistributionBeta(object):
    ''' Bounded transformation of Beta distribution '''
    
    def __init__(self,dist,scale):
        self.dist = dist
        self.scale = scale
    
    def to_u(self,a):
        return tf.clip_by_value(((a/self.scale) + 1.0)/2.0,0.00001,0.999999)
    
    def to_a(self,u):
        return self.scale * ((2.0 * u) - 1.0)

    def sample(self):
        return self.to_a(self.dist.sample())
        
    def log_prob(self,a):
        u = self.to_u(a)
        shape = a.get_shape().as_list()
        constants = shape[-1]*tf.log(tf.constant(np.array(2.0)*np.squeeze(self.scale),dtype='float32'))
        return tf.reduce_sum(self.dist.log_prob(u),axis=1) - constants
    
    def prob(self,a):
        return tf.exp(self.log_prob(a))    
    
    def entropy(self):
        return self.dist.entropy()


class Database():
    ''' Database '''
    
    def __init__(self,max_size,batch_size):
        self.max_size = max_size        
        self.batch_size = batch_size
        self.size = 0
        self.insert_index = 0
        self.experience = []
        self.sample_array = None
        self.sample_index = 0
    
    def clear(self):
        self.experience = []
        self.insert_index = 0
        self.size = 0
    
    def store(self,experience):
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size +=1
        else:
            self.experience[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0

    def store_from_array(self,*args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            self.store(entry)
        
    def reshuffle(self):
        self.sample_array = np.arange(self.size)
        random.shuffle(self.sample_array)
        self.sample_index = 0
                            
    def __iter__(self):
        return self

    def __next__(self):
        if (self.sample_index + self.batch_size > self.size) and (not self.sample_index == 0):
            self.reshuffle() # Reset for the next epoch
            raise(StopIteration)
          
        if (self.sample_index + 2*self.batch_size > self.size):
            indices = self.sample_array[self.sample_index:]
            batch = [self.experience[i] for i in indices]
        else:
            indices = self.sample_array[self.sample_index:self.sample_index+self.batch_size]
            batch = [self.experience[i] for i in indices]
        self.sample_index += self.batch_size
        
        arrays = []
        for i in range(len(batch[0])):
            to_add = np.array([entry[i] for entry in batch])
            arrays.append(to_add) 
        return tuple(arrays)
            
    next = __next__