# -*- coding: utf-8 -*-
"""
MCTS with tree uncertainty
@author: Thomas Moerland, Delft University of Technology
"""

import numpy as np
import random
import copy
import time

from pdb import set_trace

from common.putils import my_argmax, timing

def bring_Env_to_correct_state(Env,seed,a_his):
    ''' Forward simulates an environment based an a history of taken actions and a seed 
    Note: not used because seeding is just as slow as copy.deepcopy(env) '''
    if len(a_his) == 0:
        return Env
    Env.reset()
    #Env.seed(seed) # this takes just as long as copy.deepcopy(Env), and is therefore avoided. We choose to
    # only consider games with deterministic initial state and deterministic transitions, which avoids seeding
    for a in a_his:
        Env.step(a)
    return Env

def MCTS(root_index,root,Env,N,model=None,c=1.0,gamma=1.0,bootstrap_V=False,
         block_loop=False,sigma_tree=False,backup_Q='on-policy',backup_sigma_tree='uniform',
         seed=None,a_his=None,alpha=0.5,C_widening=1.0,use_prior=False,timeit=False,random_action_frac=0.1):
    ''' Monte Carlo Tree Search function '''
    if root is None:
        root = State(root_index,r=0.0,terminal=False,parent_action=None,model=model,action_space=Env.action_space,
                     sigma_tree=sigma_tree,alpha=alpha,C_widening=C_widening,use_prior=use_prior,
                     random_action_frac=random_action_frac) # initialize the root node
    else:
        root.parent_action = None # continue from current root

    if root.terminal:
        raise(ValueError("Can't do tree search from a terminal state. You need to call reset after the Env returns done."))
    
    if timeit:
        copy_time = 0.0
        forward_time = 0.0
        backward_time = 0.0    
    
    for i in range(N):     
        state = root # reset to root for new trace
        if timeit: now = time.time()
        mcts_env = copy.deepcopy(Env) # copy original Env to rollout from
        #mcts_env = bring_Env_to_correct_state(Env,seed,a_his)    
        if timeit:
            copy_time += time.time()-now 
            now = time.time()           
        
        while not state.terminal: 
                
            action = state.select(c=c)
            s1,r,t,_ = mcts_env.step(action.index)
            if hasattr(action,'child_state'):
                state = action.child_state # select
                continue
            else:
                state = action.add_child_state(s1,r,t,model,sigma_tree) # expand
                state.evaluate(model,mcts_env,bootstrap_V=bootstrap_V,gamma=gamma) # evaluate/roll-out
                # check for looping of expanded state
                if block_loop:
                    looped = check_for_loop_in_trace(state,threshold=0.01)
                    if looped:
                        state.sigma_t = 0.0 # block all uncertainty
                        state.V = 0.0
                break
        if timeit:   
            forward_time += time.time()-now 
            now = time.time()           

        # backup the expansion    
        V = np.squeeze(state.V)            
        # loop back up
        while state.parent_action is not None:
            Q = state.r + gamma * V
            action = state.parent_action
            action.update(Q,backup_Q=backup_Q)
            state = action.parent_state
            V = state.update(action.index,Q,backup_Q=backup_Q,backup_sigma_tree=backup_sigma_tree)
        
        if timeit:
            backward_time += time.time()-now 
    if timeit:
        total_time = copy_time + forward_time + backward_time
        print('total time {}\n copy % {}, forward % {}, backward % {}'.format(total_time,100*copy_time/total_time,100*forward_time/total_time,100*backward_time/total_time))
    return root  

def check_for_loop_in_trace(state,threshold=0.01):
    ''' loops back through trace to check for a loop (= repetition of state) '''
    index = state.index
    action = state.parent_action
    while state.parent_action is not None:
        state = action.parent_state
        if np.linalg.norm(state.index-index) < threshold:
            return True
        action = state.parent_action
    return False
        
class Action():
    ''' Action object '''
    def __init__(self,index,parent_state):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0 # sum
        self.n = 0 # counts
        self.Q = 0.0 # mean
                
    def add_child_state(self,s1,r,terminal,model,sigma_tree):
        self.child_state = State(index=s1,r=r,terminal=terminal,parent_action=self,model=model,use_prior=self.parent_state.use_prior,action_space=self.parent_state.action_space, 
                                 sigma_tree=sigma_tree,alpha=self.parent_state.alpha,C_widening=self.parent_state.C_widening,
                                 random_action_frac=self.parent_state.random_action_frac)
        return self.child_state
        
    def update(self,val,backup_Q='on_policy'):
        self.n += 1
        if backup_Q == 'on-policy':
            self.W += val
            self.Q = self.W/self.n
        elif backup_Q == 'max':
            self.Q = val

def stable_normalizer(x,temp):
    x = x / np.max(x)
    return (x ** temp)/np.sum(x ** temp)

def normalizer(x,temp):
    return np.abs((x ** temp)/np.sum(x ** temp))

class State():
    ''' State object '''

    def __init__(self,index,r,terminal,parent_action,model,action_space,sigma_tree=False,use_prior=False,
                 alpha=0.5,C_widening=1.0,random_action_frac=0.1):
        ''' Initialize a new state '''
        self.index = index # state
        self.r = r # reward upon arriving in this state
        self.terminal = terminal 
        self.parent_action = parent_action 
        self.n = 0 # visitation count
        self.sigma_tree = sigma_tree # boolean indicating use of sigma_tree
        self.use_prior = use_prior
        self.model = model
        
        self.alpha=alpha
        self.C_widening=C_widening
        self.action_space = action_space
        self.random_action_frac = random_action_frac

        self.sigma_t = 1.0 if not terminal else 0.0 

        if not terminal:
            self.child_actions = []
            self.priors = []
            self.sigma_actions_t = []
            #self.add_child_actions()
                    
    def required_number_of_children(self):
        return np.max([2,np.ceil(self.C_widening * (self.n ** self.alpha))])

    #@timing
    def add_child_actions(self):   
        ''' Adds child nodes for all actions '''
        if self.required_number_of_children() > len(self.child_actions):
            # add a child action
            if np.random.random() > self.random_action_frac:
                a,prior = self.model.sample_action_and_pi(self.index[None,:])
            else:
                a = self.action_space.sample()
                prior = self.model.log_prob(self.index[None,:],a[None,:])
            a=np.squeeze(a,axis=0)
            if a.ndim == 0:
                a = a[None]
            # add a child
            self.child_actions.append(Action(a,parent_state=self))
            if self.use_prior:
                self.priors.append(np.squeeze(prior))
            if self.sigma_tree:            
                self.sigma_actions_t.append(1.0)
                
    #@timing
    def select(self,c):
        ''' Select one of the child actions based on UCT rule '''
        # first check whether we need to add a child        
        self.add_child_actions()

        Q = np.array([child_action.Q for child_action in self.child_actions],dtype='float32')
        U = np.array([c * (np.sqrt(self.n)/child_action.n) if child_action.n >= 1 else np.Inf for child_action in self.child_actions],dtype='float32')
        if self.use_prior:
            U *= np.array(self.priors,dtype='float32')
        if self.sigma_tree:
            U *= np.array(self.sigma_actions_t,dtype='float32')
        scores = np.squeeze(Q + U)
        winner = my_argmax(scores)
        if np.any(np.isnan(scores)):
            print('Q (means): {}, U (UCB): {}'.format(Q,U))
            raise ValueError('Nans produced in select step')
            #set_trace()
        return self.child_actions[winner]

    def return_results(self,decision_type='count',loss_type='count',V_decision='on-policy',temperature=1):
        # aggregate some results
        counts = np.array([child_action.n for child_action in self.child_actions],dtype='float32')
        Q = np.array([child_action.Q for child_action in self.child_actions],dtype='float32')
        a_list = [child_action.index for child_action in self.child_actions]

        # decision
        if decision_type == 'count':
            a_argmax = my_argmax(counts)
        elif decision_type == 'mean':
            Q2 = np.array([child_action.Q if child_action.n > 0 else -np.Inf for child_action in self.child_actions])
            a_argmax = my_argmax(Q2) 
        a_chosen = self.child_actions[a_argmax].index
        
        # loss
        if loss_type == 'count':
            probs = stable_normalizer(counts,temperature)
        elif loss_type == 'Q':
            probs = Q # needs logsumexp

        # estimate V   
        if V_decision == 'on_policy':        
            V = np.sum((counts/np.sum(counts))*Q)[None]
        elif V_decision == 'max':
            V = np.max(Q)[None]
            
        return probs,a_list,V,a_chosen,a_argmax
    
    #@timing
    def evaluate(self,model=None,Env=None,bootstrap_V=False,gamma=1.0):
        self.n += 1
        if self.terminal:
            self.V = 0.0
        else:
            if bootstrap_V:
                self.V = model.predict_V(self.index[None,])  
            else:
                self.V = rollout(self.index,Env,policy='random',model=model,gamma=gamma,a_init=self.child_actions[0].index)
                #self.child_actions[0].update(self.V) # already log which child action was first in the roll-out
                
    def update(self,a,Q,backup_Q='on-policy',backup_sigma_tree='uniform'):
        ''' update statistics on back-ward pass'''
        self.n += 1

        # update tree sigma
        if self.sigma_tree:
            self.sigma_actions_t[a] = self.child_actions[a].child_state.sigma_t
            if backup_sigma_tree == 'uniform':
                self.sigma_t = np.sum(self.sigma_actions_t)/len(self.sigma_actions_t) 
            elif backup_sigma_tree == 'on-policy':
                counts = [child_action.n if child_action.n > 0 else 1 for child_action in self.child_actions]                
                self.sigma_t = np.sum(self.sigma_actions_t * counts)/np.sum(counts) 
            elif backup_sigma_tree == 'max':
                Q = np.array([child_action.Q if child_action.n > 0 else -np.Inf for child_action in self.child_actions])
                amax = np.argmax(Q)
                self.sigma_t = self.sigma_actions_t[amax]

        # pass-on value for upwards propagation
        if backup_Q == 'on-policy':
            V = Q
        elif backup_Q == 'max':
            Q = np.array([child_action.Q if child_action.n > 0 else -np.Inf for child_action in self.child_actions])
            V = np.max(Q)
        return V
              
    def forward(self,a,s1,r,terminal,model):
        if not hasattr(self.child_actions[a],'child_state'):
            # still need to add the next state
            self.child_actions[a].add_child_state(s1,r,terminal,model,self.sigma_tree)    
        elif np.linalg.norm(self.child_actions[a].child_state.index-s1) > 0.01:
            print('Warning: this domain seems stochastic. Throwing away the tree')
            #print(self.child_actions[a].child_state.index - s1)
            #print('Timestep {}'.format(t))
            #print(self.child_actions[a].n,self.child_actions[a].child_state.n,self.child_actions[a].child_state.terminal)
            #print(a,self.child_actions[a].index)
            return None
        else:
            return self.child_actions[a].child_state

def rollout(s,Env,policy,model,gamma,roll_max=300,a_init=None):
    ''' Small rollout function to estimate V(s)
    policy = random or targeted'''
    terminal = False
    R = 0.0
    for i in range(roll_max):
        if i == 0 and a_init is not None:
            a = a_init
        else:
            if policy == 'random':
                a = Env.action_space.sample()
            elif policy == 'targeted':
                pi = np.squeeze(model.predict_pi(s[None,]))
                a = np.random.choice(len(pi),p=pi)
        s1,r,terminal,_ = Env.step(a)
        R += (gamma**i)*r
        s = s1
        if terminal:
            break
    return R

def display_info(root,time,c):
    ''' Display MCTS node info for debugging '''
    if root is not None:
        print('MCTS status for timestep {}'.format(time))
        Q = [child_action.Q for child_action in root.child_actions]
        print('Q values: {}'.format(Q))
        print('counts: {}'.format([child_action.n for child_action in root.child_actions],[child_action.n for child_action in root.child_actions]))            
        priors = np.array(root.priors)  
        print('priors: {}'.format(priors))
        U = [c * (np.sqrt(1 + root.n)/(1 + child_action.n)) for child_action in root.child_actions]
        print('U: {}'.format(U))
        if root.use_prior:
            U *= priors
        scores = np.squeeze(np.array([Q]) + np.array([U]))
        print('scores: {}'.format(scores))
        print('winner: {}'.format(np.argwhere(scores == np.max(scores)).flatten()))             
        print('-----------------------------')