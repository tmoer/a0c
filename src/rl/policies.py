# -*- coding: utf-8 -*-
"""
Various policies
@author: thomas
"""
import numpy as np
import logging
logger = logging.getLogger('root')
logger.propagate = False

def policy(policy,model,hps,seed,eval_on_mean_output=False,eval_on_mean_params=False):
    ''' wrapper policy function '''    
    pass

def thompson_policy(s,model,sess,hps,seed,eval_on_mean_output=False,eval_on_mean_params=False):
    ''' Thompson sample value function in discrete action space 
    Input:      s - state, Thompson sampling applied across first dimension.
    Output:     a - picked action '''
    
    rep = s.shape[0]
    state_seq = np.repeat(s,model.action_dim,axis=0)
    action_seq = np.repeat(np.arange(0,model.action_dim)[None,:],rep,axis=0).reshape(-1,1)    
    rep_action_values = np.zeros([rep*model.action_dim,hps.n_thompson_sample])
      
    # sample
    for i in range(hps.n_thompson_sample):
        action_values = sample_value(sess,model,hps,state_seq,action_seq,seed,eval_on_mean_output,eval_on_mean_params)
        rep_action_values[:,i] = np.squeeze(action_values)

    # max        
    max_action_values = np.max(rep_action_values,axis=1) # max over the repetitions
    max_action_values = np.reshape(max_action_values,[rep,model.action_dim]) 
    #a = np.argmax(max_action_values,axis=1)[:,None]
    a = argmax_tiebreaking(max_action_values)
    return a
 
def egreedy_policy(s,model,sess,hps,e,seed):
    ''' e-greedy policy on discrete action-space'''
    # setup
    #hps.n_thompson_sample = 1 
    #a_exploit = thompson_policy(s,model,sess,hps,seed,eval_on_mean_output=True,eval_on_mean_params=True)    

    rep = s.shape[0]
    state_seq = np.repeat(s,model.action_dim,axis=0)
    action_seq = np.repeat(np.arange(0,model.action_dim)[None,:],rep,axis=0).reshape(-1,1)    

    action_values = get_net_mean(sess,model,state_seq,action_seq,seed,hps.p_dropout,hps.output)
    action_values = np.reshape(action_values,[rep,model.action_dim]) 
    a_exploit = argmax_tiebreaking(action_values)

    a_explore = get_discrete_random_action(model.action_dim,s.shape[0])
    a = np.array([(a1 if np.random.rand()>0.05 else a2) for a1,a2 in zip(a_exploit,a_explore)])
    return a

def ucb_policy(s,model,sess,hps,seed,eval_on_mean_output=False,eval_on_mean_params=False):
    ''' upper confidence bound policy '''
    #p_dropout = 1.0 if eval_on_mean_params else hps.p_dropout # some unexplainable bug if uncommented
    p_dropout = hps.p_dropout
    
    rep = s.shape[0]
    state_seq = np.repeat(s,model.action_dim,axis=0)
    action_seq = np.repeat(np.arange(0,model.action_dim)[None,:],rep,axis=0).reshape(-1,1)    
              
    mu = get_net_mean(sess,model,state_seq,action_seq,seed,p_dropout,hps.output)
    sds = analytic_sd(sess,model,state_seq,action_seq,seed,p_dropout,hps.output)
    #sds2 = sample_sd(40,sess,model,state_seq,action_seq,p_dropout,hps.output)
    
    ucb_multipliers = np.random.uniform(1.7,2.3,(rep*model.action_dim,1))
    ucb = np.reshape(mu + ucb_multipliers * sds,[-1,model.action_dim])
    #a = np.argmax(ucb,axis=1)[:,None]
    a = argmax_tiebreaking(ucb) 
    return a    

def get_discrete_random_action(n_act,n_sample):
    return np.random.randint(0,n_act,n_sample)[:,None]

def sample_value(sess,model,hps,sb,ab,seed,eval_on_mean_output=False,eval_on_mean_params=False):
    ''' Sample values for policy '''
    if eval_on_mean_params:
        p_dropout = 1.0
    else:
        p_dropout = hps.p_dropout

    if eval_on_mean_output:
        Qsa = get_net_mean(sess,model,sb,ab,seed,p_dropout,hps.output)
    else:
        Qsa = sample_net(sess,model,sb,ab,seed,p_dropout,hps.output)
    return Qsa
    
def sample_net(sess,model,sb,ab,seed,p_dropout,output):
    ''' Sample from network output distribution '''
    sample = sess.run(model.sample,feed_dict = {model.x:sb,
                                                  model.a:ab,
                                                  model.p_dropout: p_dropout,
                                                  model.seed:seed})  
    if output == 'categorical':
        sample = model.transformer.to_value(sample)
    return sample

def get_net_mean(sess,model,sb,ab,seed,p_dropout,output):
    ''' Expectation of network output distribution '''
    if not output == 'categorical':
        Qsa = sess.run(model.mean,feed_dict = {model.x:sb,
                                              model.a:ab,
                                              model.p_dropout: p_dropout,
                                              model.seed:seed})  
    else:
        density = sess.run(model.params,feed_dict = {model.x:sb,
                                                  model.a:ab,
                                                  model.p_dropout: p_dropout,
                                                  model.seed:seed})      
        Qsa = np.matmul(density,model.transformer.means)[:,None]
    return Qsa

def analytic_sd(sess,model,sb,ab,seed,p_dropout,output):
    ''' analytic sd calculation from network parameters '''
    params = get_net_params(sess,model,sb,ab,seed,p_dropout)
    if output == 'gaussian':
        sd = params[:,1][:,None]
    elif output == 'categorical':
        # sd = sum_i (x_i-mu)
        bin_means = model.transformer.means
        mu = np.repeat(np.matmul(params,bin_means)[:,None],params.shape[1],axis=1)
        sd = np.sqrt(np.sum(params * np.square(bin_means - mu), axis=1))[:,None] #
    elif output == 'mog':
        # need to sample
        sd = sd_mog(params)[:,None]
        #sd = sample_sd(20,sess,model,sb,ab,p_dropout,output)
    elif output == 'deterministic':
        sd = sample_sd(15,sess,model,sb,ab,p_dropout,output)
    return sd

def sd_mog(params):
    ''' Standard deviation of gaussian mixture '''
    n_mix = int(params.shape[1]/3)
    p = params[:,:n_mix]
    mu = params[:,n_mix:(2*n_mix)]
    sd = params[:,(2*n_mix):(3*n_mix)]    
    return np.sum(p * (np.square(mu) + np.square(sd)),axis=1) - np.square(np.sum(p*mu,axis=1))

def sample_sd(n,sess,model,sb,ab,p_dropout,output):
    ''' get standard deviation estimates
    Crude implementation, based on sampling. However, there is no better way
    to integrate over the parameter uncertainty '''
    samples = np.zeros([sb.shape[0],n])
    for i in range(n):
        seed = [np.random.randint(1e15),np.random.randint(1e15)] # new seed for parametric uncertainty
        sample = sample_net(sess,model,sb,ab,seed,p_dropout,output)
        samples[:,i] = np.squeeze(sample)    
    sds = np.std(samples,axis=1)[:,None]
    return sds
    
def get_net_params(sess,model,sb,ab,seed,p_dropout):
    ''' Network parameters '''
    params = sess.run(model.params,feed_dict = {model.x:sb,
                                              model.a:ab,
                                              model.p_dropout: p_dropout,
                                              model.seed:seed})  
    return params

def argmax_tiebreaking(x):
    ''' own argmax because numpy.argmax does not break ties '''
    try:    
        out = np.array([[np.random.choice(np.flatnonzero(a == a.max()))] for a in x]) # sparsely fails due to numerical errors between a and a.max()?
    except:
        out = np.array([[np.argmax(a)] for a in x])
    return out