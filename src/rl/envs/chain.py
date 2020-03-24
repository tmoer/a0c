# -*- coding: utf-8 -*-
"""
Chain environment
@author: thomas
"""

import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from rl.policies import get_net_mean, get_net_params, sample_sd, analytic_sd, thompson_policy, ucb_policy
import matplotlib.patches as patches

#plt.style.use('ggplot')
plt.rcParams['lines.linewidth'] = 4
plt.rcParams.update({'font.size': 11})
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Latin Modern Math']
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 15
plt.locator_params(axis='x', nticks=3)
plt.ion()

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

class ChainDomainPlotter(object):

    def __init__(self,Env):
        self.fig,self.ax = plt.subplots(1,figsize=(Env.n*2,4))
        self.n = Env.n
        self.truth = Env.correct
        
        for i in range(self.n):
            for j in range(2):
                if self.truth[i]==j:
                    col = 'g'
                else:
                    col = 'r'
                self.ax.add_patch(patches.Circle((i,j), radius=0.05,color=col))
                
        self.ax.set_xlim([-1,self.n+1])
        self.ax.set_ylim([-1,2])
        self.fig.canvas.draw()
    
    def update(self,counts):
        self.ax.clear()
        for i in range(self.n):
            for j in range(2):
                if self.truth[i]==j:
                    col = 'g'
                else:
                    col = 'r'
                self.ax.add_patch(patches.Circle((i,j), radius=0.05,color=col))
                self.ax.text(i-0.2,j-0.2,'s = {}, a={}\n N = {}'.format(i,j,int(counts[i,j])))

        self.fig.canvas.draw()
         
class ChainPlotter(object):
    
    def __init__(self,truth,n_plot):
        self.fig,self.ax = plt.subplots(2,n_plot,figsize=(n_plot*10,4),sharex=True,sharey=True)
        self.pl = self.ax.flatten('F')
        self.n = 2*n_plot
        
        # setup for predictions
        self.sb = np.repeat(np.arange(0,n_plot,1),2)[:,None]       
        self.ab = np.array([0,1]*n_plot)[:,None]
        self.truth = truth       
        self.fig.canvas.draw()

    def update(self,sess,model,hps,ep):
        # clear plots
        for ax in self.pl:
            ax.clear()
        overall_means = np.zeros([hps.n_rep_visualize,self.n])
        overall_max_dens = np.ones([self.n])*-np.inf
        for k in range(hps.n_rep_visualize):
            # get prediction parameters
            seed = [np.random.randint(1e15),np.random.randint(1e15)] # new seed
            params = get_net_params(sess,model,self.sb,self.ab,seed,hps.p_dropout)
            means = get_net_mean(sess,model,self.sb,self.ab,seed,hps.p_dropout,output=hps.output)        
            overall_means[k,:] = means[:,0]            
            #print(np.concatenate([np.array([0,0,1,1,2,2])[:,None],np.array([0,1,0,1,0,1])[:,None],params],axis=1))
            
            # need to determine range
            if hps.output != 'categorical':
                if hps.output == 'gaussian':
                    mu = params[:,0]
                    sigma = params[:,1]
                elif hps.output == 'mog':
                    mu = params[:,hps.n_mix:(hps.n_mix*2)]
                    sigma = params[:,(2*hps.n_mix):(3*hps.n_mix)]
                elif hps.output == 'deterministic':
                    mu = params[:,0]
                    sigma = 1.0
                
                max_sd = np.max(sigma)
                lower,upper = np.min(mu)-3*max_sd,np.max(mu)+3*max_sd   
            else:
                lower,upper = model.transformer.plot_edges[0],model.transformer.plot_edges[-1]
            
            # update all plots
            x = np.linspace(lower,upper,100)
            for i in range(self.n):
                #self.pl[i].set_xlim([lower,upper])
                param = params[i,:]
                if hps.output == 'deterministic':
                    max_dens = 1.0
                    overall_max_dens[i] = 1.0
                    mean = means[i]
                    self.pl[i].plot([mean,mean],[0,max_dens],':')
                else:
                    if hps.output == 'gaussian' or hps.output == 'mog':
                        if hps.output == 'gaussian':
                            dens = norm.pdf(x,param[0],param[1])
                        elif hps.output == 'mog':
                            dens = [param[j]*norm.pdf(x,param[hps.n_mix+j],param[2*hps.n_mix+j]) for j in range(hps.n_mix)]
                            dens = np.sum(np.array(dens),axis=0)
                        #print(x,param,dens)
                        self.pl[i].plot(x,dens,color='cornflowerblue')
                    elif hps.output == 'categorical':
                        dens = param
                        edges = model.transformer.plot_edges
                        self.pl[i].hist(model.transformer.means,bins=edges,weights=dens,color='cornflowerblue')
                    overall_max_dens[i] = np.max([overall_max_dens[i],np.max(dens)])
        # add the mean
        grand_means = np.mean(np.array(overall_means),axis=0)
        seed = [np.random.randint(1e15),np.random.randint(1e15)] # new seed for parametric uncertainty
        grand_sds = analytic_sd(sess,model,self.sb,self.ab,seed,hps.p_dropout,hps.output)
        #grand_sds = np.ones([len(grand_means),1])

        # get policy estimates
        s = np.arange(0,int(self.n/2),1)[:,None]
        a_thompson = np.array([thompson_policy(s,model,sess,hps,seed,eval_on_mean_output=False,eval_on_mean_params=False) for i in range(100)])
        a_ucb = np.array([ucb_policy(s,model,sess,hps,seed,eval_on_mean_output=False,eval_on_mean_params=False) for i in range(100)])
        
        thompson_probs = np.zeros(self.n)        
        ucb_probs = np.zeros(self.n)
        
        for j,(state,action) in enumerate(zip(self.sb,self.ab)):
            thompson_probs[j] = np.mean(a_thompson[:,state,:] == action)
            ucb_probs[j] = np.mean(a_ucb[:,state,:] == action)

        for i in range(self.n):
            grand_mean = grand_means[i]
            grand_sd = grand_sds[i]
            max_dens = overall_max_dens[i] #np.max(dens) if 'dens' in locals() else 1
            self.pl[i].plot([grand_mean,grand_mean],[0,max_dens],'--',color='orange')
            #self.pl[i].plot([grand_mean-2*grand_sd,grand_mean+2*grand_sd],[max_dens/2,max_dens/2],'--',color='orange')
            self.pl[i].text(0.1,0.75,'$\mu$={:0.2f}'.format(grand_mean),transform=self.pl[i].transAxes)
            self.pl[i].text(0.55,0.75,'$\sigma$={:0.2f}'.format(grand_sds[i][0]),transform=self.pl[i].transAxes)

            #self.pl[i].text(0.1,0.75,'$\mu$={:0.2f}\n$\sigma$={:0.2f}'.format(grand_mean,grand_sds[i][0]),transform=self.pl[i].transAxes)
            #self.pl[i].text(0.55,0.75,'tho={:0.2f}\nucb={:0.2f}'.format(thompson_probs[i],ucb_probs[i]),transform=self.pl[i].transAxes)

        
        for j in range(int(self.n/2)):
            for l in range(2):
                if self.truth[j]==l:
                    val = 1.
                    col = 'g'
                else:
                    val = 0.
                    col = 'r'
                self.ax[l,j].add_patch(patches.Rectangle((0.01,0.01),0.98,0.98,linewidth=10,edgecolor=col,facecolor='none',transform=self.ax[l,j].transAxes))
                if j>0:                
                    plt.setp(self.ax[l,j].get_yticklabels(), visible=False)
                if l==0:                
                    plt.setp(self.ax[l,j].get_xticklabels(), visible=False)
                #self.ax[l,j].set_title('V={:0.2f}'.format(val))
                self.ax[l,j].set_ylim([0,1.0])
                self.ax[l,j].set_xlim([-2.5,2.5])

                
        self.fig.canvas.draw()
        self.fig.savefig(hps.result_dir + 'episode_{}'.format(ep),dpi=300)
        self.fig.canvas.flush_events()

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