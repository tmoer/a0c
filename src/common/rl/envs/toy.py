# -*- coding: utf-8 -*-
"""
Toy environment to test distribution propagation
@author: thomas
"""

import gym.spaces
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
from rl.policies import get_net_mean, get_net_params, sample_sd, analytic_sd, thompson_policy, ucb_policy
import matplotlib.patches as patches

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


class Toy(object):
    ''' Toy 2-step MDP (deterministic)
    
             s=0
         /         \   
     a=0             a=1
        
     r=-1            r=1
        
     s=1             s=2
    /  \           /    \
        
a=0      a=1      a=0      a=1

r=4      r=1      r=1      r=0 

term     term     term     term               
    
    
    '''
    
    def __init__(self,state_space='discrete'):
        # n = length of chain
        self.action_space = gym.spaces.Discrete(2)
        if state_space == 'discrete':
            self.observation_space = gym.spaces.Discrete(3)        
        elif state_space == 'continuous':
            self.observation_space = gym.spaces.Box(-1,4,(1,))        
        self.state = 0
        self.counts = np.zeros(6)

    def reset(self):
        self.state = 0
        return self.state        

    def step(self,a):
        if self.state == 0:
            if a == 0:
                self.state = 1
                r = -1
                terminal = False
                self.counts[0] += 1
            elif a == 1:
                self.state = 2
                r = 1
                terminal = False
                self.counts[1] += 1
        elif self.state == 1:
            if a == 0:
                self.state = 4
                r = 4
                terminal = True
                self.counts[2] += 1
            elif a == 1:
                self.state = 4
                r = 1
                terminal = True
                self.counts[3] += 1                
        elif self.state == 2:
            if a == 0:
                self.state = 4
                r = 1
                terminal = True
                self.counts[4] += 1
            elif a == 1:
                self.state = 4
                r = 0
                terminal = True
                self.counts[5] += 1
        return self.state,r,terminal, {}

class ToyDomainPlotter(object):

    def __init__(self,hps):
        fontsize = 20
        self.fig,self.ax = plt.subplots(1,figsize=(8,8))
        self.ax.add_patch(patches.Circle((0,1), radius=0.05,color='k'))
        self.ax.text(0.07,1,'$s_0$',fontsize=fontsize)
        self.ax.add_patch(patches.Circle((-0.5,0), radius=0.05,color='k'))
        self.ax.text(-0.43,0,'$s_1$',fontsize=fontsize)
        self.ax.add_patch(patches.Circle((0.5,0), radius=0.05,color='k'))
        self.ax.text(0.57,0,'$s_2$',fontsize=fontsize)
        self.ax.add_patch(patches.Circle((-0.75,-1), radius=0.05,color='k'))
        self.ax.add_patch(patches.Circle((-0.25,-1), radius=0.05,color='k'))
        self.ax.add_patch(patches.Circle((0.25,-1), radius=0.05,color='k'))
        self.ax.add_patch(patches.Circle((0.75,-1), radius=0.05,color='k'))
        self.basestrings = ['$a_0$,\n$r=-1$','$a_1$,\n$r=1$','$a_0$,\n$r=4$','$a_1$,\n$r=1$','$a_0$,\n$r=1$','$a_0$,\n$r=0$']        
        counts = [[]]*6
        
        self.ax.add_line(plt.Line2D((0, -0.5), (1, 0), lw=2.5,color='k'))        
        counts[0] = self.ax.text(-0.58,0.5,' $a_0$\n$r=-1$',fontsize=fontsize)
        self.ax.add_line(plt.Line2D((0, 0.5), (1, 0), lw=2.5,color='k'))        
        counts[1] = self.ax.text(0.3,0.5,' $a_1$\n$r=1$',fontsize=fontsize)

        self.ax.add_line(plt.Line2D((-0.5, -0.75), (0, -1), lw=2.5,color='k'))        
        counts[2] = self.ax.text(-0.95,-0.5,' $a_0$\n$r=4$',fontsize=fontsize)
        self.ax.add_line(plt.Line2D((-0.5, -0.25), (0, -1), lw=2.5,color='k'))        
        counts[3] = self.ax.text(-0.33,-0.5,' $a_1$\n$r=1$',fontsize=fontsize)

        self.ax.add_line(plt.Line2D((0.5, 0.25), (0, -1), lw=2.5,color='k'))        
        counts[4] =self.ax.text(0.08,-0.5,' $a_0$\n$r=1$',fontsize=fontsize)
        self.ax.add_line(plt.Line2D((0.5, 0.75), (0, -1), lw=2.5,color='k'))        
        counts[5] = self.ax.text(0.67,-0.5,' $a_1$\n$r=0$',fontsize=fontsize)
        
        self.counts = counts
        self.ax.set_xlim([-1.25,1.25])
        self.ax.set_ylim([-1.25,1.25])
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.fig.canvas.draw()
        self.fig.savefig(hps.base_result_dir + 'domain',dpi=300)

    def update(self,new_counts):
        ''' add counts '''
        for i in range(6):
            self.counts[i].set_text(self.basestrings[i]+',\n {}'.format(new_counts[i]))
        self.fig.canvas.draw()
     
class ToyPlotter(object):
    
    def __init__(self):
        self.fig = fig = plt.figure(figsize=(13,8))
        self.pl = [[]]*6
        self.pl[0] = fig.add_subplot(2,2,1)
        self.pl[1] = fig.add_subplot(2,2,2,sharex=self.pl[0],sharey=self.pl[0])
        self.pl[2] = fig.add_subplot(2,4,5,sharex=self.pl[0],sharey=self.pl[0])
        self.pl[3] = fig.add_subplot(2,4,6,sharex=self.pl[0],sharey=self.pl[0])
        self.pl[4] = fig.add_subplot(2,4,7,sharex=self.pl[0],sharey=self.pl[0])
        self.pl[5] = fig.add_subplot(2,4,8,sharex=self.pl[0],sharey=self.pl[0])

            # Shrink current axis's height by 10% on the bottom
        for i in range(2):
            ax_ = self.pl[i]
            box = ax_.get_position()
            ax_.set_position([box.x0 + box.width*0.2, box.y0,
                             box.width*0.6, box.height])                                


        # setup for predictions
        self.sb = np.array([0,0,1,1,2,2])[:,None]       
        self.ab = np.array([0,1,0,1,0,1])[:,None]
        self.truth = [3,2,4,1,1,0]        
        
        self.fig.canvas.draw()
        
        
    def update(self,sess,model,hps,ep):
        # clear plots
        names = ['$s_0,a_0$','$s_0,a_1$','$s_1,a_0$','$s_1,a_1$','$s_2,a_0$','$s_2,a_1$']
        for i in range(6):
            self.pl[i].clear()
            self.pl[i].set_title(names[i],fontsize=22)
        
        overall_means = np.zeros([hps.n_rep_visualize,6])
        overall_max_dens = np.ones([6])*-np.inf
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
            for i in range(6):
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
        
        # get policy estimates
        s = np.array([[0],[1],[2]])
        a_thompson = np.array([thompson_policy(s,model,sess,hps,seed,eval_on_mean_output=False,eval_on_mean_params=False) for i in range(100)])
        a_ucb = np.array([ucb_policy(s,model,sess,hps,seed,eval_on_mean_output=False,eval_on_mean_params=False) for i in range(100)])
        
        thompson_probs = np.zeros(6)        
        ucb_probs = np.zeros(6)
        
        for j,(state,action) in enumerate(zip(self.sb,self.ab)):
            thompson_probs[j] = np.mean(a_thompson[:,state,:] == action)
            ucb_probs[j] = np.mean(a_ucb[:,state,:] == action)
        
        for i in range(6):
            grand_mean = grand_means[i]
            grand_sd = grand_sds[i]
            max_dens = overall_max_dens[i] #np.max(dens) if 'dens' in locals() else 1
            self.pl[i].plot([grand_mean,grand_mean],[0,max_dens],'--',color='orange')
            #self.pl[i].plot([grand_mean-2*grand_sd,grand_mean+2*grand_sd],[max_dens/2,max_dens/2],'--',color='orange')
            self.pl[i].text(0.05,0.75,'$\mu=${:0.2f}\n$\sigma=${:0.2f}'.format(grand_mean,grand_sds[i][0]),transform=self.pl[i].transAxes,fontsize=19)
            self.pl[i].text(0.56,0.75,'tho={:0.2f}\nucb={:0.2f}'.format(thompson_probs[i],ucb_probs[i]),transform=self.pl[i].transAxes,fontsize=19)
            #self.pl[i].text(0.05,0.85,'truth = {}'.format(self.truth[i]),transform=self.pl[i].transAxes)
            self.pl[i].set_ylim([0,3])
            self.pl[i].set_xlim([lower,upper])            
            self.pl[i].set_xlim([-2,7])            
            for spine in self.pl[i].spines.values():
                spine.set_edgecolor('lightgrey')
                spine.set_linewidth(5)                
            self.pl[i].grid(False)
        
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)        
        self.fig.canvas.draw()
        self.fig.savefig(hps.base_result_dir + 'episode_{}'.format(ep),dpi=300)
        self.fig.canvas.flush_events()

# Test
if __name__ == '__main__':
    Env = Toy()
    s = Env.reset()
    for i in range(500): 
        a = Env.action_space.sample()
        s,r,terminal,_ = Env.step(a)
        if terminal:
            print('Died in step',i,'with reward',r,' restarting')
            s = Env.reset() 
    print('Finished')