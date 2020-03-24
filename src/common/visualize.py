# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:17:31 2018
Result plotting

@author: thomas
"""
import matplotlib as mpl
#from prettyplotlib import mpl
#from prettyplotlib import brewer2mpl

mpl.use('Agg')
mpl.rc('font',**{'family':'sans-serif','serif':['Computer Modern Roman']})
#mpl.rc('text', usetex=True)
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams["axes.edgecolor"] = "grey"
mpl.rcParams["axes.linewidth"] = 1

import matplotlib.pyplot as plt
#from prettyplotlib import plt

# modify colour cycle
#plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
colours = plt.rcParams['axes.prop_cycle'].by_key()['color'] #mpl.rcParams['axes.color_cycle']

#plt.style.use('seaborn-colorblind')
#print(colours)
del(colours[6])
#del(colours[2:5])
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colours,linestyle=6*['--'])#,':','-']) 
#plt.rcParams['lines.linewidth'] = 4
#plt.rcParams.update({'font.size': 11})
#plt.rcParams['axes.facecolor']='white'
#plt.rcParams['savefig.facecolor']='white'
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = ['Latin Modern Math']
#plt.rcParams['xtick.labelsize'] = 15
#plt.rcParams['font.weight'] = 'bold'
#plt.rcParams['ytick.labelsize'] = 15
#plt.locator_params(axis='x', nticks=3)
#plt.rc('axes', prop_cycle=(
                            #cycler('color', [ 'g', 'b','r', 'y','c','m','k','w']*2) +
                           #cycler('linestyle', [i for i in ['-', '--'] for j in range(8)])
#                           cycler('linestyle', [['-', '--',':']])
#                           ))

import argparse
import os
import shutil
import numpy as np

from .hps_setup import get_hps_setup

#style = ['-.','--','-']

#plt.rcParams['axes.prop_cycle'] = ("cycler('color', {})".format(colours) +
#                                   "cycler('lw', {})".format(style))

#import seaborn as sns
#from itertools import cycle
#colours = cycle(sns.color_palette(n_colors=3))
#styles = cycle(['--','-.','-']) 
#colours = cycle(colours)



greek_letterz=[chr(code) for code in range(945,970)]
greek_sigma = greek_letterz[18]

def plot_single_experiment(result,game,folder,plot_type='lc',window=25,sd=False,plotlen=200,on_mean=False,item='return',**kwargs):
    ''' Simple learning curve '''
    if plot_type == 'lc':
        fig,ax = plt.subplots(1,figsize=[5,5])
        result = average_over_reps([result],plot_type,window,sd,on_mean,item,plotlen)
        print(result['steps'],result[item])
        ax.plot(result['steps'],result[item],linewidth=4)
        if plot_type == 'lc':
            ax.set_xlabel('Step',fontsize=18,fontweight='bold')
            ax.set_title('{}'.format(game),fontsize=22,fontweight='bold')
            ax.set_ylabel(item.capitalize(),fontsize=18,fontweight='bold') 
        name = '{}_curve.png'.format(item)
        plt.savefig(folder+name,bbox_inches="tight",dpi=300)
    elif plot_type == 'mean':
        pass # visualization would be a single point

def get_ax(ax,row,col):
    ''' always returns the required ax, even if nrow or ncol == 1 '''
    try:
        return ax[row,col]
    except:
        try: 
            return ax[max(col,row)] # if ax has a single dimension, than one of col or row will be 0
        except:
            return ax

class Ax_Label:
    
    def __init__(self,hps_setup,xaxis='item1',cols='item2',rows='item3'):
        self.hps_setup = hps_setup
        self.xaxis = xaxis # which item to end up on the xaxis/coloured
        self.cols = cols # which item to plot over cols
        self.rows = rows # which item to plot over rows
        
    def get_ax(self,ax,it1,it2,it3):
        ''' specify logic of where to plot, given where i1, it2 and it3 should end up '''
        if self.cols == 'item1':
            label_row = True if it1 == 0 else False
            if self.rows == 'item2':
                label_col = True if it2 == 0 else False
                axn = get_ax(ax,it2,it1)
            elif self.rows == 'item3':
                label_col = True if it3 == 0 else False
                axn = get_ax(ax,it3,it1)
        elif self.cols == 'item2':
            label_row = True if it2 == 0 else False
            if self.rows == 'item1':
                label_col = True if it1 == 0 else False
                axn = get_ax(ax,it1,it2)
            elif self.rows == 'item3':
                label_col = True if it3 == 0 else False
                axn = get_ax(ax,it3,it2)
        elif self.cols == 'item3':
            label_row = True if it3 == 0 else False
            if self.rows == 'item1':
                label_col = True if it1 == 0 else False
                axn = get_ax(ax,it1,it3)
            elif self.rows == 'item2':
                label_col = True if it2 == 0 else False
                axn = get_ax(ax,it2,it3)
        return axn, label_row, label_col
    
    def get_label_per_location(self,location,item1,item2,item3,item1_label,item2_label,item3_label):
        ''' get a specific label '''
        if location == 'xaxis':
            hps_label = getattr(self.hps_setup,self.xaxis)
            if self.xaxis == 'item1':
                label = make_label(hps_label,item1,item1_label)
            if self.xaxis == 'item2':
                label = make_label(hps_label,item2,item2_label)
            if self.xaxis == 'item3':
                label = make_label(hps_label,item3,item3_label)            
        elif location == 'col':
            hps_label = getattr(self.hps_setup,self.cols)
            if self.cols == 'item1':
                label = make_label(hps_label,item1,item1_label)
            if self.cols == 'item2':
                label = make_label(hps_label,item2,item2_label)
            if self.cols == 'item3':
                label = make_label(hps_label,item3,item3_label)          
        elif location == 'row':
            hps_label = getattr(self.hps_setup,self.rows)
            if self.rows == 'item1':
                label = make_label(hps_label,item1,item1_label)
            if self.rows == 'item2':
                label = make_label(hps_label,item2,item2_label)
            if self.rows == 'item3':
                label = make_label(hps_label,item3,item3_label)     
        return label            

def make_label(hps_label,item,item_label):
    if hps_label is None:
        return ''
    elif item_label is None:
        return '{}={}'.format(hps_label,maybe_override_item(item))
    elif item_label == 'None':
        return '{}'.format(maybe_override_item(item))
    else:
        return '{}={}'.format(item_label,maybe_override_item(item))        

def maybe_override_item(item):
    if item == 'off':
        item = 'MCTS'
    elif item == 'sigma':
        item = 'MCTS-T'
    elif item == 'sigma_loop':
        item = 'MCTS-T+'
    return item

def make_1d(x):
    ''' ensures x is a 1d vector '''
    return np.reshape(x,-1)

def is_odd(number):
    ''' checks whether number is odd, returns boolean '''
    return bool(number & 1)

def downsample(x,target_len):
    ''' downsample x by recursively halving it '''
    x = make_1d(x)    
    while x.size > (2 * target_len):
        if is_odd(x.size):
            x = x[:-1]        
        x = np.mean(np.reshape(x,[-1,2]),axis=1)
    return x

def smooth(y,window,mode):
    ''' smooth 1D vectory y '''    
    return np.convolve(y, np.ones(window)/window, mode=mode)

def sort_y_by_x(x,y):
    ''' sorts y according to x, returns both in sorted form '''
    x_order = np.argsort(x)
    x = x[x_order]
    y = y[x_order]
    return x,y

def average_lc(result_reps,window,sd,on_mean,item,plotlen,x_item='steps'):
    steps = np.array([])
    returns = np.array([])
    sds = np.array([])
    n = 0
    for result in result_reps:
        if result is not None:
            n += 1
            if x_item == 'steps':
                steps = np.append(steps,result['steps'])
            elif x_item == 'episodes':
                steps = np.append(steps,np.arange(len(result[item]))) 
            else:
                raise ValueError('x_item should either be "steps" or "episodes"')
            returns = np.append(returns,result[item])
    if len(returns) > 0:
        # sort
        steps,returns = sort_y_by_x(x=steps,y=returns)
        if sd:
            returns_raw = returns
            steps_raw = steps
        # downsample
        steps,returns = downsample(steps,plotlen),downsample(returns,plotlen)
        # smooth
        while len(returns) < (2*window):
            window = int(window/2)
            if window == 1:
                break
        returns = smooth(returns,window,mode='valid')
        steps = symmetric_remove(steps,window-1)  # need to remove (window-1) elements from steps, due to smoothing of returns
        if sd:
            sds = estimate_sd(steps,returns,steps_raw,returns_raw)
            if on_mean:
                sds /= np.sqrt(n) # shrink sd by sqrt(n) to get sd of the mean
    result = {item:returns,'steps':steps}
    if sd: result.update({'sd':sds})
    return result

def elementwise_and(list_a,list_b):
    return [ x&y for (x,y) in zip(list_a, list_b)]

def estimate_sd(steps,returns,steps_raw,returns_raw):
    boundaries = np.append(np.append([0],steps[:-1] + np.ediff1d(steps)),[np.Inf])
    sds = []
    for i,mean in enumerate(returns):
        index = elementwise_and(steps_raw > boundaries[i],steps_raw < boundaries[i+1])
        sd = np.sqrt(np.sum(np.square(returns_raw[index]-mean)))
        sds.append(sd)
    return sds

def symmetric_remove(x,n):
    ''' removes n items from beginning and end '''
    odd = is_odd(n)
    half = int(n/2)
    if half > 0:
        x = x[half:-half]
    if odd:
        x = x[1:]
    return x

def aggregate_over_reps(result_rep,item):
    out = np.array([])
    for result in result_rep:
        if result is not None:
            out = np.append(out,result[item])
    return out

def average_over_reps(result_rep,plot_type,window,sd,on_mean,item,plotlen=2000,x_item='steps'):
    ''' how to deal with repetitions '''
    if plot_type == 'lc':
        result = average_lc(result_rep,window,sd,on_mean,item,plotlen,x_item=x_item)
        return result
    elif plot_type == 'mean':
        # average over all returns in result_rep
        out = aggregate_over_reps(result_rep,item)
        if len(out)>0:
            result = {}
            result[item] = np.mean(out)
            if on_mean:
                result['sd'] = np.std(out)/np.sqrt(len(out))
            else:
                result['sd'] = np.std(out)
            return result
        else:
            return None

def errorfill(x, y, yerr, label,alpha_fill=0.3, ax=None,errlim=None):
    ax = ax if ax is not None else plt.gca()
    base_line, = ax.plot(x, y, label=label)
    if yerr is not None:
        ymin = y - yerr
        ymax = y + yerr
        if errlim is not None:
            ymin = np.maximum(ymin,errlim[0])
            ymax = np.minimum(ymax,errlim[1])
        color = base_line.get_color()
        ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def errorbar(x, y, yerr, label,ax=None,errlim=None):
    ax = ax if ax is not None else plt.gca()
    if yerr is not None:
        ymin = np.array(y) - np.array(yerr)
        ymax = np.array(y) + np.array(yerr)
        if errlim is not None:
            ymin = np.maximum(ymin,errlim[0])
            ymax = np.minimum(ymax,errlim[1])
        yerr = [np.abs(y-ymin),np.abs(ymax-y)]
    ax.errorbar(x,y,yerr=yerr,label=label,marker='o',markersize=8,capsize=8,capthick=3)        
        
def makeup_axes(axn):
    axn.tick_params(axis='both', which='major', labelsize=13)
    axn.tick_params(axis='both', which='minor', labelsize=11)
    axn.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
    axn.xaxis.major.formatter._useMathText = True

def capitalize(s):
    s = s[:1].upper() + s[1:]

def determine_fig_size(hps_setup,col_item,row_item):
    if col_item == 'item1':
        ncol = len(hps_setup.seq1)
    if col_item == 'item2':
        ncol = len(hps_setup.seq2)
    if col_item == 'item3':
        ncol = len(hps_setup.seq3)
    if row_item == 'item1':
        nrow = len(hps_setup.seq1)
    if row_item == 'item2':
        nrow = len(hps_setup.seq2)
    if row_item == 'item3':
        nrow = len(hps_setup.seq3)
    return ncol,nrow

def plot_hyperloop_results(results,hps_setup,folder,plot_type='lc',window=25,sd=False,on_mean=False,
                           item='return',plotlen=500,xlim=None,ylim=None,errlim=None,save_dir=None,
                           item1_label=None,item2_label=None,item3_label=None,no_suptitle=False,
                           x_item='steps',line_item='item1',col_item='item2',row_item='item3'):
    ''' plot results if hyperlooping '''
    n1,n2,n3 = len(hps_setup.seq1),len(hps_setup.seq2),len(hps_setup.seq3) # only used for mean plotting now
    if plot_type == 'lc':
        ncol,nrow = determine_fig_size(hps_setup,col_item,row_item)
        fig,ax = plt.subplots(ncols=ncol,nrows=nrow,sharex=True,sharey=True,figsize=[6*ncol,5*nrow])
        AxL = Ax_Label(hps_setup,xaxis=line_item,cols=col_item,rows=row_item)
        for it1,item1 in enumerate(hps_setup.seq1):
            for it2,item2 in enumerate(hps_setup.seq2):
                for it3,item3 in enumerate(hps_setup.seq3):
                    # select ax
                    #axn = get_ax(ax,it3,it2)
                    axn,label_row,label_col = AxL.get_ax(ax,it1,it2,it3)

                    #label = make_label(hps_setup.item1,item1,item1_label) if hps_setup.item1 is not None else ''
                    label = AxL.get_label_per_location('xaxis',item1,item2,item3,item1_label,item2_label,item3_label)
                    rowlabel = AxL.get_label_per_location('row',item1,item2,item3,item1_label,item2_label,item3_label)
                    collabel = AxL.get_label_per_location('col',item1,item2,item3,item1_label,item2_label,item3_label)

                    result = average_over_reps(results[it1][it2][it3],plot_type,window,sd,on_mean,item,plotlen,x_item)
                    sd_hat = result['sd'] if sd else None
                    # plot
                    print(result['steps'],result[item])
                    errorfill(result['steps'],result[item],yerr=sd_hat,label=label,ax=axn,errlim=errlim)
                    axn.set_xlabel(x_item.capitalize(),fontsize=14)#,fontweight='bold')
                    axn.set_ylabel(item.capitalize(),fontsize=14)#,fontweight='bold')  
                    if xlim is not None: axn.set_xlim(xlim)
                    if ylim is not None: axn.set_ylim(ylim)
                    makeup_axes(axn)
                    if label_col:
                        axn.set_title(collabel,fontsize=16,fontweight='bold')
                    if label_row:
                        axn.text(-0.5, 0.5,rowlabel,fontsize=16,fontweight='bold',transform = axn.transAxes)
                    
                    #if hps_setup.item2 is not None and it3==0:
                    #    axn.set_title(make_label(hps_setup.item2,item2,item2_label),fontsize=16,fontweight='bold')
                    #if hps_setup.item3 is not None and it2==0:
                    #    axn.text(-0.5, 0.5,make_label(hps_setup.item3,item3,item3_label), fontsize=16,fontweight='bold',transform = axn.transAxes)
    elif plot_type == 'mean':
        
        fig,ax = plt.subplots(ncols=n3,nrows=1,sharex=True,sharey=False,figsize=[5*n3,5])
        for it3,item3 in enumerate(hps_setup.seq3):
            axn = get_ax(ax,0,it3)                
            for it2,item2 in enumerate(hps_setup.seq2):
                label = make_label(hps_setup.item2,item2,item2_label) if hps_setup.item2 is not None else ''

                x,R,sd_hat = [],[],[]
                for it1,item1 in enumerate(hps_setup.seq1):
                    result = average_over_reps(results[it1][it2][it3],plot_type,window,sd,on_mean,item,plotlen)
                    if result is not None:
                        x.append(it1)
                        R.append(result[item])
                        if sd:
                            sd_hat.append(result['sd'])
                if not sd: 
                    sd_hat = None
                errorbar(x, R, sd_hat, label=label,ax=axn,errlim=errlim)
                axn.set_xlabel(hps_setup.item1 if item1_label is None else item1_label,fontsize=14,fontweight='bold')
                if hps_setup.item3 is not None:
                    axn.set_title(make_label(hps_setup.item3,item3,item3_label),fontsize=16,fontweight='bold')
                axn.set_ylabel(item.capitalize(),fontsize=14,fontweight='bold')                
                if ylim is not None: axn.set_ylim(ylim)
                axn.set_xlim([-0.2,n1-0.8])
                axn.set_xticks([i for i in range(n1)])
                axn.set_xticklabels([maybe_override_item(item1) for item1 in hps_setup.seq1],fontsize=12)
                axn.tick_params(axis='both', which='major', labelsize=13)
                axn.tick_params(axis='both', which='minor', labelsize=11)
            
    # add legend and save
    if not no_suptitle:
        plt.figtext(0.5, 1.05,hps_setup.game, wrap=True, horizontalalignment='center', fontsize=18)
    plt.legend(handlelength=5)
    handles,labels = axn.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(0.5,-0.02),bbox_transform = plt.gcf().transFigure,fontsize=14)#,fancybox=True)  
    fig.tight_layout()
    name = '{}_curve.png'.format(item) if plot_type == 'lc' else '{}_averages.png'.format(item)
    plt.savefig(folder+name,bbox_inches="tight",dpi=300)
    if save_dir is not None:
        plt.savefig(save_dir+name,bbox_inches="tight",dpi=300)


def get_subdirs(a_dir):
    ''' returns a list of all subdirectories '''
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def make_name(basename='',item1=None,val1=None,item2=None,val2=None,item3=None,val3=None,separator='-'):
    name = basename
    if item1 is not None:
        name += '{}:{}'.format(item1,val1)
    if item2 is not None:
        name += separator + '{}:{}'.format(item2,val2)
    if item3 is not None:
        name += separator + '{}:{}'.format(item3,val3)
    return name

def nested_list(n1,n2,n3,nrep):
    results=[]
    for i in range(n1):
        results.append([])
        for j in range(n2):
            results[-1].append([])
            for k in range(n3):
                results[-1][-1].append([])
                for l in range(nrep):
                    results[-1][-1][-1].append([])
    return results

def load_result(folder):
    ''' Try to load the result file '''
    try:
        result = np.load(folder+'result.npy').item()
    except:
        try:
            result = np.load(folder+'result_old.npy').item()
        except:
            result = None
    return result

def load_hps_setup(folder,get_hps):
    ''' Try to get the stored hps '''
    hps = get_hps()        
    with open(folder + 'hps_setup_raw.txt', 'r') as f:
        hps_list = f.read()        
    hps_setup = get_hps_setup().parse(hps_list,hps)
    return hps_setup

def retrieve_hyperloop_results(folder,hps_setup):
    n1,n2,n3 = len(hps_setup.seq1),len(hps_setup.seq2),len(hps_setup.seq3)
    results = nested_list(n1,n2,n3,hps_setup.n_rep) # handle plotting within this call, so agregate results

    for it1,item1 in enumerate(hps_setup.seq1):
        for it2,item2 in enumerate(hps_setup.seq2):
            for it3,item3 in enumerate(hps_setup.seq3):
                hyperloop_name = make_name('',hps_setup.item1,item1,hps_setup.item2,item2,
                                           hps_setup.item3,item3)
                for rep in range(hps_setup.n_rep):     
                    result_folder = folder + hyperloop_name + 'rep:{}'.format(rep)
                    results[it1][it2][it3][rep] = load_result(result_folder)
    return results

def visualize(folder,save_dir,game,rep,remove,get_hps,**kwargs):
    ''' Visualize the result contents of a folder '''
    print('Processing folder {}'.format(folder))
    if not 'h' in rep:
        # this is a solo experiment
        result = load_result(folder)
        plot_single_experiment(result=result,game=game,folder=save_dir,**kwargs)
    else:
        try: 
            hps_setup = load_hps_setup(folder,get_hps=get_hps)
        except Exception as e:
            print('Base experiment folder {} with error {}, skipping folder'.format(folder,e))
            if remove:
                print('Removing empty folder {}'.format(folder))
                shutil.rmtree(folder)
            else:
                print('Run with --remove flag to delete such subdirectories')
            return
        
        results = retrieve_hyperloop_results(folder,hps_setup)
        plot_hyperloop_results(results=results,hps_setup=hps_setup,folder=folder,save_dir=save_dir,**kwargs)

def loop_directories(result_folder,overview_dir,game=None,name=None,subindex=None,get_hps=None,**kwargs):
    ''' loops through all folders in result_folder, unless game, name and subindex are specified '''
    name_dirs = get_subdirs(result_folder) if name is None else [name]
    print(name_dirs)
    for name_dir in name_dirs:
        if name_dir == 'jobs' or name_dir == 'learning_curves' or name_dir == '.git' or name_dir == 'ignore': continue
        game_dirs = get_subdirs(result_folder + name_dir + '/' ) if game is None else [game]
        for game_dir in game_dirs:
            if not os.path.exists(result_folder + name_dir + '/' + game_dir): continue
            rep_dirs = get_subdirs(result_folder + name_dir + '/' + game_dir + '/') if subindex is None else [subindex]
            for rep_dir in rep_dirs:
                visualize(result_folder + name_dir + '/' + game_dir + '/' + rep_dir + '/',
                          overview_dir+name_dir+game_dir+rep_dir,game=game_dir,rep=rep_dir,get_hps=get_hps,**kwargs)
    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--folder', default='/home/thomas/mcts_results')
    parser.add_argument('--home', action='store_true',default=False)
    parser.add_argument('--game', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--subindex', default=None)
    parser.add_argument('--plot_type', default='lc')
    parser.add_argument('--window', type=int,default=10)
    parser.add_argument('--sd', action='store_true',default=False)
    parser.add_argument('--on_mean', action='store_true',default=False)
    parser.add_argument('--item', default='return',help='This item in result will be plotted')
    parser.add_argument('--remove', action='store_true',default=False)
    parser.add_argument('--plotlen', type=int,default=50)
    parser.add_argument('--xlim', nargs='+',type=float,default=None)
    parser.add_argument('--ylim', nargs='+',type=float,default=None)
    parser.add_argument('--errlim', nargs='+',type=float,default=None,help='Limits on the errorbars')  
    parser.add_argument('--item1_label', nargs='+', default=None)
    parser.add_argument('--item2_label', nargs='+', default=None)
    parser.add_argument('--item3_label', nargs='+', default=None)
    parser.add_argument('--no_suptitle', action='store_true',default=False)        
    parser.add_argument('--x_item', default='steps') # steps or eps    

    parser.add_argument('--line_item', default='item1') #   
    parser.add_argument('--col_item', default='item2') #     
    parser.add_argument('--row_item', default='item3') #    

    args = parser.parse_args()