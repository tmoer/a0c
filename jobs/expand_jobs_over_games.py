# -*- coding: utf-8 -*-
"""
Expand a submission over games
@author: thomas
"""

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import os
import argparse
from common.visualize import make_name

def expand_job(games,job,hp,hp_setup,item1=None,seq1=[None],item2=None,seq2=[None],item3=None,seq3=[None]):
    # hacky way to bring in games
    #games = ['CartPole-vr','MountainCar-vr','Acrobot-vr','FrozenLake-v0','FrozenLakeNotSlippery-v0','FrozenLakeNotSlippery-v1']
    #games = ['Breakout-ramDeterministic-v0','Pong-ramDeterministic-v0','AirRaid-ramDeterministic-v0','Amidar-ramDeterministic-v0',
    #         'Enduro-ramDeterministic-v0','MontezumaRevenge-ramDeterministic-v0','Venture-ramDeterministic-v0']
    # Regarding Atari:
    # Assault, Freeway, Seaquest have different initial states

    file = os.getcwd() + '/' + job
    with open(file,'w') as fp:
        fp.write('#!/bin/sh\n')  
        for i,game in enumerate(games):
            for j,it1 in enumerate(seq1):
                for k,it2 in enumerate(seq2):
                    for l,it3 in enumerate(seq3):
                        fp.write('python3 submit.py --hpsetup game={},{} --hp {}'.format(game,hp_setup,hp))
                        if item1 is not None:
                            fp.write(',{}={}'.format(item1,it1))
                        if item2 is not None:
                            fp.write(',{}={}'.format(item2,it2))
                        if item3 is not None:
                            fp.write(',{}={}'.format(item3,it3))
                        hyperloop_name = make_name('',item1,it1,item2,it2,item3,it3)
                        if hyperloop_name != '':
                            fp.write(',name={}'.format(hyperloop_name))    
                        fp.write('\n')

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', nargs='+',type=str,default=[])
    parser.add_argument('--job', default='job.sh')    
    parser.add_argument('--slurm_mode', default='off')    
    parser.add_argument('--hp', help='Hyperparameter configuration',default='')
    parser.add_argument('--hpsetup', help='Hyperparameter configuration of slurm and hyperparameters and distribution',default='')
    # extra items
    parser.add_argument('--item1',type=str,default=None)
    parser.add_argument('--seq1', nargs='+',type=str,default=[None])
    parser.add_argument('--item2',type=str,default=None)
    parser.add_argument('--seq2', nargs='+',type=str,default=[None])
    parser.add_argument('--item3',type=str,default=None)
    parser.add_argument('--seq3', nargs='+',type=str,default=[None])
    
    args = parser.parse_args()
    
    if args.slurm_mode == 'short':
        args.hpsetup += ',slurm=True,slurm_qos=short,slurm_time=3:59:59'
    elif args.slurm_mode == 'long':
        args.hpsetup += ',slurm=True,slurm_qos=long,slurm_time=5-0:00:00'
        
    expand_job(games=args.games,job=args.job,hp=args.hp,hp_setup=args.hpsetup,
               item1=args.item1,seq1=args.seq1,item2=args.item2,seq2=args.seq2,
               item3=args.item3,seq3=args.seq3)