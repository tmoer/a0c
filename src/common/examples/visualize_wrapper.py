#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for visualize.py

@author: thomas
"""

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
import argparse
from common.visualize import loop_directories
from hps import get_hps
   
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--folder', default='/home/thomas/mcts_results')
    parser.add_argument('--home', action='store_true',default=False)
    parser.add_argument('--game', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--subindex', default=None)
    parser.add_argument('--plot_type', default='lc')
    parser.add_argument('--window', type=int,default=25)
    parser.add_argument('--sd', action='store_true',default=False)
    parser.add_argument('--on_mean', action='store_true',default=False)
    parser.add_argument('--item', default='return',help='This item in result will be plotted')
    parser.add_argument('--remove', action='store_true',default=False)
    parser.add_argument('--plotlen', type=int,default=25)
    parser.add_argument('--xlim', nargs='+',type=float,default=None)
    parser.add_argument('--ylim', nargs='+',type=float,default=None)
    parser.add_argument('--errlim', nargs='+',type=float,default=None,help='Limits on the errorbars')    
    parser.add_argument('--item1_label', nargs='+', default=None)
    parser.add_argument('--item2_label', nargs='+', default=None)
    parser.add_argument('--item3_label', nargs='+', default=None)

    args = parser.parse_args()
    
    if args.home:
        result_folder = os.getcwd() + '/results/'
    else:
        result_folder = args.folder + '/'
    print('Start processing folder {}'.format(result_folder))
    overview_dir= result_folder+'learning_curves/'
    if not os.path.exists(overview_dir):
        os.makedirs(overview_dir)
        
    loop_directories(result_folder=result_folder,overview_dir=overview_dir,game=args.game,name=args.name,
                     subindex=args.subindex,plot_type=args.plot_type,window=args.window,sd=args.sd,on_mean=args.on_mean,
                     item=args.item,remove=args.remove,plotlen=args.plotlen,xlim=args.xlim,ylim=args.ylim,errlim=args.errlim,
                     get_hps=get_hps) 

