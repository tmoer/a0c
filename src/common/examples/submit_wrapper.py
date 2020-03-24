#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for submit function

@author: thomas
"""

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import argparse
from common.submit import submit
from hps import get_hps,override_hps_settings
from agent import agent        
        
if __name__ == "__main__":   
    '''Set-up training'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp', help='Hyperparameter configuration',default='')
    parser.add_argument('--hpsetup', help='Hyperparameter configuration of slurm and hyperparameters and distribution',default='')
    parser.add_argument('--no_plot', action='store_true',default=False)
    args = parser.parse_args()
    submit(args.hp,args.hpsetup,args.no_plot,agent,get_hps,override_hps_settings)