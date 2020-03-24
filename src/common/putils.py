#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python utils
@author: thomas
"""
import time
import os
import numpy as np
import random
from shutil import copyfile

def timing(f):
    ''' function decorator '''
    def wrap(*args,**kwargs):
        time1 = time.time()
        ret = f(*args,**kwargs)
        time2 = time.time()
        print('{} function took {} sec'.format(f.__name__,time2-time1))
        return ret
    return wrap

def store_safely(folder,name,to_store):
    ''' to prevent losing information due to interruption of process'''
    new_name = folder+name+'.npy'
    old_name = folder+name+'_old.npy'
    if os.path.exists(new_name):
        copyfile(new_name,old_name)
    np.save(new_name,to_store)
    if os.path.exists(old_name):            
        os.remove(old_name)

def my_argmax(x):
    ''' assumes a 1D vector x '''
    x = x.flatten()
    if np.any(np.isnan(x)):
        print('Warning: Cannot argmax when vector contains nans, results will be wrong')
    try:
        winners = np.argwhere(x == np.max(x)).flatten()   
        winner = random.choice(winners)
    except:
        winner = np.argmax(x) # numerical instability ? 
    return winner 