# -*- coding: utf-8 -*-
"""
Chain experiments
@author: thomas
"""

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

global mpl
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import os
import time
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import argparse
from pprint import pformat
#from pdb import set_trace

# common package import
from src.common.rl.make_game import make_game
from src.common.submit import make_unique_subfolder
from src.common.hps_setup import hps_to_dict
from src.common.visualize import plot_single_experiment
from src.common.putils import store_safely

# local imports
from config.hps import get_hps,override_hps_settings
from src.mcts import MCTS,display_info
from src.network import Model,Database

def agent(hps):
    ''' Agent function '''
    tf.reset_default_graph()
    
    # storage
    result = {}
    env_steps,ep_return = [],[] # will indicate the timestep for the learning curve
    losses,gn = [],[]
    best_R = -np.Inf    
               
    Env = make_game(hps.game)
    D = Database(max_size=max(hps.data_size,hps.n_mcts*hps.steps_per_ep),batch_size=hps.batch_size)        
    model = Model(Env,lr=hps.lr,n_mix=hps.n_mix,clip_gradient_norm=hps.clip_gradient_norm,loss_type=hps.loss_type,
                  bound=hps.bound,temp=hps.temp,entropy_l=hps.entropy_l)        

    #with tf.Session() as sess,sess.as_default():
    with tf.Session() as sess:
        if hps.tfdb:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        model.sess = sess
        sess.run(tf.global_variables_initializer())
        global_t_mcts = 0
        global_t = 0 
        
        for ep in range(hps.n_eps):
            start = time.time()
            root_index = Env.reset() 
            root = None
            R = 0.0 # episode reward
            t = 0 # episode steps        
            seed = np.random.randint(1e7) 
            Env.seed(seed)                                   
            a_store = []

            while True:
                # run an episode
                if hps.timeit: now = time.time()
                root = MCTS(root_index,root,Env,N=hps.n_mcts,model=model,c=hps.c,bootstrap_V=hps.bootstrap_V,
                            block_loop=hps.block_loop,sigma_tree=hps.sigma_tree,backup_Q=hps.backup_Q,
                            backup_sigma_tree=hps.backup_sigma_tree,seed=seed,a_his=a_store,
                            alpha=hps.alpha,C_widening=hps.C_widening,use_prior=hps.use_prior,timeit=hps.timeit,
                            random_action_frac=hps.random_action_frac)
                if hps.timeit: print('One MCTS search takes {} seconds'.format(time.time()-now))                    
                if hps.verbose_mcts: display_info(root,'{}'.format(t),hps.c)
                    
                probs,a_list,V,a,a_argmax = root.return_results(decision_type=hps.decision_type,loss_type=hps.loss_type,
                                                              temperature=hps.temp,V_decision=hps.V_decision)
                for k,prob in enumerate(probs):
                    D.store((root.index,V,a_list[k],np.array([prob])))
                    #if count == 0:
                    #    print('Warning',[child_action.n for child_action in root.child_actions],display_info(root,'{}'.format(t),hps.c))
                        
                # Make the step
                a_store.append(a)
                s1,r,terminal,_ = Env.step(a)
                R += r
                t += 1
                global_t += 1
                global_t_mcts += hps.n_mcts
    
                #if hps.verbose:
                #    if (t % 50) == 0: 
                #        print('Overall step {}, root currently returns V {}, and considers a {} with counts {}'.format(global_t,V,a_list,probs))
                        
                if terminal or (t > hps.steps_per_ep):
                    if hps.verbose:
                        print('Episode terminal, total reward {}, steps {}'.format(R,t))
                    ep_return.append(R)
                    env_steps.append(global_t_mcts)
                    break # break out, start new episode
                else:
                    root = root.forward(a_argmax,s1,r,terminal,model)

            # saving
            result.update({'steps':env_steps,'return':ep_return})
            if hps.verbose:
                result.update({'gn':gn,'loss':losses})
            #if R > best_R:
            #    result.update({'seed':seed,'actions':a_store,'R':best_R})
            #    best_R = R
            store_safely(hps.result_dir,'result',result)  

            # Train 
            if (global_t_mcts > hps.n_t) or (ep > hps.n_eps):
                break # end learning
            else:
                n_epochs = hps.n_epochs * (np.ceil(hps.n_mcts/20)).astype(int)
                #print(n_epochs)
                loss = model.train(D,n_epochs,hps.lr)
                losses.append(loss['total_loss'])
                gn.append(loss['gn'])
            
            if hps.verbose:            
                print('Time {}, Episode {}, Return {}, V {}, gn {}, Vloss {}, piloss {}'.format(
                    global_t_mcts,ep,R,loss['V'],loss['gn'],loss['V_loss'],loss['pi_loss']))
                print('Actions {}, probs {}'.format(np.array(a_list),probs))
                print('One full episode loop + training in {} seconds'.format(time.time()-start))
        
    return result

if __name__ == '__main__':
    '''Set-up training'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp', help='Hyperparameter configuration',default='')
    parser.add_argument('--no_plot', action='store_true',default=False)
    args = parser.parse_args()
    hps = get_hps().parse(args.hp)    
    hps = override_hps_settings(hps)

    # set-up result folder if not prespecified, then we are not hyperlooping
    if hps.result_dir == '': 
        result_folder = os.getcwd() + '/results/{}/{}/'.format(hps.name,hps.game)
        hps.result_dir = make_unique_subfolder(result_folder,hyperloop=False)
        with open(hps.result_dir + 'hps.txt','w') as file:
            file.write(pformat(hps_to_dict(hps)))

    #with open(subfolder + 'hps_raw.txt','w') as file:
    #    file.write(hps_to_list(hps)) 
    print(' ________________________________________ ')     
    print('Start learning on game {}'.format(hps.game))               
    result = agent(hps)
    
    if not args.no_plot:
        plot_single_experiment(result,hps.game,hps.result_dir,plot_type='lc')