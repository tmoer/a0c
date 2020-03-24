    # -*- coding: utf-8 -*-
"""
Default hyperparameter settings
@author: thomas
"""
from common.hps_setup import HParams

def override_hps_settings(hps):
    ''' some more global modifications to multiple settings based on 1 indicator '''
    if hps.mode == 'off':
        pass
    return hps

def get_hps():
    ''' Hyperparameter settings '''
    return HParams(      
        # General
        game = 'MountainCarContinuous-v0', # Environment name
        name = 'unnamed', # Name of experiment
        result_dir = '',

        # Steps & limits
        n_t = 2000, # max timesteps
        n_eps = 100, # max episodes
        steps_per_ep = 300,

        mode = 'off', # overall indicator to jointly change a group of settings. Use with override_hps_settings()
        
        # MCTS
        n_mcts = 10,
        c = 1.0,
        alpha = 0.5,
        C_widening = 1.0,
        decision_type = 'count',        
        backup_Q = 'on-policy', # 'on-policy', 'max' or 'thompson': Type of policy used for value back-up. Thopmpson requires additional sampling
        sigma_tree = False, # whether to use tree uncertainty
        backup_sigma_tree = 'on-policy', # 'uniform', 'on-policy', 'max', 'thompson': policy used for sigma_tree back-up
        block_loop = False, # Whether to block loops

        # MCTS + DL
        loss_type = 'count', # 'count' or 'Q'
        bound = 'beta', # 'tanh' or 'beta'
        entropy_l = 0.0,
        random_action_frac = 0.0,
        temp = 1.0,
        n_mix = 1,
        use_prior = False,
        bootstrap_V = True,
        V_decision = 'on_policy',

        # Train
        lr = 0.005,        
        n_epochs = 1,
        batch_size = 32,
        data_size = 5000, # total database, if distributed summed over the agents
        clip_gradient_norm = 0.0,
        tfdb = False,

        # Other
        timeit = False,
        verbose = False,
        verbose_mcts = False
        )