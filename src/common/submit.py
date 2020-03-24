# -*- coding: utf-8 -*-
"""
Script to submit jobs
Handles slurm settings, hyperparameter looping, and potential plotting (if not on slurm)
@author: thomas
"""
import os
import argparse
from pprint import pformat
from .hps_setup import get_hps_setup,hps_to_dict,hps_to_list
from .visualize import nested_list,make_name,plot_hyperloop_results

def make_unique_subfolder(folder,hyperloop=False):
    ''' adds a unique four digit subfolder to folder '''
    i = 0
    while os.path.exists(folder + candidate(i,hyperloop)):
        i += 1
    subfolder = folder + candidate(i,hyperloop)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    return subfolder

def candidate(i,hyperloop):
    return '{0:04}h/'.format(i) if hyperloop else '{0:04}/'.format(i)

def submit_slurm(hps,hps_setup,hyperloopname,job_dir,slurmout_dir,ntasks,nodes,n_cpu,mem_per_cpu):
    # make sh file        
    run_file = job_dir + hps.game + hyperloopname + '0.sh'
    
    if hps_setup.distributed:
        base = ' '.join(['mpirun  -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH'
                         '-mca pml ob1 -mca btl ^openib python3 agent.py --hp {} --no_plot']).format(hps_to_list(hps)) # this should become mpirun  
    else:
        base = 'srun python3 agent.py --hp {} --no_plot '.format(hps_to_list(hps)) # this should become mpirun   
            
    with open(run_file,'w') as fp:
        fp.write('#!/bin/sh\n')    
        fp.write(base)
    
    # prepare sbatch command
    my_sbatch = ' '.join(['sbatch --partition=general --qos={} --time={} --ntasks={}',
                '--nodes={} --cpus-per-task={} --mem-per-cpu={} --mail-type=NONE',
                '--output={}slurm-%j.out',
                '--exclude=ess-2',
                '--workdir={}',
                '--job-name={} {}']).format(hps_setup.slurm_qos,hps_setup.slurm_time,ntasks,
                                            nodes,n_cpu,mem_per_cpu,slurmout_dir,
                                            os.getcwd(),hps.game,run_file)
    # run
    os.system('chmod +x {}'.format(run_file))
    return_val = os.system(my_sbatch)
    if return_val != 0:
        raise ValueError('submission went wrong')   

def submit(hp_string,hpsetup_string,no_plot,agent,get_hps,override_hps_settings):
    hps = get_hps().parse(hp_string)    
    hps_setup = get_hps_setup().parse(hpsetup_string,hps)  
    # override game and name from hyperlooping
    if hps_setup.game != 'None':
        hps.game = hps_setup.game
    if hps_setup.name != 'None':
        hps.name = hps_setup.name
    
    # set-up base result folder
    result_folder = os.getcwd() + '/results/{}/{}/'.format(hps.name,hps.game)

    # check whether we should be hyperlooping
    loop_hyper = True if (hps_setup.item1 is not None or hps_setup.item2 is not None or hps_setup.item3 is not None) else False

    # make the unique subfolder
    subfolder = make_unique_subfolder(result_folder,loop_hyper)

    # Write hyperparameters in nice format   
    with open(subfolder + 'hps_setup.txt','w') as file:
        file.write(pformat(hps_to_dict(hps_setup)))
    with open(subfolder + 'hps.txt','w') as file:
        file.write(pformat(hps_to_dict(hps)))
    with open(subfolder + 'hps_setup_raw.txt','w') as file:
                file.write(hps_to_list(hps_setup))
    
    if not hps_setup.slurm:
        # for automatically plotting results if not on slurm
        n1,n2,n3 = len(hps_setup.seq1),len(hps_setup.seq2),len(hps_setup.seq3)
        results = nested_list(n1,n2,n3,hps_setup.n_rep) # handle plotting within this call, so agregate results
    else:
        no_plot = True
        # prepare slurm submission folders
        job_dir = os.getcwd() + '/results/jobs/'
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        slurmout_dir = os.getcwd() + '/results/slurmout/'
        if not os.path.exists(slurmout_dir):
            os.makedirs(slurmout_dir)  
        # some slurm settings initialization due to the specific Delft slurm cluster    
        if hps_setup.distributed:
            ntasks = hps_setup.n_tasks
            nodes = '1-3'
            n_cpu = hps_setup.cpu_per_task   
            mem_per_cpu = int((16384/(ntasks*n_cpu)) - 5)
        else:
            ntasks = 1
            nodes = 1
            n_cpu = hps_setup.cpu_per_task   
            mem_per_cpu = hps_setup.mem_per_cpu

    for rep in range(hps_setup.n_rep): 
        hps.rep = rep          
        for it1,item1 in enumerate(hps_setup.seq1):
            if hps_setup.item1 is not None: 
                hps._set(hps_setup.item1,item1)
            for it2,item2 in enumerate(hps_setup.seq2):
                if hps_setup.item2 is not None: 
                    hps._set(hps_setup.item2,item2)
                for it3,item3 in enumerate(hps_setup.seq3):
                    if hps_setup.item3 is not None: 
                        hps._set(hps_setup.item3,item3)
                    hyperloop_name = make_name('',hps_setup.item1,item1,hps_setup.item2,item2,
                                               hps_setup.item3,item3)
                    # if loop_hyper:
                    result_folder = subfolder + hyperloop_name                 
                    hps.result_dir = result_folder + 'rep:{}'.format(rep)   
                    
                    hps = override_hps_settings(hps) # maybe some hps_setup parameter overrides a number of hps parameters

                    # Submit slurm job or launch agent in this process
                    if hps_setup.slurm:
                        submit_slurm(hps,hps_setup,hyperloop_name,job_dir,slurmout_dir,ntasks,nodes,n_cpu,mem_per_cpu)  
                    else:
                        print(' ________________________________________ ')     
                        print('Start learning on game {} with hyperparams {}'.format(hps.game,hyperloop_name)) 
                        curve = agent(hps)   
                        results[it1][it2][it3][rep] = curve
                        
    if not no_plot:
        plot_hyperloop_results(results,hps_setup,subfolder,plot_type='mean',sd=True)        
        
if __name__ == "__main__":   
    '''Set-up training'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp', help='Hyperparameter configuration',default='')
    parser.add_argument('--hpsetup', help='Hyperparameter configuration of slurm and hyperparameters and distribution',default='')
    parser.add_argument('--no_plot', action='store_true',default=False)
    args = parser.parse_args()