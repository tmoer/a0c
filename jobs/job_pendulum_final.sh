#!/bin/sh
python3 submit.py --hpsetup game=Pendulum-v0s,item1=n_mcts,seq1=1+5+10+25+50+100,item2=c,seq2=0.005+0.05,item3=lr,seq3=0.001+0.0001+0.00001,n_rep=10,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp bound=beta,n_t=20000000000,n_eps=50000000,V_decision=max,clip_gradient_norm=1.0,temp=0.1,entropy_l=0.1
