#!/bin/sh

python3 submit.py --hpsetup game=Pendulum-v0s,name=mcts20lr01,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=20,lr=0.01
python3 submit.py --hpsetup game=Pendulum-v0s,name=mcts50lr01,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=50,lr=0.01
python3 submit.py --hpsetup game=Pendulum-v0s,name=mcts150lr01,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=150,lr=0.01

python3 submit.py --hpsetup game=Pendulum-v0s,name=mcts20lr001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=20,lr=0.001
python3 submit.py --hpsetup game=Pendulum-v0s,name=mcts50lr001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=50,lr=0.001
python3 submit.py --hpsetup game=Pendulum-v0s,name=mcts150lr001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=150,lr=0.001

python3 submit.py --hpsetup game=Pendulum-v0s,name=mcts20lr0001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=20,lr=0.0001
python3 submit.py --hpsetup game=Pendulum-v0s,name=mcts50lr0001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=50,lr=0.0001
python3 submit.py --hpsetup game=Pendulum-v0s,name=mcts150lr0001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=150,lr=0.0001

python3 submit.py --hpsetup game=MountainCarContinuous-v0,name=mcts20lr01,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=20,lr=0.01
python3 submit.py --hpsetup game=MountainCarContinuous-v0,name=mcts50lr01,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=50,lr=0.01
python3 submit.py --hpsetup game=MountainCarContinuous-v0,name=mcts150lr01,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=150,lr=0.01

python3 submit.py --hpsetup game=MountainCarContinuous-v0,name=mcts20lr001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=20,lr=0.001
python3 submit.py --hpsetup game=MountainCarContinuous-v0,name=mcts50lr001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=50,lr=0.001
python3 submit.py --hpsetup game=MountainCarContinuous-v0,name=mcts150lr001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=150,lr=0.001

python3 submit.py --hpsetup game=MountainCarContinuous-v0,name=mcts20lr0001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=20,lr=0.0001
python3 submit.py --hpsetup game=MountainCarContinuous-v0,name=mcts50lr0001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=50,lr=0.0001
python3 submit.py --hpsetup game=MountainCarContinuous-v0,name=mcts150lr0001,item1=c,seq1=1.0+2.5+5.0,item2=loss_type,seq2=count+Q,item3=temp,seq3=0.1+1.0+10.0,n_rep=3,slurm=True,slurm_qos=short,slurm_time=3:59:59 --hp n_eps=10000,n_t=10000000,n_mcts=150,lr=0.0001
