#!/bin/sh
python3 submit.py --hpsetup game=Ant-v2,item1=n_mcts,seq1=50,item2=entropy_l,seq2=0.01,item3=temp,seq3=0.1+1.0+10.0,n_rep=3 --hp c=0.1,bound=beta,n_t=10000000,n_ep=20000,lr=0.01,loss_type=Q,name=lr:0.01-loss_type:Q
python3 submit.py --hpsetup game=Ant-v2,item1=n_mcts,seq1=50,item2=entropy_l,seq2=0.01,item3=temp,seq3=0.1+1.0+10.0,n_rep=3 --hp c=0.1,bound=beta,n_t=10000000,n_ep=20000,lr=0.01,loss_type=count,name=lr:0.01-loss_type:count
python3 submit.py --hpsetup game=Ant-v2,item1=n_mcts,seq1=50,item2=entropy_l,seq2=0.01,item3=temp,seq3=0.1+1.0+10.0,n_rep=3 --hp c=0.1,bound=beta,n_t=10000000,n_ep=20000,lr=0.001,loss_type=Q,name=lr:0.001-loss_type:Q
python3 submit.py --hpsetup game=Ant-v2,item1=n_mcts,seq1=50,item2=entropy_l,seq2=0.01,item3=temp,seq3=0.1+1.0+10.0,n_rep=3 --hp c=0.1,bound=beta,n_t=10000000,n_ep=20000,lr=0.001,loss_type=count,name=lr:0.001-loss_type:count
python3 submit.py --hpsetup game=Ant-v2,item1=n_mcts,seq1=50,item2=entropy_l,seq2=0.01,item3=temp,seq3=0.1+1.0+10.0,n_rep=3 --hp c=0.1,bound=beta,n_t=10000000,n_ep=20000,lr=0.0001,loss_type=Q,name=lr:0.0001-loss_type:Q
python3 submit.py --hpsetup game=Ant-v2,item1=n_mcts,seq1=50,item2=entropy_l,seq2=0.01,item3=temp,seq3=0.1+1.0+10.0,n_rep=3 --hp c=0.1,bound=beta,n_t=10000000,n_ep=20000,lr=0.0001,loss_type=count,name=lr:0.0001-loss_type:count
