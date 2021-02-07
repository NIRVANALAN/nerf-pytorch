#!/bin/bash
srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=75 --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-31 python -u train.py --config $@
#configs/lego.txt
