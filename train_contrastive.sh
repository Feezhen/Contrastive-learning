#!/bin/sh

python train_contrastive.py --seed 44 --batch_size 32 --focal --gamma 1 --keep_w 0.1 --gpu 0