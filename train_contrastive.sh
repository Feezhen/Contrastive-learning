#!/bin/sh
python train_contrastive.py --seed 44 --batch_size 32 --focal CurricularNCE --gamma 1 --keep_w 0 --gpu 0