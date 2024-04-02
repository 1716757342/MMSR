#!/bin/bash

CUDA_VISIBLE_DEVICES='2,3' python3 -m MMSR generate-dataset \
    --output-dir ./data/train/bivariate \
    --dataset-size 80000 \
    --n-processes 4 \
    --seed 1004 \
    --num-variables 2

# 5678 10000000
# 5679 10000000
# python3 -m MMSR generate-dataset \
#     --output-dir ./data/train/bivariate \
#     --dataset-size 1000000 \
#     --n-processes 128 \
#     --seed 1234 \
#     --num-variables 2