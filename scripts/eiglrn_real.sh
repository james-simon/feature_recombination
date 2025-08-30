#!/bin/bash

# Expt 1 "Gaussian Kernel @ CIFAR10"
# Expt 2 "Laplace Kernel @ CIFAR10"
# Expt 3 "Gaussian Kernel @ SVHN"
# Expt 4 "Laplace Kernel @ SVHN"

for i in {1..4}; do
  CUDA_VISIBLE_DEVICES=$i nohup uv run python -u expts/hehe_eigenlearning.py $((2+i)) original > "nohup$i.out" 2>&1 &
done

# CUDA_VISIBLE_DEVICES=1 nohup uv run python -u expts/hehe_eigenlearning.py 3 original > "nohup1.out" 2>&1 &