#!/bin/bash

# Expt 1 "Gaussian Kernel @ Gaussian data"
# Expt 2 "Laplace Kernel @ Gaussian data"
# Expt 3 "Gaussian Kernel @ CIFAR10"
# Expt 4 "Laplace Kernel @ CIFAR10"

for i in {1..4}; do
  CUDA_VISIBLE_DEVICES=$i nohup uv run python -u expts/learning_curves.py $i monomials > "nohup$i.out" 2>&1 &
done
