#!/bin/bash

# Expt 1 "Gaussian Kernel @ synthetic"
# Expt 2 "Laplace Kernel @ synthetic"
# Expt 3 "Gaussian Kernel @ CIFAR10"
# Expt 4 "Laplace Kernel @ CIFAR10"
# Expt 5 "Gaussian Kernel @ ImageNet"
# Expt 6 "Laplace Kernel @ ImageNet"

for i in {3..3}; do
  CUDA_VISIBLE_DEVICES=$i nohup uv run python -u expts/monomials.py $i > "nohup$i.out" 2>&1 &
done
