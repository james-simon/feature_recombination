#!/bin/bash

# Expt 0 "Gaussian Kernel @ CIFAR10 Vehicle vs Animal"
# Expt 1 "Gaussian Kernel @ CIFAR10 Domesticated vs Wild"
# Expt 2 "Gaussian Kernel @ SVHN Odd vs Even"
# Expt 3 "Gaussian Kernel @ SVHN Looped numerals vs Others"
# Expt 4 "Laplace Kernel @ CIFAR10 Vehicle vs Animal"
# Expt 5 "Laplace Kernel @ CIFAR10 Domesticated vs Wild"
# Expt 6 "Laplace Kernel @ SVHN Odd vs Even"
# Expt 7 "Laplace Kernel @ SVHN Looped numerals vs Others"

declare -a KERNELS=("Gaussian" "Gaussian" "Gaussian" "Gaussian" "Laplace" "Laplace" "Laplace" "Laplace")
declare -a DATASETS=("CIFAR10" "CIFAR10" "SVHN" "SVHN" "CIFAR10" "CIFAR10" "SVHN" "SVHN")
declare -a TASKS=("vehicle" "domesticated" "evenodd" "loops" "vehicle" "domesticated" "evenodd" "loops")
declare -a ARGS=(3 3 5 5 4 4 6 6)

for i in {1..8}; do
  echo "${KERNELS[$((i-1))]} Kernel @ ${DATASETS[$((i-1))]} ${TASKS[$((i-1))]//_/ }"
  CUDA_VISIBLE_DEVICES=$((i%7)) nohup uv run python -u expts/hehe_eigenlearning.py ${ARGS[$((i-1))]} ${TASKS[$((i-1))]} > "nohup$i.out" 2>&1 &
done
