#!/bin/bash

for i in {1..4}; do
  CUDA_VISIBLE_DEVICES=$i nohup uv run python -u expts/hermite_eigenstructure.py $i > "nohup$i.out" 2>&1 &
done
