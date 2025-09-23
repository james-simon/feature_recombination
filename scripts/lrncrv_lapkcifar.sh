#!/bin/bash

for i in {17..20}; do
  CUDA_VISIBLE_DEVICES=$(((i % 4)+1)) nohup uv run python -u expts/learning_curves.py $i > "nohup$i.out" 2>&1 &
done
