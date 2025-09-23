#!/bin/bash

for i in {1..7}; do
  CUDA_VISIBLE_DEVICES=$(((i % 7)+1)) nohup uv run python -u expts/learning_curves.py $i > "nohup$i.out" 2>&1 &
done
