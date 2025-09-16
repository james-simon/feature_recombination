#!/bin/bash

for i in {6..6}; do
  CUDA_VISIBLE_DEVICES=$(((i % 7)+1)) nohup uv run python -u expts/learning_curves.py $i > "nohup$i.out" 2>&1 &
done

# CUDA_VISIBLE_DEVICES=1 nohup uv run python -u expts/learning_curves.py 9 > "nohup9.out" 2>&1 &