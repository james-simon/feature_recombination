#!/bin/bash

for i in 13; do #{11..16}; do
  CUDA_VISIBLE_DEVICES=$(((i % 6)+1)) nohup uv run python -u expts/learning_curves.py $i > "nohup$i.out" 2>&1 &
done
