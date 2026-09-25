#!/bin/sh
# Run a list of arms sequentially on one split: queue.sh TAG TRAIN TEST SEED arm...
tag=$1 train=$2 test=$3 seed=$4; shift 4
for a in "$@"; do
  python research/model-search/harness.py run --arm "$a" --train "$train" --test "$test" --seed "$seed" --tag "$tag"
done
