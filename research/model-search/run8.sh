#!/bin/sh
# run 8 queue (MS_FEAT_DIR=ms_features_v4 with V6 cols)
export MS_FEAT_DIR=ms_features_v4
Q=research/model-search/queue.sh
H=research/model-search/harness.py
sh $Q ds1v6 2026-09-07..2026-09-18 2026-09-19..2026-09-21 42 baseline dtcat_stopcat_big sc_big_nxt sc_big_cur sc_big_v6
sh $Q ds1shiftv6 2026-09-07..2026-09-17 2026-09-18..2026-09-20 7 baseline dtcat_stopcat_big sc_big_v6 sc_big_nxt sc_big_cur
# 2x training rows (2% of snapshots/day instead of 1%)
for a in baseline dtcat_stopcat_big; do
  python $H run --arm $a --train 2026-09-07..2026-09-18 --test 2026-09-19..2026-09-21 --seed 42 --tag ds1v6k2 --keep-pct 2
done
echo RUN8 DONE
