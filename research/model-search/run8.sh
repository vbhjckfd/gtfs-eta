#!/bin/sh
# run 8 queue (MS_FEAT_DIR=ms_features_v4 with V6 cols)
export MS_FEAT_DIR=ms_features_v4
Q=research/model-search/queue.sh
H=research/model-search/harness.py
sh $Q ds1v6 2026-09-07..2026-09-18 2026-09-19..2026-09-21 42 baseline dtcat_stopcat_big sc_big_nxt sc_big_cur sc_big_v6 sc_big_lr08
sh $Q ds1shiftv6 2026-09-07..2026-09-17 2026-09-18..2026-09-20 7 baseline dtcat_stopcat_big sc_big_v6 sc_big_nxt sc_big_cur
echo RUN8 DONE
