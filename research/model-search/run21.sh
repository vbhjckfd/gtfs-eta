#!/bin/sh
# run 21: fresh later day set ds3 (data 09-07..09-28; train 09-14..09-25, test Sat 09-26 / Sun 09-27 / Mon 09-28, seed 42)
# and its shift twin ds3shift (train 09-14..09-24, test Fri 09-25 / 09-26 / 09-27, seed 7). Arms: baseline, leader,
# and sc_big_hold (V13) now that the route-133 trip ids have 10+ days of history.
D=research/model-search
export RUN=21 DAYS=2026-09-07..2026-09-28 TRAIN_A=2026-09-14..2026-09-25 TEST_A=2026-09-26..2026-09-28 PFX_A=ds3 \
  TRAIN_B=2026-09-14..2026-09-24 TEST_B=2026-09-25..2026-09-27 PFX_B=ds3shift
PREP_DIR=ms_features_v10 TAG=v13 ADDON=addon_v13.py ADDON_DAYS=2026-09-14..2026-09-28 FEAT_DIR=ms_features_v13 \
  MS_SAVE_PRED=data/pred_ds3v13.parquet ARMS_A="baseline sc_big_dwell sc_big_hold" ARMS_B="" sh $D/run.sh
PREP_DIR= FEAT_DIR=ms_features_v13 TAG=v13 ARMS_A="" ARMS_B="baseline sc_big_dwell sc_big_hold" sh $D/run.sh
echo RUN21 ALL DONE
