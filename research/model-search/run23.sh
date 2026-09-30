#!/bin/sh
# run 23: ds3 / ds3shift (data 09-07..09-28, v10 features; tag ds3v13 as in runs 21-22, the arms here don't use V13 cols).
# Arms: max_features 0.7 (sc_big_mf7) and a 2-seed bag of it (sc_big_bag2). Baseline + leader are refit on ds3 to check
# the rebuild reproduces run 22, and the leader's test predictions are saved for the route-137 diagnostic.
D=research/model-search
export RUN=23 DAYS=2026-09-07..2026-09-28 TRAIN_A=2026-09-14..2026-09-25 TEST_A=2026-09-26..2026-09-28 PFX_A=ds3 \
  TRAIN_B=2026-09-14..2026-09-24 TEST_B=2026-09-25..2026-09-27 PFX_B=ds3shift TAG=v13
PREP_DIR=ms_features_v10 MS_SAVE_PRED=data/pred_ds3_leader.parquet ARMS_A="sc_big_dwell" ARMS_B="" sh $D/run.sh
PREP_DIR= FEAT_DIR=ms_features_v10 ARMS_A="baseline sc_big_mf7 sc_big_bag2" ARMS_B="sc_big_mf7 sc_big_bag2" sh $D/run.sh
echo RUN23 ALL DONE
