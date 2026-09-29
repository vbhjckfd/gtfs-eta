#!/bin/sh
# run 22: finish run 21 on the ds3 day set (data 09-07..09-28). ds3 = train 09-14..09-25, test Sat 09-26 / Sun 09-27 /
# Mon 09-28, seed 42; ds3shift = train 09-14..09-24, test Fri 09-25 / 09-26 / 09-27, seed 7.
# Baseline on ds3 is refit to check the rebuild reproduces run 21 bit-for-bit.
D=research/model-search
export RUN=22 DAYS=2026-09-07..2026-09-28 TRAIN_A=2026-09-14..2026-09-25 TEST_A=2026-09-26..2026-09-28 PFX_A=ds3 \
  TRAIN_B=2026-09-14..2026-09-24 TEST_B=2026-09-25..2026-09-27 PFX_B=ds3shift
PREP_DIR=ms_features_v10 TAG=v13 ADDON=addon_v13.py ADDON_DAYS=2026-09-14..2026-09-28 FEAT_DIR=ms_features_v13 \
  ARMS_A="baseline sc_big_hold" ARMS_B="baseline sc_big_dwell sc_big_hold" sh $D/run.sh
echo RUN22 ALL DONE
