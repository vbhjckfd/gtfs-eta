#!/bin/sh
# run 29: V29 long-window odometer + odometer-vs-shape cols on top of v28 (ds4 / ds4shift, data 09-08..09-29).
# Stage 1 (pipeline + v10 prep + addons v26/v28) is run28.sh with ARMS_A= ARMS_B=.
# Controls: ds4v29 baseline must reproduce 113.95, sc_big_mf3_age_fo 86.28 (ds4v28).
D=research/model-search
export RUN=29 TRAIN_A=2026-09-15..2026-09-26 TEST_A=2026-09-27..2026-09-29 PFX_A=ds4 \
  TRAIN_B=2026-09-15..2026-09-25 TEST_B=2026-09-26..2026-09-28 PFX_B=ds4shift SEED_B=11 TAG=v29
PREP_DIR= FEAT_DIR=ms_features_v29 ADDON=addon_v29.py ADDON_DAYS=2026-09-15..2026-09-29 \
  ARMS_A="${ARMS_A-baseline sc_big_mf3_age_fo sc_big_fo_v29 sc_big_fo_long sc_big_fo_gap}" \
  ARMS_B="${ARMS_B-sc_big_fo_v29}" sh $D/run.sh
echo RUN29 ALL DONE
