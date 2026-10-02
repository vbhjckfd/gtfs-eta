#!/bin/sh
# run 30: V30 "vehicles on the path now" cols on top of v29, + serving ablations of sc_big_fo_long (ds4 / ds4shift, data 09-08..09-29).
# Stage 1 (pipeline + v10 prep + addons v26/v28) is `ARMS_A=" " ARMS_B=" " sh run28.sh`.
# Controls: ds4v30 baseline must reproduce 113.95, sc_big_fo_long 85.65 (ds4v29).
D=research/model-search
export RUN=30 TRAIN_A=2026-09-15..2026-09-26 TEST_A=2026-09-27..2026-09-29 PFX_A=ds4 \
  TRAIN_B=2026-09-15..2026-09-25 TEST_B=2026-09-26..2026-09-28 PFX_B=ds4shift SEED_B=11 TAG=v30
PREP_DIR= FEAT_DIR=ms_features_v30 ADDON="addon_v29.py addon_v30.py" ADDON_DAYS=2026-09-15..2026-09-29 \
  ARMS_A="${ARMS_A-baseline sc_big_fo_long sc_big_fo_pa sc_big_fo_novl sc_big_fo_nolap sc_big_fo_noveh}" \
  ARMS_B="${ARMS_B-sc_big_fo_pa sc_big_fo_novl}" sh $D/run.sh
echo RUN30 ALL DONE
