#!/bin/sh
# run 25: ds4 / ds4shift (data 09-08..09-29, v10 features): newest days; leader + mf3 confirmation, cheaper-to-serve mf3 variants.
D=research/model-search
export RUN=25 DAYS=2026-09-08..2026-09-29 TRAIN_A=2026-09-15..2026-09-26 TEST_A=2026-09-27..2026-09-29 PFX_A=ds4 \
  TRAIN_B=2026-09-15..2026-09-25 TEST_B=2026-09-26..2026-09-28 PFX_B=ds4shift SEED_B=11 TAG=v25
PREP_DIR=ms_features_v10 ARMS_A="baseline sc_big_dwell sc_big_mf3 sc_big_mf3_fast sc_big_mf3_127" \
  ARMS_B="baseline sc_big_mf3 sc_big_mf3_fast" sh $D/run.sh
echo RUN25 ALL DONE
