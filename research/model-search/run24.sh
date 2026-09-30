#!/bin/sh
# run 24: ds3 / ds3shift (data 09-07..09-28, v10 features). max_features 0.4 / 0.3 and mf5 + min_samples_leaf 20;
# baseline + mf5 refit to check the rebuild reproduces run 23.
D=research/model-search
export RUN=24 DAYS=2026-09-07..2026-09-28 TRAIN_A=2026-09-14..2026-09-25 TEST_A=2026-09-26..2026-09-28 PFX_A=ds3 \
  TRAIN_B=2026-09-14..2026-09-24 TEST_B=2026-09-25..2026-09-27 PFX_B=ds3shift TAG=v13
PREP_DIR=ms_features_v10 ARMS_A="baseline sc_big_mf4 sc_big_mf3" ARMS_B="sc_big_mf4 sc_big_mf3" sh $D/run.sh
PREP_DIR= FEAT_DIR=ms_features_v10 ARMS_A="sc_big_mf5_msl20 sc_big_mf5" ARMS_B="sc_big_mf5_msl20" sh $D/run.sh
echo RUN24 ALL DONE
# run 24, second batch (mf3 best so far): max_features 0.2
PREP_DIR= FEAT_DIR=ms_features_v10 ARMS_A="sc_big_mf2" ARMS_B="sc_big_mf2" sh $D/run.sh
