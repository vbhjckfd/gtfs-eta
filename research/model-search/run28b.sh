#!/bin/sh
# run 28 part 2: V28b cols on top of v28 (data from run28.sh must be on disk), arm sc_big_mf3_age_fo2 on both splits.
D=research/model-search
python $D/addon_v28b.py --days 2026-09-15..2026-09-29 || exit 1
echo ADDON28B DONE
export RUN=28 TRAIN_A=2026-09-15..2026-09-26 TEST_A=2026-09-27..2026-09-29 PFX_A=ds4 \
  TRAIN_B=2026-09-15..2026-09-25 TEST_B=2026-09-26..2026-09-28 PFX_B=ds4shift SEED_B=11 TAG=v28b
PREP_DIR= FEAT_DIR=ms_features_v28b ADDON= ARMS_A="${ARMS_A:-sc_big_mf3_age_fo2}" ARMS_B="${ARMS_B:-sc_big_mf3_age_fo2}" sh $D/run.sh
echo RUN28B ALL DONE
