#!/bin/sh
# run 26: ds4 / ds4shift days again (09-08..09-29) so baseline / mf3 reproduce run 25; adds the
# V26 position-age cols (addon_v26.py) on top of v10 features.
D=research/model-search
python $D/pipeline_lite.py --parallel 3 --days 2026-09-08..2026-09-29
echo PIPELINE DONE $(date -u +%H:%M)
export RUN=26 DAYS=2026-09-08..2026-09-29 TRAIN_A=2026-09-15..2026-09-26 TEST_A=2026-09-27..2026-09-29 PFX_A=ds4 \
  TRAIN_B=2026-09-15..2026-09-25 TEST_B=2026-09-26..2026-09-28 PFX_B=ds4shift SEED_B=11 TAG=v26
PREP_DIR=ms_features_v10 FEAT_DIR=ms_features_v26 ADDON=addon_v26.py ADDON_DAYS=2026-09-15..2026-09-29 \
  ARMS_A="${ARMS_A:-baseline sc_big_mf3 sc_big_mf3_age}" ARMS_B="${ARMS_B:-sc_big_mf3_age}" sh $D/run.sh
echo RUN26 ALL DONE
