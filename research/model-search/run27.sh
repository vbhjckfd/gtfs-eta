#!/bin/sh
# run 27: ds4 / ds4shift days (09-08..09-29). Prep with MS_AGE_CORR=1 (link traversals and own speed on the
# GPS fix clock) into ms_features_v27c, then the V26 age cols on top (ms_features_v27). Same arms as run 26;
# compare against ds4v26 / ds4shiftv26 (feed-clock features), the baseline must reproduce 113.95.
D=research/model-search
python $D/pipeline_lite.py --parallel 3 --days 2026-09-08..2026-09-29
echo PIPELINE DONE $(date -u +%H:%M)
export MS_AGE_CORR=1 MS_ADDON_SRC=ms_features_v27c MS_ADDON_DST=ms_features_v27
export RUN=27 DAYS=2026-09-08..2026-09-29 TRAIN_A=2026-09-15..2026-09-26 TEST_A=2026-09-27..2026-09-29 PFX_A=ds4 \
  TRAIN_B=2026-09-15..2026-09-25 TEST_B=2026-09-26..2026-09-28 PFX_B=ds4shift SEED_B=11 TAG=v27
PREP_DIR=ms_features_v27c FEAT_DIR=ms_features_v27 ADDON=addon_v26.py ADDON_DAYS=2026-09-15..2026-09-29 \
  ARMS_A="${ARMS_A:-baseline sc_big_mf3_age sc_big_mf3 sc_big_mf3_age1}" ARMS_B="${ARMS_B:-sc_big_mf3_age}" sh $D/run.sh
echo RUN27 ALL DONE
