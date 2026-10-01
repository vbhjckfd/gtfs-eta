#!/bin/sh
# run 28: ds4 / ds4shift days (09-08..09-29). Feed-clock prep (ms_features_v10) -> V26 age cols (v26)
# -> V28 feed-speed / odometer cols (v28). Arms: the feed-clock pos_age_s-only ablation (run 27 next
# step 2) and the V28 arms. Baseline must reproduce 113.95, mf3_age 87.22 (ds4v26).
D=research/model-search
python $D/pipeline_lite.py --parallel 3 --days 2026-09-08..2026-09-29
echo PIPELINE DONE $(date -u +%H:%M)
export RUN=28 DAYS=2026-09-08..2026-09-29 TRAIN_A=2026-09-15..2026-09-26 TEST_A=2026-09-27..2026-09-29 PFX_A=ds4 \
  TRAIN_B=2026-09-15..2026-09-25 TEST_B=2026-09-26..2026-09-28 PFX_B=ds4shift SEED_B=11 TAG=v28
PREP_DIR=ms_features_v10 FEAT_DIR=ms_features_v28 ADDON="addon_v26.py addon_v28.py" ADDON_DAYS=2026-09-15..2026-09-29 \
  ARMS_A="${ARMS_A:-baseline sc_big_mf3_age sc_big_mf3_age_fo sc_big_mf3_age1 sc_big_mf3_age_spd sc_big_mf3_age_odo}" \
  ARMS_B="${ARMS_B:-sc_big_mf3_age_fo sc_big_mf3_age1}" sh $D/run.sh
echo RUN28 ALL DONE
