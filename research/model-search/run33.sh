#!/bin/sh
# run 33: stacked optional add-ons + pace target, same V32 cols (ds4 / ds4shift, data 09-08..09-29).
# Pipeline: `python research/model-search/pipeline_lite.py --parallel 3 --days 2026-09-08..2026-09-29` (started separately; run.sh waits per day).
# Controls: ds4v32 baseline must reproduce 113.95, sc_big_fo_duty 84.77 (ds4v32).
D=research/model-search
export RUN=33 DAYS=2026-09-08..2026-09-29 TRAIN_A=2026-09-15..2026-09-26 TEST_A=2026-09-27..2026-09-29 PFX_A=ds4 \
  TRAIN_B=2026-09-15..2026-09-25 TEST_B=2026-09-26..2026-09-28 PFX_B=ds4shift SEED_B=11 TAG=v33
PREP_DIR=ms_features_v10 FEAT_DIR=ms_features_v32 ADDON="addon_v26.py addon_v28.py addon_v29.py addon_v30.py addon_v31.py addon_v32.py" \
  ADDON_DAYS=2026-09-15..2026-09-29 \
  ARMS_A="${ARMS_A-baseline sc_big_fo_duty sc_big_duty_pace sc_big_fo_all}" \
  ARMS_B="${ARMS_B-sc_big_fo_duty sc_big_duty_pace sc_big_fo_all}" sh $D/run.sh
echo RUN33 ALL DONE
