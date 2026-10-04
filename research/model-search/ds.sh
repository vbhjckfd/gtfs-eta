#!/bin/sh
# One runner for a full day set (replaces run28..run33.sh, run 35 housekeeping).
# Defaults = ds4 / ds4shift (data 09-08..09-29), current feature chain (ms_features_v32) and the hand-off arms.
#   sh research/model-search/ds.sh                         # pipeline + prep + addons + fits (≈ 2.5 h + fits)
#   PIPE=0 ARMS_A="sc_big_fo_duty" ARMS_B=" " TAG=v35 sh research/model-search/ds.sh
# New day set (e.g. ds5): override DAYS, ADDON_DAYS (= first train day .. last test day), TRAIN_A/TEST_A/PFX_A, TRAIN_B/TEST_B/PFX_B.
# Controls on ds4: baseline 113.946, sc_big_fo_long 85.65, sc_big_fo_pa 85.197, sc_big_fo_duty 84.770 (ds4shift duty 83.623).
D=research/model-search
export DAYS=${DAYS:-2026-09-08..2026-09-29}
if [ "${PIPE:-1}" = 1 ]; then
  python $D/pipeline_lite.py --parallel 3 --days $DAYS &   # run.sh preps each day as it lands
fi
export RUN=${RUN:-ds} TAG=${TAG:-v35} \
  TRAIN_A=${TRAIN_A:-2026-09-15..2026-09-26} TEST_A=${TEST_A:-2026-09-27..2026-09-29} PFX_A=${PFX_A:-ds4} SEED_A=${SEED_A:-42} \
  TRAIN_B=${TRAIN_B:-2026-09-15..2026-09-25} TEST_B=${TEST_B:-2026-09-26..2026-09-28} PFX_B=${PFX_B:-ds4shift} SEED_B=${SEED_B:-11} \
  PREP_DIR=${PREP_DIR-ms_features_v10} FEAT_DIR=${FEAT_DIR:-ms_features_v32} \
  ADDON="${ADDON-addon_v26.py addon_v28.py addon_v29.py addon_v30.py addon_v31.py addon_v32.py}" \
  ADDON_DAYS=${ADDON_DAYS:-2026-09-15..2026-09-29} \
  ARMS_A="${ARMS_A-baseline sc_big_fo_long sc_big_fo_pa sc_big_fo_duty}" \
  ARMS_B="${ARMS_B-baseline sc_big_fo_long sc_big_fo_duty}"
sh $D/run.sh
wait
echo DS ALL DONE
