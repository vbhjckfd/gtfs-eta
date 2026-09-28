#!/bin/sh
# run 20 add-on: V13 trip-keyed hold cols (addon_v13.py -> ms_features_v13), fits on ds1v13 / ds1shiftv13.
D=research/model-search
while pgrep -f addon_v13.py >/dev/null; do sleep 30; done
python $D/addon_v13.py --days 2026-09-07..2026-09-21 || exit 1   # no-op for days already built
MS_SAVE_PRED=data/pred_ds1v13.parquet RUN=20 PREP_DIR= FEAT_DIR=ms_features_v13 TAG=v13 \
  ARMS_A="baseline sc_big_dwell sc_big_hold" ARMS_B="" sh $D/run.sh
RUN=20 PREP_DIR= FEAT_DIR=ms_features_v13 TAG=v13 ARMS_A="" ARMS_B="baseline sc_big_dwell sc_big_hold" sh $D/run.sh
