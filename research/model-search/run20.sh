#!/bin/sh
# run 20: leader without hour sample weights (sc_big_nw) on both splits; the ds1 fits save per-row
# predictions (last arm = sc_big_dwell) for the route-133 diagnostic (diag_route.py).
D=research/model-search
for i in 1 2 3; do
  MS_SAVE_PRED=data/pred_ds1v20.parquet RUN=20 PREP_DIR=ms_features_v10 TAG=v20 \
    ARMS_A="baseline sc_big_nw sc_big_dwell" ARMS_B="" sh $D/run.sh && break
  echo "retry $i"; sleep 60
done
RUN=20 PREP_DIR= FEAT_DIR=ms_features_v10 TAG=v20 ARMS_A="" ARMS_B="baseline sc_big_nw sc_big_dwell" sh $D/run.sh
echo RUN20 ALL DONE
