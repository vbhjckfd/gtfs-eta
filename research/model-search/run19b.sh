#!/bin/sh
# run 19b: serving timing of baseline / sc_big_dwell / sc_big_511_it2k (models saved via MS_SAVE_MODEL).
D=research/model-search
mkdir -p data/ms_models
for a in ${ARMS-baseline sc_big_dwell sc_big_511_it2k}; do
  [ -f data/ms_models/$a.pkl ] || MS_FEAT_DIR=ms_features_v10 MS_SAVE_MODEL=data/ms_models/$a.pkl python $D/harness.py run \
    --arm $a --train 2026-09-07..2026-09-18 --test 2026-09-19..2026-09-21 --seed 42 --tag ds1v19t || exit 1
done
python $D/timing.py $(for a in ${ARMS-baseline sc_big_dwell sc_big_511_it2k}; do echo data/ms_models/$a.pkl; done)
echo RUN19B DONE
