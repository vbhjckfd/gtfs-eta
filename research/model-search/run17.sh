#!/bin/sh
# run 17: history-table lookback (MS_HIST_DAYS) for the leader's hist link / dwell tables.
# Pipeline days 08-17..09-21, prep all into v10 (14 d), re-fit ds1 baseline + leader (tag v10f),
# then re-prep 09-07..09-21 with 21 d and 7 d lookback into separate dirs (prior-day links_/runs_
# symlinked from v10; they don't depend on the lookback) and fit the leader on each.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
D=research/model-search
H=$D/harness.py
RUN=17 DAYS=2026-08-17..2026-09-21 PREP_DIR=ms_features_v10 TAG=v10f \
  ARMS_A="${ARMS_A-baseline sc_big_dwell}" ARMS_B="" sh $D/run.sh || exit 1
save() {
  git add $D/results >/dev/null 2>&1
  git commit -qm "model-search run 17: results ($1)" >/dev/null 2>&1 && \
    for i in 1 2 3 4; do git push -q origin claude/model-search && break; sleep $((2*i)); done
}
for HD in ${HDS-21 7}; do
  FD=ms_features_v10h$HD; mkdir -p data/$FD
  for f in data/ms_features_v10/links_2026-08-*.parquet data/ms_features_v10/runs_2026-08-*.parquet \
           data/ms_features_v10/links_2026-09-0[1-6].parquet data/ms_features_v10/runs_2026-09-0[1-6].parquet; do
    ln -sf "$(pwd)/$f" data/$FD/
  done
  for day in $(python -c "import sys;sys.path.insert(0,'$D');from harness import expand_days;print(' '.join(expand_days('2026-09-07..2026-09-21')))"); do
    MS_HIST_DAYS=$HD MS_FEAT_DIR=$FD python $H prep --days $day || { echo "PREP FAIL h$HD $day"; exit 1; }
    echo "PREP OK h$HD $day $(date -u +%H:%M)"
  done
  MS_FEAT_DIR=$FD python $H run --arm sc_big_dwell --train 2026-09-07..2026-09-18 --test 2026-09-19..2026-09-21 \
    --seed 42 --tag ds1h$HD
  save "ds1h$HD sc_big_dwell"
done
echo RUN17 DONE
