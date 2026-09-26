#!/bin/sh
# run 11: re-run of run 9's plan (runs 9 and 10 were lost mid-run). pipeline_lite
# for 08-31..09-21, prep chain into ms_features_v7 as days land, then fits.
# Results are committed and pushed after every fit so a reclaimed session keeps them.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
export MS_FEAT_DIR=ms_features_v7
H=research/model-search/harness.py
Q=research/model-search/queue.sh
save() {
  git add research/model-search/results >/dev/null 2>&1
  git commit -qm "model-search run 11: results ($1)" >/dev/null 2>&1 && \
    for i in 1 2 3 4; do git push -q origin claude/model-search && break; sleep $((2*i)); done
}
for day in $(python -c "import sys;sys.path.insert(0,'research/model-search');from harness import expand_days;print(' '.join(expand_days('2026-08-31..2026-09-21')))"); do
  while [ ! -f data/ms_lite/$day.pos.parquet ] || [ ! -f data/ms_lite/$day.parquet ]; do sleep 30; done
  sleep 20
  while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 6 ]; do sleep 30; done
  python $H prep --days $day || { echo "PREP FAIL $day"; exit 1; }
  echo "PREP OK $day $(date -u +%H:%M)"
done
echo PREP DONE
while pgrep -f pipeline_lite.py >/dev/null; do sleep 30; done
mkdir -p data/ms_models
A="2026-09-07..2026-09-18 2026-09-19..2026-09-21 42"
B="2026-09-07..2026-09-17 2026-09-18..2026-09-20 7"
MS_SAVE_MODEL=data/ms_models/baseline.pkl sh $Q ds1v7 $A baseline; save baseline
MS_SAVE_MODEL=data/ms_models/sc_big_cur.pkl sh $Q ds1v7 $A sc_big_cur; save sc_big_cur
sh $Q ds1v7 $A sc_big_lap; save sc_big_lap
sh $Q ds1shiftv7 $B baseline sc_big_cur; save shift
sh $Q ds1shiftv7 $B sc_big_lap; save shift_lap
sh $Q ds1v7 $A sc_big_cur_reg; save reg
echo RUN11 DONE
