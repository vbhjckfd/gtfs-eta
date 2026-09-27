#!/bin/sh
# run 13b: 3x-data check of the leader (train rows at 3% of snapshots) on ds1v9.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
export MS_FEAT_DIR=ms_features_v9
H=research/model-search/harness.py
save() {
  git add research/model-search/results >/dev/null 2>&1
  git commit -qm "model-search run 13: results ($1)" >/dev/null 2>&1 && \
    for i in 1 2 3 4; do git push -q origin claude/model-search && break; sleep $((2*i)); done
}
while pgrep -f run13.sh >/dev/null; do sleep 30; done
for a in baseline sc_big_veh; do
  python $H run --arm $a --train 2026-09-07..2026-09-18 --test 2026-09-19..2026-09-21 \
    --seed 42 --tag ds1v9k3 --keep-pct 3; save k3_$a
done
echo RUN13B DONE
