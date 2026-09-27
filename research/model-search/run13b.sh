#!/bin/sh
# run 13b: 3x-data check of the leader (train rows at 3% of snapshots) on ds1v9.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
export MS_FEAT_DIR=ms_features_v9
# the first attempt (all v9 cols) was OOM-killed on 15.7 GB; skip cols neither arm uses
export MS_DROP_COLS=live_path_sec,since_last_at_target,live_path_n,live_path_sec_m7,live_path_sec_ewm,fill_path_sec,live_hist_ratio,hist_path_p75,own_hist_ratio3,own_hist_ratio8,own_excess8,hist_x_own,next_stop_id,lap6_path_sec,lap6_path_cov,lap6_path_age,lap6_fill_path_sec,net_ratio15,net_n15,veh_hist_ratio20,veh_hist_ratio60c
H=research/model-search/harness.py
save() {
  git add research/model-search/results >/dev/null 2>&1
  git commit -qm "model-search run 13: results ($1)" >/dev/null 2>&1 && \
    for i in 1 2 3 4; do git push -q origin claude/model-search && break; sleep $((2*i)); done
}
for a in baseline sc_big_veh; do
  python $H run --arm $a --train 2026-09-07..2026-09-18 --test 2026-09-19..2026-09-21 \
    --seed 42 --tag ds1v9k3 --keep-pct 3; save k3_$a
done
echo RUN13B DONE
