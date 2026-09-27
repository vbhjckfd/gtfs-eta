#!/bin/sh
# run 15: 3x-data check of the leader sc_big_dwell (tag ds1v12k3), after the ds1 V12 fits.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
while pgrep -f "run15b.sh" | grep -v $$ >/dev/null || pgrep -f "harness.py run" >/dev/null; do sleep 30; done
export MS_FEAT_DIR=ms_features_v12
export MS_DROP_COLS=live_path_sec,since_last_at_target,live_path_n,live_path_sec_m7,live_path_sec_ewm,fill_path_sec,live_hist_ratio,hist_path_p75,own_hist_ratio3,own_hist_ratio8,own_excess8,hist_x_own,next_stop_id,lap6_path_sec,lap6_path_cov,lap6_path_age,lap6_fill_path_sec,net_ratio15,net_n15,veh_hist_ratio20,veh_hist_ratio60c,hw_ahead_cur,hw_ahead_tgt,hw_ahead_prev,dwell_c_rem_med,dwell_c_p_more120,dwell_c_n
H=research/model-search/harness.py
save() {
  git add research/model-search/results >/dev/null 2>&1
  git commit -qm "model-search run 15: results ($1)" >/dev/null 2>&1 && \
    for i in 1 2 3 4; do git push -q origin claude/model-search && break; sleep $((2*i)); done
}
save "ds1v12 sc_big_v12"
export MS_INPLACE=1   # sc_big_dwell at 3x OOM-killed with the frame copy in _sentinel
for a in ${K3_ARMS:-sc_big_dwell baseline}; do
  python $H run --arm $a --train 2026-09-07..2026-09-18 --test 2026-09-19..2026-09-21 \
    --seed 42 --tag ds1v12k3 --keep-pct 3; save "k3 $a"
done
echo RUN15B DONE
