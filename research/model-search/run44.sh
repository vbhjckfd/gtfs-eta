#!/bin/sh
# Run 44: fresh days, production trainer. Split A: train 10-01..10-09, test 10-10 Sat..10-13 Tue (seed 42).
# Split B: train 10-01..10-08, test 10-09 Fri..10-11 Sun (seed 11). Feature cache built by the first (EXTRA=1) call.
D=research/model-search
A="TRAIN=2026-10-01..2026-10-09 TEST=2026-10-10..2026-10-13 SEED=42"
B="TRAIN=2026-10-01..2026-10-08 TEST=2026-10-09..2026-10-11 SEED=11"
X=duty_s,duty_odo
P=pa_n,pa_spd_mean,pa_spd_min,pa_stop_frac,pa_gap_m,pa_gap_spd,pa_cov
env $A NAME=lv2x_dutypa_A_s42 EXTRA=1 CMP=1 sh $D/tl.sh
env $A NAME=lv2_A_s42 sh $D/tl.sh
env $A NAME=lv2x_duty_A_s42 EXTRA=1 DROP=$P sh $D/tl.sh
env $A NAME=lv2x_pa_A_s42 EXTRA=1 DROP=$X sh $D/tl.sh
env $B NAME=lv2_B_s11 sh $D/tl.sh
env $B NAME=lv2x_dutypa_B_s11 EXTRA=1 sh $D/tl.sh
echo RUN44 DONE
