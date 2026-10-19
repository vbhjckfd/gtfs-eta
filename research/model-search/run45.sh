#!/bin/sh
# Run 45: residual feedback on fresh days (split C). Waits for the pipeline, then rfb.py (baseline + 3 OOF fits + candidate).
# Split C: train 10-05..10-14, test 10-15 Thu .. 10-18 Sun, seed 42 (data 09-21..10-18 for the 14-day history tables).
while pgrep -f run_pipeline.py > /dev/null; do sleep 60; done  # (re-run after the OOM: stage 1 is cached)
python research/model-search/rfb.py --train 2026-10-05..2026-10-14 --test 2026-10-15..2026-10-18 --seed 42 --split C
echo RUN45 DONE
