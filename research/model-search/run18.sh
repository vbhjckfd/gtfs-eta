#!/bin/sh
# run 18: capacity / constraint arms on the leader (no new cols). Prep v10 (pruned code;
# the leader's cols are unchanged), fit on ds1 (tag v18); shift fits are chosen after.
RUN=18 PREP_DIR=ms_features_v10 TAG=v18 \
  ARMS_A="${ARMS_A-baseline sc_big_dwell sc_big_mono sc_big_511 sc_big_it2k}" ARMS_B="${ARMS_B-}" \
  sh research/model-search/run.sh
