#!/usr/bin/env bash
# TrainCheck launcher for the two-rank TP=2 save job. The collector rewrites
# the script name below to the instrumented copy before running this file.
set -e
python -m torch.distributed.run --nproc_per_node=2 --master_port=29579 save_tp2.py
