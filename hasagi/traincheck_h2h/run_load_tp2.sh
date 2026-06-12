#!/usr/bin/env bash
# TrainCheck launcher for the two-rank same-TP load control. The collector
# rewrites the script name below to the instrumented copy before running.
set -e
python -m torch.distributed.run --nproc_per_node=2 --master_port=29582 load_tp1.py
