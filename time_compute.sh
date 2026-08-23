#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python report_timing.py \
--surface-sample-ratio 1.0 \
--obj-sample-ratio 1.0 \
--material-hash mata30bf2022a \
--gpu 1 