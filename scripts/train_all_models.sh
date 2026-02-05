#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-config/config_books.yaml}
python train.py --config "$CONFIG_PATH" --model-type baseline
python train.py --config "$CONFIG_PATH" --model-type qmorph
