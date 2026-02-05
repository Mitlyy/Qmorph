#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-config/config_books.yaml}
python compare_models.py --config "$CONFIG_PATH"
