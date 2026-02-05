#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 \"text\" [config]"
  exit 1
fi

TEXT=$1
CONFIG_PATH=${2:-config/config_books.yaml}
python inference.py --config "$CONFIG_PATH" --model-type qmorph --sentence "$TEXT"
