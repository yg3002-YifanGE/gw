#!/usr/bin/env bash
set -euo pipefail

# This script runs the AI Interview Coach purely via Docker.
# Usage: bash docker_run.sh [IMAGE_NAME]
# IMAGE_NAME defaults to ai-interview-coach

DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${1:-ai-interview-coach}"

# Allow overriding host paths via env vars
DATA_DIR_HOST="${DATA_DIR_HOST:-"$DIR/../kaggle_data"}"
INDEX_DIR_HOST="${INDEX_DIR_HOST:-"$DIR/data"}"
WEB_DIR_HOST="${WEB_DIR_HOST:-"$DIR/web"}"

if [ ! -d "$DATA_DIR_HOST" ]; then
  echo "[ERROR] Kaggle data directory not found: $DATA_DIR_HOST" 1>&2
  echo "        Please ensure ../kaggle_data exists or set DATA_DIR_HOST to the correct path." 1>&2
  exit 1
fi

mkdir -p "$INDEX_DIR_HOST"

RUN_ARGS=(
  -p 8000:8000
  -v "$DATA_DIR_HOST:/data/kaggle_data:ro"
  -v "$INDEX_DIR_HOST:/app/data"
  -e DATA_DIR="/data/kaggle_data"
)

# If a .env file exists in the project folder, use it automatically
if [ -f "$DIR/.env" ]; then
  RUN_ARGS+=( --env-file "$DIR/.env" )
fi

# Mount web directory for development; image should already include /app/web
if [ -d "$WEB_DIR_HOST" ]; then
  RUN_ARGS+=( -v "$WEB_DIR_HOST:/app/web:ro" )
fi

echo "[INFO] Running image: $IMAGE"
echo "[INFO] Mounting data: $DATA_DIR_HOST -> /data/kaggle_data (ro)"
echo "[INFO] Persisting index: $INDEX_DIR_HOST -> /app/data"
docker run "${RUN_ARGS[@]}" "$IMAGE"
