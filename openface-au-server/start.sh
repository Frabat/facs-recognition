#!/usr/bin/env bash
set -e

FIFO="/tmp/live.mkv"

echo "[startup] Creating FIFO at $FIFO"
rm -f "$FIFO"
mkfifo "$FIFO"

echo "[startup] Starting RTSP pull using ffmpeg..."
ffmpeg -rtsp_transport tcp \
    -i "$RTSP_URL" \
    -c copy \
    -f matroska \
    "$FIFO" \
    &

PID_FFMPEG=$!

echo "[startup] Starting OpenFace FeatureExtraction..."
/opt/OpenFace/build/bin/FeatureExtraction \
    -f "$FIFO" \
    -aus \
    -q \
    -of /dev/stdout \
| tail -n +2 \
| while read -r line; do
      mosquitto_pub -h "$MQTT_HOST" -p "$MQTT_PORT" -t "$MQTT_TOPIC" -m "$line"
  done

echo "[startup] Pipeline terminated."

kill $PID_FFMPEG 2>/dev/null || true
