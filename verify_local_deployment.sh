#!/usr/bin/env bash
# verify_local_deployment.sh
# Verifies the BPO environment by running the Docker container locally and 
# testing all tasks via inference.py.

set -e

# Configuration
CONTAINER_NAME="bpo_test_server"
IMAGE_NAME="openenv-bpo:latest"
PORT=8000

echo "============================================================"
echo "   BPO Environment: Local Docker Verification Suite"
echo "============================================================"

echo "[1/5] Cleanup: Stopping any existing test containers..."
sudo docker stop $CONTAINER_NAME 2>/dev/null || true
sudo docker rm $CONTAINER_NAME 2>/dev/null || true

echo "[2/5] Deploy: Starting $IMAGE_NAME on port $PORT..."
# Ensure the image exists
if ! sudo docker images -q $IMAGE_NAME > /dev/null; then
    echo "Error: Docker image $IMAGE_NAME not found. Run 'openenv build' first."
    exit 1
fi

sudo docker run -d -p $PORT:$PORT --name $CONTAINER_NAME $IMAGE_NAME

echo "[3/5] Wait: Polling /health endpoint..."
READY=0
# Increased timeout to 60 seconds (30 tries * 2s)
for i in {1..30}; do
    # More robust health check (looking for HTTP 200)
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health || true)
    if [ "$STATUS" = "200" ]; then
        echo "Server is healthy and responding (HTTP 200)."
        READY=1
        break
    fi
    echo "  waiting for server... ($i/30) Status: $STATUS"
    sleep 2
done

if [ $READY -eq 0 ]; then
    echo "Error: Server failed to start within 60 seconds."
    sudo docker logs $CONTAINER_NAME
    sudo docker stop $CONTAINER_NAME
    sudo docker rm $CONTAINER_NAME
    exit 1
fi

echo "[4/5] Test: Running inference.py in 'test' mode..."
# Ensure local dependencies are used
export IMAGE_NAME=""
export SERVER_URL="http://localhost:$PORT"
export APP_ENV="test"

# Run the inference script
python inference.py

echo "[5/5] Cleanup: Shutting down test container..."
sudo docker stop $CONTAINER_NAME > /dev/null
sudo docker rm $CONTAINER_NAME > /dev/null

echo "============================================================"
echo "   Verification Complete!"
echo "============================================================"
