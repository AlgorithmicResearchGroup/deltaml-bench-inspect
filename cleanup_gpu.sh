#!/bin/bash
# GPU cleanup script - removes dead containers holding GPU reservations

echo "=== GPU Cleanup Script ==="
echo "Checking for dead containers with GPU allocations..."

# Find containers that have exited but still exist (holding resources)
DEAD_CONTAINERS=$(docker ps -a --filter "status=exited" --format "{{.Names}}" | grep "v0run")

if [ -z "$DEAD_CONTAINERS" ]; then
    echo "No dead containers found."
else
    echo "Found dead containers:"
    echo "$DEAD_CONTAINERS"
    echo ""
    echo "Removing dead containers to free GPU reservations..."
    echo "$DEAD_CONTAINERS" | xargs docker rm -f
    echo "Cleanup complete!"
fi

echo ""
echo "Current GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "Active containers with potential GPU allocations:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep "v0run"
