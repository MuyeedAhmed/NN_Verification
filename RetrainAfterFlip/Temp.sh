#!/bin/bash

TARGET_PID=69581
PYTHON_SCRIPT="Flip.py"

while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 600
done

echo "Process $TARGET_PID has terminated. Running Python script..."
python "$PYTHON_SCRIPT"