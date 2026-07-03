#!/bin/bash
set -e
echo "Building face-recognition image (dlib compiles from source — ~15 min)..."
docker build -t face-recognition .
echo "Build complete. Run with: ./run.sh"
