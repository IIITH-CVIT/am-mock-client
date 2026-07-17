#!/bin/bash
set -e

echo "Installing PyInstaller..."
pip install pyinstaller

echo "Building standalone executable..."
pyinstaller \
    --onefile \
    --name face-recognition-client \
    --collect-all onnxruntime \
    --collect-all cv2 \
    --hidden-import onnxruntime.capi._pybind_state \
    client.py

echo ""
echo "Done! Executable is at: dist/face-recognition-client"
echo "Copy 'dist/face-recognition-client', 'config.yaml', and the 'models/' directory"
echo "(same directory layout) to the other machine and run it."
