FROM python:3.13-slim

# System deps for opencv's GUI/camera bits (no compile toolchain needed —
# opencv-python and onnxruntime both ship prebuilt wheels).
# Builds fine under both Podman (team standard) and Docker — standard syntax.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk-3-0 \
    libgl1 \
    v4l-utils \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY client.py ./
COPY config.yaml ./
COPY config.auraface.yaml ./
COPY models/ ./models/

RUN pip install --no-cache-dir \
    setuptools==80.9.0 \
    numpy==2.4.6 \
    onnxruntime==1.27.0 \
    opencv-python==5.0.0.93 \
    requests==2.34.2 \
    pyyaml==6.0.3 \
    protobuf==7.35.1 \
    flatbuffers==25.12.19 \
    certifi==2026.6.17 \
    urllib3==2.7.0 \
    charset-normalizer==3.4.7 \
    idna==3.18

ENTRYPOINT ["python", "client.py"]
CMD ["--help"]
