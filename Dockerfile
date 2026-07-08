FROM python:3.13-slim

# System deps for dlib (C++ compile) + camera tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    libopenblas-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY client.py config.yaml ./
COPY models/ ./models/

RUN pip install --no-cache-dir \
    dlib==20.0.1 \
    face_recognition_models==0.3.0 \
    numpy==2.4.6 \
    onnxruntime==1.27.0 \
    opencv-python-headless==4.13.0.92 \
    requests==2.34.2 \
    protobuf==7.35.1 \
    flatbuffers==25.12.19 \
    certifi==2026.6.17 \
    urllib3==2.7.0 \
    charset-normalizer==3.4.7 \
    idna==3.18

CMD ["python", "client.py", "--help"]
