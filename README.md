# Am-FaceRecognition-Client

A Python-based face recognition client that detects faces in images or a live camera feed, extracts embeddings, and identifies people either via a remote FRU server or a local offline database (diagnostic mode).

---

## Overview

```
Input Image / Live Camera Frame
     │
     ▼
DlibFaceDetector (HOG)           ← detects face, returns bbox + score
YuNetDetector (ONNX)             ← alternate; bbox + 5-point landmarks
     │
     ▼
DlibEmbedder (ResNet 128-dim)    ← raw embedding (un-normalised)
MobileFaceNetEmbedder (512-dim)  ← alternate; L2-normalised via ONNX, landmark-aligned
     │
     ▼
┌──────────────┐     ┌──────────────────────────┐
│  Server mode │  OR │  Diagnostic mode (SQLite) │
│  POST /api/  │     │  Cosine similarity search │
│  v1/identify │     │  against local DB         │
└──────────────┘     └──────────────────────────┘
     │
     ▼
 Prints name (or overlays it live on the camera window)
```

The default detector (dlib HOG) + embedder (dlib ResNet, 128-dim) pair mirrors
`am-master-server`'s `DlibBackend` exactly (`app/core/face_rec.py`: HOG detection
with `number_of_times_to_upsample=1`, 128-dim raw embedding, server-side L2 cutoff
`dlib_threshold=0.6`), so embeddings computed here match what the server has
enrolled. YuNet + MobileFaceNet is kept as a selectable alternate pairing,
mirroring `am-master-server`'s `AurafaceBackend` instead (512-dim L2-normalised,
server-side L2 cutoff `auraface_threshold=0.8`). **The two pairings are never
mixed** — the server never re-derives embeddings from pixels, so whichever
detector you pick, use its matching embedder.

**Two modes:**

| Mode | When to use | Storage |
|---|---|---|
| `server` | FRU API is running and reachable | Remote Qdrant/PostgreSQL |
| `diagnostic` | Offline / testing / server down | Local SQLite |

When mode is `server` and the server is unreachable, the client automatically falls back to the local SQLite database.

---

## Known Issues & Fixes (testing against the mock server)

- **`ServerClient.identify()` read dead schema fields**: was falling back to `visitor_name`/`similarity`, neither of which exist in the server's actual `IdentifyResponse`. Fixed: now reads `name`/`confidence`/`distance` directly and logs all three, instead of merging `distance` (lower=better) into a `sim` label that implied higher=better. Covered by `tests/test_server_client.py`.

- **Shipped `config.yaml` defaults (`dlib`, 128-dim) don't match the mock server**, which only implements the 512-dim `yunet`/`mobilefacenet` pairing (mirroring `AurafaceBackend`). This is correct behavior against the *real* `am-master-server` (which does support dlib) — for testing against *this* mock server specifically, use a separate override config rather than editing `config.yaml`:

```bash
  cp config.yaml config.mock-server.yaml
```
  then in `config.mock-server.yaml`:
```yaml
  server:
    url: "http://localhost:8000"   # matches mock-server's compose.yml port
  detection:
    detector: yunet
  embedder:
    model: mobilefacenet
```
  Run with `--config config.mock-server.yaml`. See "Running end-to-end against the mock server" below.

- **`server.url` in the README's own example (`192.168.1.19:8000`) doesn't match the shipped `config.yaml` default (`localhost:8100`)** — neither matches the mock server's actual port (`8000`). Worth double-checking whichever server you're pointing at.

## Project Structure

```
Am-FaceRecognition-Client/
├── client.py               # Main client (all logic)
├── config.yaml             # Configuration (mode, server URL, thresholds, …)
├── environment.yml         # Conda environment spec
├── models/
│   ├── face_detection_yunet_2023mar.onnx  # YuNet face detector
│   └── mobilefacenet.onnx                 # MobileFaceNet embedder
├── diagnostic_mode/
│   ├── diagnostic.db       # SQLite face database (auto-created)
│   └── faces/              # Saved face crops from --register
└── README.md
```

---

## Requirements

- Python 3.13
- dlib 20.0.1 (built from source via zig)
- dlib model files (provided by `face_recognition_models` package)
- YuNet + MobileFaceNet ONNX models in `models/` (bundled in this repo; same files as `mock-server/models/`) — only needed if using the yunet/mobilefacenet alternate pairing

### Setup

**Create the virtual environment:**

```bash
uv venv --python 3.13 .venv
uv pip install -r requirements.txt --python .venv/bin/python
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate face-recognition
```

---

## Configuration

All settings live in `config.yaml`. Edit this file before running.

```yaml
# Active mode: "server" or "diagnostic"
mode: server

server:
  url: "http://192.168.1.19:8000"   # FRU server address
  timeout: 10                        # seconds

diagnostic:
  db_path: ./diagnostic_mode/diagnostic.db
  faces_dir: ./diagnostic_mode/faces
  cosine_threshold: 0.6              # raise to be stricter (0.0–1.0)
  topk: 3                            # nearest neighbours shown on unknown

models:
  yunet: ./models/face_detection_yunet_2023mar.onnx
  mobilefacenet: ./models/mobilefacenet.onnx

detection:
  detector: dlib                     # dlib (HOG, matches server's DlibBackend) | yunet (ONNX, matches AurafaceBackend)
  threshold: 0.3                     # detector confidence threshold (dlib default 0.3, yunet default 0.5)
  input_size: 640                    # yunet letterbox size — higher finds smaller faces, slower
  num_upsamples: 1                   # dlib only — higher finds smaller faces, slower

embedder:
  model: dlib                        # dlib (128-dim, matches server's DlibBackend) | mobilefacenet (512-dim, matches AurafaceBackend)
  num_jitters: 1                     # dlib only — >1 = more stable, slower

camera:
  device: 0                          # camera device index (0 = default webcam)
  frame_skip: 10                     # run recognition every N frames

logging:
  level: INFO                        # DEBUG | INFO | WARNING | ERROR
```

You can also use a different config file per run:

```bash
.venv/bin/python client.py --config prod.yaml photo.jpg
```

---

## Usage

```
.venv/bin/python client.py [options] [image]
```

### Identify a face

```bash
# Uses mode from config.yaml (server → auto-fallback to diagnostic if down)
.venv/bin/python client.py photo.jpg

# Force server mode
.venv/bin/python client.py --server photo.jpg

# Force diagnostic mode (local SQLite, no network)
.venv/bin/python client.py --diag photo.jpg
```

### Live camera detection

```bash
# Explicit flag...
.venv/bin/python client.py --camera

# ...or just run with no arguments (camera is the default)
.venv/bin/python client.py
```

Opens `camera.device` from `config.yaml`, detects + identifies a face every `camera.frame_skip` frames, and overlays the bbox and matched name (or "Unknown") live on the video window. Press `q` to quit.

### Register a face (diagnostic DB)

```bash
.venv/bin/python client.py --register Alice alice.jpg
```

Detects the face, saves a crop to `diagnostic_mode/faces/`, and stores the embedding (dimension depends on `embedder.model`) in the local SQLite database.

### List registered faces

```bash
.venv/bin/python client.py --list
```

```
  ID  Name                  Registered            Image
────────────────────────────────────────────────────────────────────────────────
   3  Narayan               2026-06-20 04:55:50   diagnostic_mode/faces/narayan_…
   2  Sandeep               2026-06-20 04:55:18   diagnostic_mode/faces/sandeep_…
   1  KS_Rajan              2026-06-20 04:54:53   diagnostic_mode/faces/ks_rajan_…

Total: 3 face(s)
```

---

## Modes in detail

### Server mode

Sends the configured embedder's vector to the FRU API (`POST /api/v1/identify/`).

- Uses **dlib 128-dim** (raw) by default — matches `am-master-server`'s `DlibBackend` enrollment, server-side L2 cutoff `dlib_threshold=0.6`
- Switch `embedder.model: mobilefacenet` (with `detection.detector: yunet`) for **512-dim L2-normalised** embeddings instead, matching `AurafaceBackend`, server-side L2 cutoff `auraface_threshold=0.8`
- If the server is unreachable, **automatically falls back** to the local SQLite database

### Diagnostic mode

Identifies faces locally using cosine similarity against embeddings stored in `diagnostic_mode/diagnostic.db`.

- Embedding dimension/backend follows `embedder.model` (dlib 128-dim or mobilefacenet 512-dim)
- **Cosine similarity** search (vectorised, matches reference `search_diagnostic_faces`)
- Returns the best match above `cosine_threshold` (default `0.6`)
- Rows are dimension-gated — switching `embedder.model` won't match faces registered under the other embedder; re-register if you switch
- On unknown faces, logs the **top-3 nearest matches** and their similarity scores
- Face crops are saved to `diagnostic_mode/faces/` on `--register`

---

## Detector + embedding details

| Component | Backend | Dim | Normalisation | Notes |
|---|---|---|---|---|
| Detector | dlib (HOG) | — | — | Default; `number_of_times_to_upsample=1`, no landmarks (bbox-crop alignment) |
| Detector | YuNet (ONNX) | — | — | Alternate; returns bbox + 5-point landmarks for alignment |
| Embedder | dlib ResNet (`face_recognition_models`) | 128 | None (raw) | Default; matches `am-master-server`'s `DlibBackend` |
| Embedder | MobileFaceNet (ONNX) | 512 | L2 | Alternate; matches `am-master-server`'s `AurafaceBackend` |

Pairings mirror `am-master-server/app/core/face_rec.py` exactly and **must not be mixed** — the server never re-derives embeddings from pixels, it only vector-searches whatever the client submits and infers the model from vector dimensionality (`app/api/v1/identify.py`). Use dlib detector with dlib embedder, or YuNet detector with MobileFaceNet embedder.

---

## Models

| File | Source | Used for |
|---|---|---|
| `models/face_detection_yunet_2023mar.onnx` | Bundled in this repo (same file as `mock-server/models/`) | YuNet face detection |
| `models/mobilefacenet.onnx` | Bundled in this repo (same file as `mock-server/models/`) | MobileFaceNet 512-dim embedding |
| `dlib_face_recognition_resnet_model_v1.dat` | `face_recognition_models` package | dlib 128-dim embedding |
| `shape_predictor_5_face_landmarks.dat` | `face_recognition_models` package | Face alignment for dlib |

`models/*.onnx` are checked out with this repo (relative paths resolve against `config.yaml`'s directory, not the current working directory). dlib model files are bundled with the `face_recognition_models` pip package.

---

## Reference

This client mirrors three references:

- **`am-master-server/app/core/face_rec.py`** (the real production identification backends —
  primary reference for both detector/embedder pairings):
  - `DlibBackend` — `face_recognition.face_locations(model="hog", number_of_times_to_upsample=1)`
    + `face_recognition.face_encodings(...)`, 128-dim. Default pairing here.
  - `AurafaceBackend` — YuNet (640 input, `score_threshold=0.5`, strides 8/16/32) + landmark-aligned
    MobileFaceNet, 512-dim L2-normalised. Alternate pairing here.
  - `RecognitionConfig` (`app/core/config.py`) — server-side match thresholds:
    `dlib_threshold=0.6`, `auraface_threshold=0.8` (L2 distance, lower is better).
- The YuNet + MobileFaceNet ONNX runtime plumbing was cross-checked against `mock-server/app/core/face_engine.py`
  (`FaceEngine._detect_face` manual multi-scale YuNet decode, `FaceEngine.embed` landmark alignment) — same
  model files, same preprocessing, since the mock server implements the same `AurafaceBackend` pipeline.
- The dlib-backend recognition flow from `am-fru-desktop-app-Fru-MacOs/face_recognition_pipeline.py`:
  - `DlibFaceDetector` — HOG frontal face detector (same `threshold`, `num_upsamples`)
  - `DlibEmbeddingExtractor` — own HOG re-detect + 5-point shape predictor + ResNet descriptor
  - `DiagnosticFacePipeline` — offline SQLite pipeline with cosine similarity search
  - `DatabaseManager.search_diagnostic_faces()` — vectorised cosine similarity, top-K logging
