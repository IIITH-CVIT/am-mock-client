# Am-FaceRecognition-Client

A Python-based face recognition client that detects faces in images or a live camera feed, extracts embeddings, and identifies people either via a remote FRU server or a local offline database (diagnostic mode).

---

## Overview

```
Input Image / Live Camera Frame
     │
     ▼
FaceDetector (YuNet, native OpenCV 5 — cv2.FaceDetectorYN)
     │                                bbox + 5-point landmarks
     ▼
SFaceEmbedder (128-dim)           ← default; native cv2.FaceRecognizerSF, L2-normalised
AuraFaceEmbedder (512-dim)        ← alternate; ONNX R100 (ArcFace-style), L2-normalised, landmark-aligned
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

Detection is always YuNet (`cv2.FaceDetectorYN`, built into OpenCV 5 — no more
manual ONNX decode, no dlib). Only the **embedder** varies, and it must match
whatever the server is enrolled with (`models.embedder_model` in the server's
own `config.yaml` — the mock server enrols vectors from exactly **one** model
at a time, never both):

- **sface (128-dim)** — the **default** (`config.yaml`). `cv2.FaceRecognizerSF`,
  native OpenCV 5, no `onnxruntime` needed for this path.
- **auraface / R100 (512-dim)** — the alternate (`config.auraface.yaml`).
  `aurar100.onnx` (ArcFace-style ResNet-100), via `onnxruntime`, L2-normalised,
  landmark-aligned into the 112×112 ArcFace reference pose.

**Never mix embedders against the same server/DB** — the server matches by
vector dimension and whatever model it's configured with; the client and
server must agree. To switch, change one config setting (or pass
`--config config.auraface.yaml`); nothing about the detector changes either
way.

**Two modes:**

| Mode | When to use | Storage |
|---|---|---|
| `server` | FRU API is running and reachable | Remote Qdrant/PostgreSQL |
| `diagnostic` | Offline / testing / server down | Local SQLite |

When mode is `server` and the server is unreachable, the client automatically falls back to the local SQLite database.

---

## Quick start

Recognise someone against the mock server in four commands (assumes `am-mock-server/` and `am-mock-client/` sit side by side):

```bash
# 1. Start the mock server (other repo) — installs Podman, builds, runs on :8000
cd ../am-mock-server && ./run.sh
#    ...then register a face at http://localhost:8000 (see that repo's "How to use")

# 2. Set up this client — native venv + all deps (prebuilt wheels, no compile)
cd ../am-mock-client && ./setup.sh

# 3. Identify a photo of the person you registered (default sface model)
.venv/bin/python client.py --server photo.jpg
```

Expect `>>> Recognised: <name>`. That uses the **default sface model**. If the
server is instead enrolled with `auraface`, add `--config config.auraface.yaml`.
Step-by-step detail (and the offline/diagnostic flow) is under
[`## How to use`](#how-to-use) below; full setup options under
[`### Setup`](#setup).

---

## Project Structure

```
Am-FaceRecognition-Client/
├── client.py                # Main client (all logic)
├── config.yaml               # DEFAULT config — sface model, mock server :8000
├── config.auraface.yaml      # Alternate preset — AuraFace R100 (512-dim)
├── environment.yml           # Conda environment spec
├── models/
│   ├── face_detection_yunet_2026may.onnx     # YuNet face detector (native OpenCV 5)
│   ├── face_recognition_sface_2021dec.onnx   # SFace embedder (native OpenCV 5)
│   └── aurar100.onnx                         # AuraFace R100 embedder (onnxruntime)
├── diagnostic_mode/
│   ├── diagnostic.db       # SQLite face database (auto-created)
│   └── faces/              # Saved face crops from --register
└── README.md
```

---

## Requirements

- Python 3.13
- `opencv-python-headless>=5.0.0` (ships YuNet detection + SFace recognition natively — no compile step)
- `onnxruntime` (only exercised by the `auraface` embedder path)
- The three `.onnx` models in `models/` (bundled in this repo; same files as `mock-server/models/`)

### Setup

**Quickest — `./setup.sh` (recommended):** bootstraps a native `.venv` and installs
every dependency in one go. No manual pip steps, no `sudo`, no compile step.

```bash
./setup.sh
```

The script prefers `uv` and falls back to `python3 -m venv`; it's safe to re-run.

**Manual alternative — create the virtual environment yourself:**

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

### Running the tests

```bash
python3 -m pytest tests/test_server_client.py tests/test_client_errors.py tests/test_diagnostic_db.py
```

- `test_server_client.py` — server response parsing (`IdentifyResponse` field names, error/dimension-mismatch handling).
- `test_client_errors.py` — the `ClientError` error paths (missing image, no face detected, missing models).
- `test_diagnostic_db.py` — local `DiagnosticDB` cosine search: best-match-above-threshold, top-K ordering, dimension-mismatch gating, and the empty-DB path.

These import `client.py` directly and don't need a camera or a running server. `tests/test_containerfile.py` is separate: it requires Podman and a built `face-recognition` image (`./build.sh`) and skips automatically if Podman isn't installed.

---

### Running in Podman (live camera)

`./run.sh` builds and runs the client in a **Podman** container (team standard — not
Docker) with your host's cameras and X display forwarded in, for live camera mode:

```bash
./build.sh   # first time / after code changes — installs Podman if missing, then builds
./run.sh     # builds automatically on first run if needed
```

This requires an X server (Linux desktop). `run.sh` handles the X11 forwarding (`xhost`, `DISPLAY`, `/tmp/.X11-unix`) and camera devices automatically; it also passes `--group-add keep-groups` so rootless Podman can open the camera (if it can't, add your user to the `video` group). It won't work over SSH without X forwarding of your own (`ssh -X`), and needs extra X server setup on macOS/Windows (XQuartz / VcXsrv) which isn't covered here.

For single-image identify/register/list (`--server photo.jpg`, `--register`, `--list`), you don't need a container or a display at all — just run natively:
```bash
./setup.sh
.venv/bin/python client.py --server photo.jpg
```
This is the simpler path for testing against the mock server. The Podman camera/X11 setup is only worth the overhead if you specifically need live-camera testing.

## Configuration

All settings live in `config.yaml` (the default, sface model). Edit this file before
running, or use the ready-made `config.auraface.yaml` for the AuraFace R100 model.

```yaml
# Active mode: "server" or "diagnostic"
mode: server

server:
  url: "http://localhost:8000"       # the mock server; change for a real am-master-server
  timeout: 10                        # seconds

diagnostic:
  db_path: ./diagnostic_mode/diagnostic.db
  faces_dir: ./diagnostic_mode/faces
  cosine_threshold: 0.6              # raise to be stricter (0.0–1.0)
  topk: 3                            # nearest neighbours shown on unknown

models:
  face_detector_path: ./models/face_detection_yunet_2026may.onnx
  face_recognizer_path: ./models/face_recognition_sface_2021dec.onnx
  face_recognizer_auraface_path: ./models/aurar100.onnx

detection:
  threshold: 0.5                     # YuNet score threshold
  input_size: 640                    # initial square size; each frame's real size is set at detect-time

embedder:
  model: sface                       # sface (128-dim, native OpenCV 5) | auraface (512-dim, R100 via onnxruntime)

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
> **Want the AuraFace R100 model instead of the default sface one?**
> Use the ready-made preset instead of editing `config.yaml`:
> ```bash
> .venv/bin/python client.py --config config.auraface.yaml <image>
> ```
> It talks to the same mock server (`localhost:8000`) — only the embedder differs.
> Make sure the server's own `models.embedder_model` is also set to `auraface`,
> since the server enrols one model at a time.
---

## How to use

```
.venv/bin/python client.py [options] [image]
```

### Tutorial A — identify someone via the mock server

Prereq: the mock server is running (`am-mock-server/run.sh`) and you've registered a face there via its web UI.

```bash
.venv/bin/python client.py --server alice.jpg
```
Using the **default sface model**, the client detects the face, computes a 128-dim SFace embedding, POSTs it to `http://localhost:8000/api/v1/identify/`, and prints the server's answer:
```
[INFO] MODE: server (http://localhost:8000)  <- alice.jpg
[INFO] Face: bbox=(...)  score=0.98
[INFO] Embedding: 128-dim
[INFO] Server -> name='Alice Kumar' confidence = 0.83 distance = 0.34
>>> Recognised: Alice Kumar
```
- **`distance`** is normalised L2 (lower = better); under the server's match threshold (`0.8` by default) = a match.
- If the server is enrolled with **AuraFace R100** instead, add `--config config.auraface.yaml` (the log then shows `Embedding: 512-dim`).
- If the server is unreachable, the client automatically falls back to the local diagnostic DB.

### Tutorial B — offline diagnostic mode (no server)

Register faces into a local SQLite DB and match against it — no server, no network.

```bash
# Register people (default sface model)
.venv/bin/python client.py --register Alice alice.jpg
.venv/bin/python client.py --register Bob   bob.jpg

# Identify against them
.venv/bin/python client.py --diag alice2.jpg
# -> >>> Recognised: Alice  (cosine sim=0.79)
```
On an unknown face it prints the top-3 nearest matches with similarity scores. Face crops are saved under `diagnostic_mode/faces/`.

> Keep the model consistent within the local DB: faces registered under sface (128-dim) only match sface queries, and auraface (512-dim) only matches auraface. Pick one model (don't mix `config.yaml` and `config.auraface.yaml` against the same local DB) — or re-register if you switch.

**Command reference** — every flag:

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

- Uses **sface 128-dim** (L2-normalised) by default
- For **auraface R100 512-dim** instead, pass `--config config.auraface.yaml` (or set `embedder.model: auraface`) — and make sure the server is enrolled with the matching model
- If the server is unreachable, **automatically falls back** to the local SQLite database

### Diagnostic mode

Identifies faces locally using cosine similarity against embeddings stored in `diagnostic_mode/diagnostic.db`.

- Embedding dimension/backend follows `embedder.model` (sface 128-dim or auraface 512-dim)
- **Cosine similarity** search (vectorised, matches reference `search_diagnostic_faces`)
- Returns the best match above `cosine_threshold` (default `0.6`)
- Rows are dimension-gated — switching `embedder.model` won't match faces registered under the other embedder; re-register if you switch
- On unknown faces, logs the **top-3 nearest matches** and their similarity scores
- Face crops are saved to `diagnostic_mode/faces/` on `--register`

---

## Detector + embedding details

| Component | Backend | Dim | Normalisation | Notes |
|---|---|---|---|---|
| Detector | YuNet (`cv2.FaceDetectorYN`, native OpenCV 5) | — | — | Only detector; returns bbox + 5-point landmarks |
| Embedder | SFace (`cv2.FaceRecognizerSF`, native OpenCV 5) | 128 | L2 | Default; aligned via `alignCrop()` |
| Embedder | AuraFace R100 (ONNX, ArcFace-style) | 512 | L2 | Alternate; landmark-aligned into the 112×112 ArcFace pose |

Mirrors `am-mock-server/app/core/face_engine.py` exactly — the server never
re-derives embeddings from pixels, it only vector-searches whatever the client
submits, gated by dimension (`app/api/v1/identify.py`). The client and the
server it talks to must be configured with the same embedder.

---

## Models

| File | Source | Used for |
|---|---|---|
| `models/face_detection_yunet_2026may.onnx` | Bundled in this repo (same file as `mock-server/models/`) | YuNet face detection |
| `models/face_recognition_sface_2021dec.onnx` | Bundled in this repo (same file as `mock-server/models/`) | SFace 128-dim embedding |
| `models/aurar100.onnx` | Bundled in this repo (same file as `mock-server/models/`) | AuraFace R100 512-dim embedding |

`models/*.onnx` are checked out with this repo (relative paths resolve against `config.yaml`'s directory, not the current working directory). `aurar100.onnx` is a large (~250MB) R100 backbone — consider Git LFS if repo size becomes a concern.

---

## Reference

This client mirrors `am-mock-server/app/core/face_engine.py`'s `FaceEngine`:
one shared YuNet detector (`cv2.FaceDetectorYN`, native OpenCV 5) feeding
either `cv2.FaceRecognizerSF` (sface, 128-dim) or `aurar100.onnx` via
`onnxruntime` (auraface, 512-dim) — same preprocessing, same landmark
alignment, same L2 normalisation. That's why vectors computed here match what
the mock server enrols, as long as both sides are configured with the same
`embedder_model`.

Note: the production `am-master-server` (`app/core/face_rec.py`) has not yet
migrated off its older `DlibBackend`/`AurafaceBackend` (MobileFaceNet)
pairings — this client will not match a real `am-master-server` deployment
until that side is updated to sface/auraface too. Point `server.url` at the
mock server, or a migrated deployment, accordingly.
