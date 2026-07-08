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

## Quick start

Recognise someone against the mock server in four commands (assumes `am-mock-server/` and `am-mock-client/` sit side by side):

```bash
# 1. Start the mock server (other repo) — installs Podman, builds, runs on :8000
cd ../am-mock-server && ./run.sh
#    ...then register a face at http://localhost:8000 (see that repo's "How to use")

# 2. Set up this client — native venv + light deps, no dlib, no sudo
cd ../am-mock-client && ./setup.sh

# 3. Identify a photo of the person you registered
.venv/bin/python client.py --config config.mock-server.yaml --server photo.jpg
```

Expect `>>> Recognised: <name>`. Step-by-step detail (and the offline/diagnostic flow) is under [`## How to use`](#how-to-use) below; full setup options under [`### Setup`](#setup).

---

## Known Issues & Fixes (testing against the mock server)

- **`ServerClient.identify()` read dead schema fields**: was falling back to `visitor_name`/`similarity`, neither of which exist in the server's actual `IdentifyResponse`. Fixed: now reads `name`/`confidence`/`distance` directly and logs all three, instead of merging `distance` (lower=better) into a `sim` label that implied higher=better. Covered by `tests/test_server_client.py`.

- **Shipped `config.yaml` defaults (`dlib`, 128-dim) don't match the mock server**, which only implements the 512-dim `yunet`/`mobilefacenet` pairing (mirroring `AurafaceBackend`). Fixed: a dedicated `config.mock-server.yaml` ships alongside `config.yaml` for this exact purpose (see the callout in `## Configuration` above). A runtime guard in `ServerClient.identify()` also catches this specific mismatch and logs guidance pointing at the override config, if the server has the corresponding dimension-validation fix applied.

- **`server.url` was inconsistent across the docs**: the README's Configuration example showed `192.168.1.19:8000` while the shipped `config.yaml` (and `client.py`'s built-in default) use `localhost:8100`. Fixed: the README example now matches `config.yaml` at `localhost:8100` — the default for the real `am-master-server`; edit it for your own deployment. The mock server is a deliberate, separate case: use `config.mock-server.yaml` (`localhost:8000`), whose header documents the port difference inline. So there are exactly two intentional values now — `8100` for the real server, `8000` for the mock — and nothing drifts.

- **Fatal errors called `sys.exit(1)` deep in the client**: `_load_image`, `_ensure_models`, `_detect_and_embed`, camera-open and model-init all exited the process directly, so they couldn't be unit-tested (the test runner itself would exit). Fixed: these now raise a typed `ClientError`, and `main()` is the single place that catches it and translates it to a clean `exit(1)`. User-facing behaviour is unchanged (a logged error and exit code `1`, no traceback), but every function below `main()` is now importable and testable. Error paths are covered by `tests/test_client_errors.py`.

- **`import dlib` / `import face_recognition_models` at module top made `client.py` unimportable without the (heavy, compiled) dlib wheel** — including for tests using only the `yunet`/`mobilefacenet` ONNX pairing. Fixed: both are now imported lazily; `client.py` imports fine without them, and the dlib-backed detector/embedder raise a clear `ClientError` only if the dlib backend is actually selected while the packages are missing.

- **`DiagnosticDB.search` / `search_topk` carried ~30 lines of near-duplicated load/dim-filter/cosine logic**: maintainability risk, no functional bug. Fixed: the shared work is now one private `_scored_candidates()` helper; `search()` applies the argmax + threshold and `search_topk()` sorts + slices. Behaviour is unchanged (including that the dim-mismatch warning fires from `search()` but not the `search_topk()` call right after it). Covered by `tests/test_diagnostic_db.py`.

- **`ServerClient._post` formats the face vector with `%.6f` (6 decimal places) before sending** — flagged as an undocumented truncation. Examined, not just documented: embeddings are L2-normalised (or small-magnitude raw dlib descriptors), so over 2000 random 512-d unit vectors `%.6f` introduces at most ~5e-7 per component / ~7e-6 whole-vector L2 error, shifting the server's match distance by ≤1.5e-6 — six orders of magnitude below the `0.8` threshold, and it never changed a nearest-neighbour pick. Left as-is (full float32 round-trip needs ~9 significant figures and buys nothing for matching); the rationale is now a comment in `_post`.

- **`_ensure_models()` validated the ONNX weights but not the two dlib `.dat` files** (`dlib_face_recognition_resnet_model_v1.dat`, `shape_predictor_5_face_landmarks.dat`) that the dlib embedder loads from the `face_recognition_models` package. A broken/partial install failed with an obscure dlib `RuntimeError` mid-run instead of the clear up-front message the ONNX paths got. Fixed: when the dlib embedder is selected, `_ensure_models` now checks the package is importable and both `.dat` files exist, failing fast with the same "Missing model" message + a reinstall hint. Covered by `tests/test_client_errors.py`.

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
- YuNet + MobileFaceNet ONNX models in `models/` (bundled in this repo; same files as `mock-server/models/`) — used by the yunet/mobilefacenet pairing, which is what the mock server needs
- dlib 20.0.1 + its model files (from the `face_recognition_models` package) — **only** for the `dlib` pairing (real `am-master-server`), not for the mock-server path. `client.py` imports dlib lazily, so the client runs fine without it when using the yunet/mobilefacenet pairing (see `./setup.sh` vs `./setup.sh --full`).

### Setup

**Quickest — `./setup.sh` (recommended):** bootstraps a native `.venv` and installs
every dependency in one go. No manual pip steps, no `sudo`.

```bash
./setup.sh          # default: MOCK-SERVER path (light — no dlib)
./setup.sh --full   # dlib / real am-master-server path
```

By default it installs **only** what the mock-server path (yunet + mobilefacenet,
512-dim) needs — `onnxruntime` / `opencv` / `numpy` / `requests` / `pyyaml`. That
path uses **no dlib**, so there's no ~15-min source compile and no C++ build tools
required — ideal for testing against `iiith-cvit-am-mock-server`. Pass `--full` to
add `dlib` (compiles from source, needs `cmake`/`g++`/BLAS) for the real-server
`dlib` pairing. The script prefers `uv` and falls back to `python3 -m venv`; it's
safe to re-run.

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
- `test_client_errors.py` — the `ClientError` error paths (missing image, no face detected, missing models, dlib backend selected without the dlib packages).
- `test_diagnostic_db.py` — local `DiagnosticDB` cosine search: best-match-above-threshold, top-K ordering, dimension-mismatch gating, and the empty-DB path.

These import `client.py` directly and don't need `dlib`, a camera, or a running server. `tests/test_dockerfile.py` is separate: it requires Docker and a built `face-recognition` image (`./build.sh`) and won't pass without them.

---

### Running in Docker

`./run.sh` builds and runs the client in a container with your host's cameras and X display forwarded in, for live camera mode specifically:

```bash
./build.sh   # first time / after code changes
./run.sh
```

This requires an X server (Linux desktop). `run.sh` handles the X11 forwarding (`xhost`, `DISPLAY`, `/tmp/.X11-unix`) automatically, but it won't work over SSH without X forwarding of your own (`ssh -X`), and doesn't work on macOS/Windows Docker Desktop without extra X server setup (XQuartz / VcXsrv) which is not covered here.

For single-image identify/register/list (`--server photo.jpg`, `--register`, `--list`), you don't need Docker or a display at all, just run natively:
```bash
uv venv --python 3.13 .venv
uv pip install -r requirements.txt --python .venv/bin/python
.venv/bin/python client.py --server photo.jpg
```
This is also the simpler path for testing against the mock server (see `config.mock-server.yaml` above). Docker's camera/X11 setup is only worth the overhead if you specifically need live-camera testing.

## Configuration

All settings live in `config.yaml`. Edit this file before running.

```yaml
# Active mode: "server" or "diagnostic"
mode: server

server:
  url: "http://localhost:8100"       # FRU server address (matches config.yaml; edit for your deployment)
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
> **Testing against `iiith-cvit-am-mock-server` instead of the real `am-master-server`?** 
> Use `config.mock-server.yaml` instead of editing `config.yaml`:
> ```bash
> .venv/bin/python client.py --config config.mock-server.yaml <image>
> ```
> The mock server only implements the `yunet`/`mobilefacenet` (512-dim) pairing: `config.yaml`'s `dlib` default is correct for the real server, not this mock. See `config.mock-server.yaml`'s header comment for details.
---

## How to use

```
.venv/bin/python client.py [options] [image]
```

### Tutorial A — identify someone via the mock server

Prereq: the mock server is running (`am-mock-server/run.sh`) and you've registered a face there via its web UI.

```bash
.venv/bin/python client.py --config config.mock-server.yaml --server alice.jpg
```
The client detects the face (YuNet), computes a 512-dim MobileFaceNet embedding, POSTs it to `http://localhost:8000/api/v1/identify/`, and prints the server's answer:
```
[INFO] MODE: server (http://localhost:8000)  <- alice.jpg
[INFO] Face: bbox=(...)  score=0.98
[INFO] Embedding: 512-dim
[INFO] Server -> name='Alice Kumar' confidence = 0.83 distance = 0.34
>>> Recognised: Alice Kumar
```
- **`distance`** is L2 (lower = better); under the server's `0.8` cutoff = a match.
- **"Dimension mismatch … 512"** → you used the default `config.yaml` (dlib/128-dim) instead of `config.mock-server.yaml`. This is the #1 gotcha — always pass `--config config.mock-server.yaml` for the mock.
- If the server is unreachable, the client automatically falls back to the local diagnostic DB.

### Tutorial B — offline diagnostic mode (no server)

Register faces into a local SQLite DB and match against it — no server, no network.

```bash
# Register people (use config.mock-server.yaml so embeddings are 512-dim, consistent with the server)
.venv/bin/python client.py --config config.mock-server.yaml --register Alice alice.jpg
.venv/bin/python client.py --config config.mock-server.yaml --register Bob   bob.jpg

# Identify against them
.venv/bin/python client.py --config config.mock-server.yaml --diag alice2.jpg
# -> >>> Recognised: Alice  (cosine sim=0.79)
```
On an unknown face it prints the top-3 nearest matches with similarity scores. Face crops are saved under `diagnostic_mode/faces/`.

> Keep the embedder consistent: faces registered under `mobilefacenet` (512-dim) only match queries also made with `mobilefacenet`. Don't mix with the dlib (128-dim) pairing.

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
