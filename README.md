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

This client can recognise faces with either of two models, and the **mock server
enrols both** for every registration, so either works out of the box:

- **dlib (HOG + ResNet, 128-dim)** — the **default** (`config.yaml`). Mirrors
  `am-master-server`'s `DlibBackend` exactly (HOG detection with
  `number_of_times_to_upsample=1`, 128-dim raw embedding, server-side L2 cutoff
  `dlib_threshold=0.6`).
- **YuNet + MobileFaceNet (512-dim)** — the alternate (`config.yunet.yaml`).
  Mirrors `am-master-server`'s `AurafaceBackend` (512-dim L2-normalised,
  server-side L2 cutoff `auraface_threshold=0.8`).

**The detector and embedder are always used as a matched pair** — dlib+dlib, or
yunet+mobilefacenet — never mixed. The server never re-derives embeddings from
pixels; it just searches whichever gallery matches the size of the vector you send
(128 → dlib, 512 → mobilefacenet). To switch models you change one config setting
(or pass `--config config.yunet.yaml`); nothing on the server side changes.

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

# 2. Set up this client — native venv + all deps (default includes dlib, ~10-15 min compile)
cd ../am-mock-client && ./setup.sh

# 3. Identify a photo of the person you registered (default dlib model)
.venv/bin/python client.py --server photo.jpg
```

Expect `>>> Recognised: <name>`. That uses the **default dlib model**. To use the
YuNet+MobileFaceNet model instead, add `--config config.yunet.yaml` (and you can
skip the dlib compile with `./setup.sh --light`). Step-by-step detail (and the
offline/diagnostic flow) is under [`## How to use`](#how-to-use) below; full setup
options under [`### Setup`](#setup).

---

## Known Issues & Fixes (testing against the mock server)

- **`ServerClient.identify()` read dead schema fields**: was falling back to `visitor_name`/`similarity`, neither of which exist in the server's actual `IdentifyResponse`. Fixed: now reads `name`/`confidence`/`distance` directly and logs all three, instead of merging `distance` (lower=better) into a `sim` label that implied higher=better. Covered by `tests/test_server_client.py`.

- **The mock server now enrols both models, so the default `config.yaml` (dlib) works against it directly.** (Earlier, the mock only implemented the 512-dim `yunet`/`mobilefacenet` pairing, and a separate `config.mock-server.yaml` was needed.) Today: `config.yaml` is the dlib default and points at the mock server (`localhost:8000`); `config.yunet.yaml` is the ready-made 512-dim alternate. Both hit the same server. A runtime guard in `ServerClient.identify()` still catches a genuinely wrong vector size (neither 128 nor 512 — e.g. a mismatched detector/embedder pairing) and logs which pairings are valid.

- **`server.url` used to point at the real server by default**: `config.yaml` and `client.py`'s built-in default were `localhost:8100` (the real `am-master-server`). Since this client ships as a matched pair with the mock server, the default is now `localhost:8000` (the mock) so it works out of the box — `config.yaml` and `config.yunet.yaml` both use it. For a real `am-master-server`, change `server.url` in `config.yaml` to your deployment's address.

- **Fatal errors called `sys.exit(1)` deep in the client**: `_load_image`, `_ensure_models`, `_detect_and_embed`, camera-open and model-init all exited the process directly, so they couldn't be unit-tested (the test runner itself would exit). Fixed: these now raise a typed `ClientError`, and `main()` is the single place that catches it and translates it to a clean `exit(1)`. User-facing behaviour is unchanged (a logged error and exit code `1`, no traceback), but every function below `main()` is now importable and testable. Error paths are covered by `tests/test_client_errors.py`.

- **`import dlib` / `import face_recognition_models` at module top made `client.py` unimportable without the (heavy, compiled) dlib wheel** — including for tests using only the `yunet`/`mobilefacenet` ONNX pairing. Fixed: both are now imported lazily; `client.py` imports fine without them, and the dlib-backed detector/embedder raise a clear `ClientError` only if the dlib backend is actually selected while the packages are missing.

- **`DiagnosticDB.search` / `search_topk` carried ~30 lines of near-duplicated load/dim-filter/cosine logic**: maintainability risk, no functional bug. Fixed: the shared work is now one private `_scored_candidates()` helper; `search()` applies the argmax + threshold and `search_topk()` sorts + slices. Behaviour is unchanged (including that the dim-mismatch warning fires from `search()` but not the `search_topk()` call right after it). Covered by `tests/test_diagnostic_db.py`.

- **`ServerClient._post` formats the face vector with `%.6f` (6 decimal places) before sending** — flagged as an undocumented truncation. Examined, not just documented: embeddings are L2-normalised (or small-magnitude raw dlib descriptors), so over 2000 random 512-d unit vectors `%.6f` introduces at most ~5e-7 per component / ~7e-6 whole-vector L2 error, shifting the server's match distance by ≤1.5e-6 — six orders of magnitude below the `0.8` threshold, and it never changed a nearest-neighbour pick. Left as-is (full float32 round-trip needs ~9 significant figures and buys nothing for matching); the rationale is now a comment in `_post`.

- **`_ensure_models()` validated the ONNX weights but not the two dlib `.dat` files** (`dlib_face_recognition_resnet_model_v1.dat`, `shape_predictor_5_face_landmarks.dat`) that the dlib embedder loads from the `face_recognition_models` package. A broken/partial install failed with an obscure dlib `RuntimeError` mid-run instead of the clear up-front message the ONNX paths got. Fixed: when the dlib embedder is selected, `_ensure_models` now checks the package is importable and both `.dat` files exist, failing fast with the same "Missing model" message + a reinstall hint. Covered by `tests/test_client_errors.py`.

## Project Structure

```
Am-FaceRecognition-Client/
├── client.py               # Main client (all logic)
├── config.yaml             # DEFAULT config — dlib model, mock server :8000
├── config.yunet.yaml       # Alternate preset — YuNet + MobileFaceNet (512-dim)
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
- YuNet + MobileFaceNet ONNX models in `models/` (bundled in this repo; same files as `mock-server/models/`) — used by the YuNet+MobileFaceNet model
- dlib 20.0.1 + its model files (from the `face_recognition_models` package) — used by the **default** dlib model. `client.py` imports dlib lazily, so the client still runs without it if you only ever use the YuNet+MobileFaceNet model (`./setup.sh --light`).

### Setup

**Quickest — `./setup.sh` (recommended):** bootstraps a native `.venv` and installs
every dependency in one go. No manual pip steps, no `sudo`.

```bash
./setup.sh           # default: installs EVERYTHING, including dlib (~10-15 min compile)
./setup.sh --light   # skip dlib — YuNet + MobileFaceNet only (fast, no compile)
```

The **default** installs dlib because the default `config.yaml` uses the dlib
model. dlib compiles from source (~10-15 min, needs `cmake`/`g++`/BLAS, which most
Linux boxes have). If you only plan to use the YuNet+MobileFaceNet model, run
`./setup.sh --light` (installs just `onnxruntime` / `opencv` / `numpy` / `requests`
/ `pyyaml`, no compile) and always pass `--config config.yunet.yaml`. The script
prefers `uv` and falls back to `python3 -m venv`; it's safe to re-run.

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

These import `client.py` directly and don't need `dlib`, a camera, or a running server. `tests/test_containerfile.py` is separate: it requires Podman and a built `face-recognition` image (`./build.sh`) and skips automatically if Podman isn't installed.

---

### Running in Podman (live camera)

`./run.sh` builds and runs the client in a **Podman** container (team standard — not
Docker) with your host's cameras and X display forwarded in, for live camera mode:

```bash
./build.sh   # first time / after code changes — installs Podman if missing, then builds (dlib compile ~10-15 min)
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

All settings live in `config.yaml` (the default, dlib model). Edit this file before
running, or use the ready-made `config.yunet.yaml` for the YuNet+MobileFaceNet model.

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
> **Want the YuNet + MobileFaceNet model instead of the default dlib one?**
> Use the ready-made preset instead of editing `config.yaml`:
> ```bash
> .venv/bin/python client.py --config config.yunet.yaml <image>
> ```
> It talks to the same mock server (`localhost:8000`) — only the face model differs.
> The mock server enrols both models per registration, so either works.
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
Using the **default dlib model**, the client detects the face, computes a 128-dim dlib descriptor, POSTs it to `http://localhost:8000/api/v1/identify/`, and prints the server's answer:
```
[INFO] MODE: server (http://localhost:8000)  <- alice.jpg
[INFO] Face: bbox=(...)  score=0.98
[INFO] Embedding: 128-dim
[INFO] Server -> name='Alice Kumar' confidence = 0.83 distance = 0.34
>>> Recognised: Alice Kumar
```
- **`distance`** is L2 (lower = better); under the model's cutoff (`0.6` for dlib, `0.8` for MobileFaceNet) = a match.
- To use the **YuNet + MobileFaceNet** model instead, add `--config config.yunet.yaml` (the log then shows `Embedding: 512-dim`). Both models work because the server enrolled both.
- If the server is unreachable, the client automatically falls back to the local diagnostic DB.

### Tutorial B — offline diagnostic mode (no server)

Register faces into a local SQLite DB and match against it — no server, no network.

```bash
# Register people (default dlib model)
.venv/bin/python client.py --register Alice alice.jpg
.venv/bin/python client.py --register Bob   bob.jpg

# Identify against them
.venv/bin/python client.py --diag alice2.jpg
# -> >>> Recognised: Alice  (cosine sim=0.79)
```
On an unknown face it prints the top-3 nearest matches with similarity scores. Face crops are saved under `diagnostic_mode/faces/`.

> Keep the model consistent within the local DB: faces registered under dlib (128-dim) only match dlib queries, and mobilefacenet (512-dim) only matches mobilefacenet. Pick one model (don't mix `config.yaml` and `config.yunet.yaml` against the same local DB) — or re-register if you switch.

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
- For **512-dim L2-normalised** embeddings instead (matching `AurafaceBackend`, cutoff `auraface_threshold=0.8`), pass `--config config.yunet.yaml` (or set `detection.detector: yunet` + `embedder.model: mobilefacenet`)
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
- Cross-checked against `mock-server/app/core/face_engine.py`, which now implements **both** backends:
  `FaceEngine` (YuNet + MobileFaceNet — same model files and preprocessing as here, mirroring `AurafaceBackend`)
  and `DlibEngine` (dlib HOG + ResNet, mirroring `DlibBackend` — byte-for-byte the same vectors this client's
  dlib path produces). That's why every mock-server registration enrols one of each, and either model matches.
- The dlib-backend recognition flow from `am-fru-desktop-app-Fru-MacOs/face_recognition_pipeline.py`:
  - `DlibFaceDetector` — HOG frontal face detector (same `threshold`, `num_upsamples`)
  - `DlibEmbeddingExtractor` — own HOG re-detect + 5-point shape predictor + ResNet descriptor
  - `DiagnosticFacePipeline` — offline SQLite pipeline with cosine similarity search
  - `DatabaseManager.search_diagnostic_faces()` — vectorised cosine similarity, top-K logging
