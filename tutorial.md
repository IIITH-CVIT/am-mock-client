# Face Recognition Client Tutorial

For the full business logic, modes, and troubleshooting, see [README.md](README.md).

---

## What this is

A Python client that detects a face (in a photo or live camera feed) and recognises the person.

---

## 1. Prerequisites

- The **mock server** running and reachable at `http://localhost:8000` (see the `am-mock-server` repo's `tutorial.md`), with at least one face registered.

---

## 2. Set up the client

Go to the command line interface and type the following command:

```bash
./setup.sh           # creates a native .venv and installs everything
```

---

## 3. Using the minimal scripts (`identify.py` / `register.py` / `delete.py`)

The simplest way to use the client — three small files at the repo root, each
doing one thing. Edit the values directly in the file, or pass them on the
command line — either works.

**Register a face:**

```bash
.venv/bin/python register.py <Name> path/to/photo.jpg
```

**Identify a face:**

```bash
.venv/bin/python identify.py path/to/photo.jpg
```

Expected output (via the server, or the local diagnostic DB if the server's unreachable):

```
>>> Recognised: <Name>
```

**Delete a registered face**:

```bash
.venv/bin/python client.py --list
.venv/bin/python delete.py <face_id>
```

Under the hood, each script just does this — the whole interface in one line:

```python
from face_client import FaceRecognitionClient

client = FaceRecognitionClient()
client.register("Alice", "alice.jpg")   # register.py
client.identify("alice2.jpg")           # identify.py
client.delete_face(1)                   # delete.py
```

---

## 4. Using the full CLI (`client.py`)

The same actions, plus a few extras (live camera, forcing server/diagnostic
mode, switching config files), all through one command:

```bash
# Register
.venv/bin/python client.py --register <Name> path/to/photo.jpg

# Identify against the server (auto-falls back to the local DB if unreachable)
.venv/bin/python client.py --server path/to/photo.jpg
```

Expected output:

```
[INFO] MODE: server (http://localhost:8000)  <- alice.jpg
[INFO] Face: bbox=(...)  score=0.98
[INFO] Embedding: 128-dim
[INFO] Server -> name='Alice Kumar' confidence=0.83 distance=0.34
>>> Recognised: Alice Kumar
```

```bash
# List everything registered locally
.venv/bin/python client.py --list

# Delete by face_id (from --list above)
.venv/bin/python client.py --delete 1

# Force diagnostic (local-only) mode
.venv/bin/python client.py --diag alice.jpg

# Live camera feed
.venv/bin/python client.py --camera
```

---

## 5. Other `config.yaml` settings (optional)

- **Point at a different server**: the client defaults to the mock server at
  `http://localhost:8000`. To use a real `am-master-server`, edit `server.url`
  in `config.yaml`.

- **Switch camera backend**: if you change what hardware `--camera` mode runs
  on, edit `camera.backend` — `opencv` (default, works for USB/laptop
  webcams) or `picamera2` (Raspberry Pi 5, or any libcamera-only CSI camera —
  see README's "Running on Raspberry Pi 5" section for the one-time setup this
  needs).

- **Switch embedder model**: to change what embeddings the client produces,
  edit `embedder.model` — `sface` (default, 128-dim) or `auraface` (512-dim).
  This **must match** whatever the server is enrolled with — the two ready-made
  presets already exist so you don't need to hand-edit this: run with
  `--config config.auraface.yaml` instead of editing `config.yaml` directly if
  you need the `auraface` model.

---

That's it and you're running. For everything else (modes in detail, config
reference, models, known issues), see [README.md](README.md).
