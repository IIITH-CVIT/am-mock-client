# Face Recognition Client Tutorial

For the full business logic, modes, and troubleshooting, see [README.md](README.md).

---

## What this is

A Python client that detects a face (in a photo or live camera feed), computes an embedding, and identifies the person - either against a remote server or a local offline SQLite database.

---

## 1. Prerequisites

- Python 3.13.
- The **mock server** running and reachable at `http://localhost:8000` (see the `am-mock-server` repo's `tutorial.md`), with at least one face registered.
- The ONNX models are already bundled in `./models` and nothing to download.

---

## 2. Set up the client

```bash
./setup.sh           # creates a native .venv and installs everything
```

This bootstraps a `.venv` and installs all dependencies. No `sudo`, no manual pip steps.

> The default install includes **dlib**, which compiles from source (~10-15 min).
> If you only plan to use the YuNet + MobileFaceNet model, run `./setup.sh --light` instead (fast, no compile) and always pass `--config config.yunet.yaml`.

---

## 3. Identify a photo (against the server)

Use a photo of someone you registered on the server:

```bash
.venv/bin/python client.py --server alice.jpg
```

Expected output:

```
[INFO] MODE: server (http://localhost:8000)  <- alice.jpg
[INFO] Face: bbox=(...)  score=0.98
[INFO] Embedding: 128-dim
[INFO] Server -> name='Alice Kumar' confidence=0.83 distance=0.34
>>> Recognised: Alice Kumar
```

- `distance` is lower = better; under the model's cutoff counts as a match.
- If the server is unreachable, the client automatically falls back to the local diagnostic database which uses the **default dlib model**. To use YuNet + MobileFaceNet instead, add `--config config.yunet.yaml` (the log then shows `Embedding: 512-dim`).

---

## 4. Other common commands

**Live camera** — detects and identifies faces on your webcam, overlaying the name. Press `q` to quit:

```bash
.venv/bin/python client.py --camera      # or just: client.py (camera is default)
```

**Offline diagnostic mode** — register and match locally, no server needed:

```bash
# Register people into the local SQLite DB
.venv/bin/python client.py --register Alice alice.jpg
.venv/bin/python client.py --register Bob   bob.jpg

# Identify against them
.venv/bin/python client.py --diag alice2.jpg
# -> >>> Recognised: Alice  (cosine sim=0.79)
```

**List locally registered faces:**

```bash
.venv/bin/python client.py --list
```

---

## 5. Point at a different server (optional)

The client defaults to the mock server at `http://localhost:8000`. To use a real `am-master-server`, edit `server.url` in `config.yaml`.

---

## Notes

- **Keep the model paired.** dlib detector goes with dlib embedder (128-dim); YuNet goes with MobileFaceNet (512-dim). Don't mix them. Switch the whole pair by passing `--config config.yunet.yaml`.
- The single-image commands (`--server`, `--register`, `--list`, `--diag`) run natively - no container or display needed. The Podman + X11 setup is only for live-camera mode; see [README.md](README.md) if you need it.

---

That's it and you're running. For everything else (modes in detail, config
reference, models, known issues), see [README.md](README.md).
