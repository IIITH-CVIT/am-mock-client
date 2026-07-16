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

## 3. Identify a photo (against the server)

Use a photo of someone you registered on the server:

Go to the command line interface and type the following command:

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

---

## 4. Other common commands

**List locally registered faces:**

```bash
.venv/bin/python client.py --list
```

---

## 5. Point at a different server (optional)

The client defaults to the mock server at `http://localhost:8000`. To use a real `am-master-server`, edit `server.url` in `config.yaml`.

---

That's it and you're running. For everything else (modes in detail, config
reference, models, known issues), see [README.md](README.md).
