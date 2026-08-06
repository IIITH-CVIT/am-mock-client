# Kiosk Integration — Face Recognition Client

A standalone one-pager for the kiosk team: what to call, what it returns, and why it's built this way. Full details live in the "Kiosk mode" section of `README.md`.

---

## What it does

`kiosk.py` is a bounded, single-shot face recognition pass. Invoke it once per person (e.g. when your presence sensor fires); it opens the camera, tries to recognize a face until either a match is found or a timeout elapses, then **always** releases the camera and exits.

## Why it always exits, instead of pausing and resuming

Your main algorithm needs the same camera device right after ours is done with it. That rules out holding the camera open in a paused/frozen process — a frozen process still keeps the device locked, which would block you from opening it. So `kiosk.py` fully releases the camera and exits on every outcome: a match, no match, a timeout, or even if you kill the process with a signal.

## How to call it

```bash
.venv/bin/python kiosk.py         # uses the default timeout (currently 15s)
.venv/bin/python kiosk.py 20      # override to a 20-second timeout for this run
```

## Exit codes — branch on these, not on stdout text

| Exit code | Meaning |
|---|---|
| `0` | Face recognized — the name is the *only* line printed on stdout |
| `1` | No match within the timeout (unknown face, no face at all, or the process was interrupted by `SIGTERM`/`SIGINT` — all treated the same) |
| `2` | Hardware/model error (camera not found, a model file is missing, etc.) |

## Recommended bash pattern

Capture `$?` immediately — don't run any other command between the `if` and reading it, or you'll read the wrong exit code:

```bash
if name=$(.venv/bin/python kiosk.py); then
    echo "Recognized: $name"
    # ... start your main algorithm here — the camera is free now ...
else
    rc=$?
    if [ "$rc" -eq 2 ]; then
        echo "Hardware/model error — check camera/models" >&2
    else
        echo "No one recognized within the timeout"
    fi
fi
```

## Config

Two settings control kiosk mode, in `config.yaml` (or `config.auraface.yaml`):

```yaml
kiosk:
  timeout_seconds: 15   # give up and exit 1 if nobody is recognized within this many seconds
  frame_skip: 1          # run recognition every N frames (1 = every frame)
```

`timeout_seconds` is a placeholder — let's tune it together once your algorithm's own timing is settled. You don't need to edit the file to test different values; just pass one as an argument: `kiosk.py <seconds>`.

## Try it yourself

`scripts/demo_kiosk_cycle.sh` simulates your whole Step 1→5 sequence end-to-end (with your hardware presence sensor and main algorithm stubbed out) — run it to see the exit codes and camera hand-off working live:

```bash
./scripts/demo_kiosk_cycle.sh
```
