"""Tests for kiosk mode: _kiosk_identify_loop and FaceRecognitionClient.identify_kiosk().

  _kiosk_identify_loop is pure/dependency-injected (no camera, no
  ServerClient/DiagnosticDB), so most of these run against fakes with no I/O at
  all. The identify_kiosk() tests (further down) exist specifically to prove
  the camera is released on every exit path — the entire reason this feature
  exists (the kiosk's own "main algorithm" needs the same camera device
  immediately after this one releases it).
  """

import time 
import pytest

from face_client.config import Config 
from face_client.errors import ClientError
from face_client.pipeline import _kiosk_identify_loop
import face_client.pipeline as pipeline 
from face_client.pipeline import FaceRecognitionClient

def _cfg(data):
    """Build a Config without running __init__ (no config file needed)."""
    cfg = Config.__new__(Config)
    cfg._data = data
    return cfg 

class _NeverMatchDB:
    """Fake DiagnosticDB that never matches. Just proves the search() call site works."""
    def __init__(self, cfg):
        pass

    def search(self, vec):
        return(None, None, 0.0)

def test_kiosk_loop_resolves_on_early_match():
    call_count = {"identify": 0}

    class _FakeDetector:
        def detect(self, frame):
            return object() # any truthy will return "face was found"
    
    class _FakeEmbedder:
        def embed(self, frame, detection):
            return "vec"
    
    def identify_fn(vec):
        call_count["identify"] += 1
        return "Alice"
    
    name = _kiosk_identify_loop(read_frame=lambda:"frame", detector = _FakeDetector(), embedder = _FakeEmbedder(), identify_fn = identify_fn, deadline = time.monotonic() + 1000) # deadline is far i n the future, irrelevant here

    assert name == "Alice"
    assert call_count["identify"] == 1

def test_kiosk_loop_returns_none_after_deadline():
    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0]) # each "now" call advanced by 1.0

    class _FakeDetector:
        def detect(self, frame):
            return None # no face in this frame which is the common case
    class _FakeEmbedder:
        def embed(self, frame, detection):
            raise AssertionError("embed() must not run when no face was detected")
    
    name = _kiosk_identify_loop(read_frame=lambda:"frame", detector = _FakeDetector(), embedder = _FakeEmbedder(), identify_fn = lambda vec: None, deadline = 3.0, now = lambda: next(ticks))

    assert name is None

def test_kiosk_loop_stops_early_via_should_stop():
    call_state = {"stop_checks": 0, "reads": 0}

    def should_stop():
        call_state["stop_checks"] += 1
        return call_state["stop_checks"] >= 2 # True from second check onwards
    
    def read_frame():
        call_state["reads"] += 1
        return "frame"
    
    class _FakeDetector:
        def detect(self, frame):
            return None
        
    name = _kiosk_identify_loop(read_frame = read_frame, detector = _FakeDetector(), embedder = None, identify_fn = lambda vec: None, deadline = time.monotonic() + 1000, should_stop = should_stop) # embedder is none as detection always misses, 

    assert name is None
    assert call_state["stop_checks"] == 2
    assert call_state["reads"] == 1 # stopped before a 2nd frame was used.

def test_kiosk_loop_raises_on_persistent_read_failure():
    with pytest.raises(ClientError, match = "stopped returning frames"):
        _kiosk_identify_loop(read_frame = lambda: None, detector = None, embedder = None, identify_fn = lambda vec: None,  deadline = time.monotonic() + 1000,max_consecutive_read_failures = 3)
    
def test_kiosk_loop_respects_frame_skip():
    detect_calls = []
    ticks = iter(range(20))

    class _FakeDetector:
        def detect(self, frame):
            detect_calls.append(frame)
            return None #Should never match, we only care which frames are checked
    
    _kiosk_identify_loop(read_frame = lambda: "frame", detector = _FakeDetector(), embedder = None, identify_fn = lambda vec: None, deadline = 9, now = lambda: next(ticks), frame_skip = 3)

    # detection should only fire on multiple of 3 frames
    assert len(detect_calls) == 3

def test_identify_kiosk_releases_camera_on_timeout(monkeypatch):
    state = {"released": False}

    def fake_open_source(device):
        def read():
            return "frame"

        def release():
            state["released"] = True
        
        return read, release

    monkeypatch.setattr(pipeline, "_open_opencv_source", fake_open_source)
    monkeypatch.setattr(pipeline, "DiagnosticDB", _NeverMatchDB)

    class _FakeDetector:
        def detect(self, frame):
            return None # It is not a face. I want it to return a timeout path
    
    cfg = _cfg({
        "mode": "diagnostic",
        "camera": {"backend": "opencv", "device": 0},
        "kiosk": {"timeout_seconds": 0.05, "frame_skip": 1},
    })

    client = FaceRecognitionClient.__new__(FaceRecognitionClient)
    client.cfg = cfg
    client._detector = _FakeDetector()
    client._embedder = object() # This should never be reached. Detection always misses first

    name = client.identify_kiosk()

    assert name is None
    assert state["released"] is True

def test_identify_kiosk_releases_camera_when_loop_raises(monkeypatch):
    state = {"released": False}

    def fake_open_source(device):
        def read():
            return None # camera never returns a frame resulting in a loop that raises ClientError

        def release():
            state["released"] = True
        
        return read, release
    
    monkeypatch.setattr(pipeline, "_open_opencv_source", fake_open_source)
    monkeypatch.setattr(pipeline, "DiagnosticDB", _NeverMatchDB)

    cfg = _cfg({
        "mode": "diagnostic",
        "camera": {"backend": "opencv", "device": 0},
        "kiosk": {"timeout_seconds": 5, "frame_skip": 1},
    })

    client = FaceRecognitionClient.__new__(FaceRecognitionClient)
    client.cfg = cfg
    client._detector = object() # This should never be reached. read_frame() always fails first
    client._embedder = object()

    with pytest.raises(ClientError):
        client.identify_kiosk()
    
    assert state["released"] is True

def test_identify_kiosk_treats_signal_as_no_match(monkeypatch):
    state = {"released": False}
    captured = {}

    def fake_signal(sig, handler):
        prev = captured.get(sig, lambda *a: None)
        captured[sig] = handler
        return prev 
    
    monkeypatch.setattr(pipeline.signal, "signal", fake_signal)

    call_count = {"reads": 0}

    def fake_open_source(device):
        def read():
            call_count["reads"] += 1
            if call_count['reads'] == 2:
                # Simulate the signal arriving between reads. Invoke the handler exactly the way the OS would.
                captured[pipeline.signal.SIGTERM](pipeline.signal.SIGTERM, None)
            return "frame"
        
        def release():
            state["released"] = True
        return read, release
    
    monkeypatch.setattr(pipeline, "_open_opencv_source", fake_open_source)
    monkeypatch.setattr(pipeline, "DiagnosticDB", _NeverMatchDB)

    class _FakeDetector:
        def detect(self, frame):
            return None 
    
    cfg = _cfg({
        "mode": "diagnostic",
        "camera": {"backend": "opencv", "device": 0},
        "kiosk": {"timeout_seconds": 5, "frame_skip": 1},
    })

    client = FaceRecognitionClient.__new__(FaceRecognitionClient)
    client.cfg = cfg 
    client._detector = _FakeDetector()
    client._embedder = object()

    name = client.identify_kiosk()

    assert name is None 
    assert state["released"] is True 
    assert call_count["reads"] == 2 # loop should stop right after the signal fired