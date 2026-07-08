from unittest.mock import MagicMock, patch
import numpy as np 
from client import ServerClient, Config 

def _client_with_mocked_post(respinse_json):
    cfg = Config.__new__(Config)
    cfg._data = {"server": {"url": "http://test", "timeout": 5}}
    client = ServerClient(cfg)
    client._post = MagicMock(return_value=response_json)
    return client

def test_identify_reads_real_schema_fields():
    """Locks in the actual IdentifyResponse field names: name/confidence/distance.
    If this fails, either the server's schema changed or someone reintroduced
    the dead visitor_name/similarity fallback."""
    response = {
        "registration_id": "abc-123",
        "name": "Alice",
        "confidence": 0.82,
        "distance": 0.35,
        "match_type": "face",
        "message": "match found",
    }
    client = _client_with_mocked_post(response)
    result = client.identify(np.zeros(512, dtype=np.float32))
    assert result == "Alice"

def test_identify_does_not_use_dead_fallback_keys():
    """A response with ONLY the dead keys populated should NOT resolve a name,
    proves the fallback is gone, not just deprioritized."""
    response = {
        "name": None,
        "visitor_name": "Bob",     # legacy/nonexistent field, must be ignored
        "distance": None,
        "similarity": 0.9,          # legacy/nonexistent field, must be ignored
        "message": "no match found",
    }
    client = _client_with_mocked_post(response)
    result = client.identify(np.zeros(512, dtype=np.float32))
    assert result is None

def test_identify_handles_server_error():
    client = _client_with_mocked_post({"error": "connection refused"})
    result = client.identify(np.zeros(512, dtype=np.float32))
    assert result is None