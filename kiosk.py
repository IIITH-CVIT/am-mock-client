#!/usr/bin/env python3
"""
Kiosk-mode entry point — see KIOSK_INTEGRATION.md / README "Kiosk mode".

Exit codes (branch on these, never on stdout text):
    0  recognized — name is the ONLY line on stdout
    1  no match within timeout (unknown face / no face / SIGTERM|SIGINT — all the same)
    2  hardware/model error — message on stderr

Usage:
    .venv/bin/python kiosk.py        # timeout from config.yaml's kiosk.timeout_seconds
    .venv/bin/python kiosk.py 20     # override to 20s for this run
"""

from face_client import FaceRecognitionClient, ClientError
import sys
import logging 

client = FaceRecognitionClient()

timeout = None 
if len(sys.argv) > 1:
    try:
        timeout = float(sys.argv[1])
    except ValueError:
        logging.error("Invalid timeout argument: %r", sys.argv[1])
        sys.exit(2)

try:
    name = client.identify_kiosk(timeout = timeout)
except ClientError as e:
    logging.error(e.message)
    sys.exit(2)

if name:
    print(name)
    sys.exit(0)
sys.exit(1)
