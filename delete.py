#!/usr/bin/env python3
from face_client import FaceRecognitionClient, ClientError
import sys
import logging 

client = FaceRecognitionClient()

try:
    face_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    client.delete_face(face_id)
except ClientError as e:
    logging.error(e.message)
except ValueError as e:
    logging.error(str(e))
