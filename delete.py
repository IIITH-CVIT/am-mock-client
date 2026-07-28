#!/usr/bin/env python3
from face_client import FaceRecognitionClient, ClientError
import sys

client = FaceRecognitionClient()
face_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

try:
    client.delete_face(face_id)
except (ClientError, ValueError) as e:
    print(f"Error: {e}")
