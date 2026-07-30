#!/usr/bin/env python3

from face_client import FaceRecognitionClient, ClientError
import sys
import logging

client = FaceRecognitionClient()
image_path = sys.argv[1] if len(sys.argv) > 1 else "path/to/photo.jpeg"

try:
    client.identify(image_path)
except ClientError as e:
    logging.error(e.message)
