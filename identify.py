#!/usr/bin/env python3

from face_client import FaceRecognitionClient, ClientError
import sys

client = FaceRecognitionClient()
image_path = sys.argv[1] if len(sys.argv) > 1 else "psp_photo.jpeg"

try:
    client.identify(image_path)
except ClientError as e:
    print(f"Error: {e}")
