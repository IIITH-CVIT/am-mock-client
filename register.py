#!/usr/bin/env python3 
from face_client import FaceRecognitionClient, ClientError
import sys 

client = FaceRecognitionClient()
name = sys.argv[1] if len(sys.argv) > 1 else "Ekanth"
image_path = sys.argv[2] if len(sys.argv) > 2 else "psp_photo.jpeg"

try:
    client.register(name, image_path)
except ClientError as e:
    print(f"Error: {e}")