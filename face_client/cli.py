import argparse 
import logging
import sys 

from .errors import ClientError
from .pipeline import FaceRecognitionClient

logger = logging.getLogger("face_client")

def _run(args, client: FaceRecognitionClient) -> None:
    if args.list:
        client.list_faces()
        return

    if args.delete is not None:
        client.delete_face(args.delete)
        return

    if not args.camera and not args.image:
        args.camera = True

    if args.camera:
        client.run_camera()
    elif args.register:
        client.register(args.register, args.image)
    elif args.diag:
        client.identify(args.image, mode="diagnostic")
    elif args.server:
        client.identify(args.image, mode="server")
    else:
        client.identify(args.image)

def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python client.py photo.jpg # use mode from config.yaml
            python client.py --server photo.jpg # force server mode
            python client.py --diag photo.jpg # force diagnostic mode
            python client.py --register Alice photo.jpg # register a face locally
            python client.py --list # list registered faces
            python client.py --delete 3 # delete face_id 3 from the local DB
            python client.py --config custom.yaml photo.jpg # use a different config
                """,
      )

    parser.add_argument("image", nargs="?", help="Path to the input face image")
    parser.add_argument("--server", action="store_true", help="Force server mode")
    parser.add_argument("--diag", action="store_true", help="Force diagnostic (local SQLite) mode")
    parser.add_argument("--camera", action="store_true", help="Use live camera feed")
    parser.add_argument("--register", metavar="NAME",      help="Register a face with the given name")
    parser.add_argument("--list", action="store_true", help="List all registered faces in local DB")
    parser.add_argument("--delete", type=int, default=None, metavar="FACE_ID", help="Delete a face by ID from the local DB")
    parser.add_argument("--config", default=None, metavar="FILE", help="Config file (default: config.yaml)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    client = FaceRecognitionClient(args.config)
    cfg = client.cfg
    logging.getLogger().setLevel(cfg.log_level)
    logger.setLevel(cfg.log_level)

    logger.info("Config: %s  |  mode=%s  |  server=%s", args.config or cfg.config_path, cfg.mode, cfg.server_url)

    try:
        _run(args, client)
    except ClientError as e:
        logger.error("%s", e)
        sys.exit(1)    

if __name__ == "__main__":
    main()                                             