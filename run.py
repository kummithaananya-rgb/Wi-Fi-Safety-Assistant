"""
Production launcher – starts the app under Waitress (Windows-compatible WSGI server).

Usage:
    python run.py              # uses HOST/PORT from .env (default 0.0.0.0:5000)
    python run.py --port 8080  # override port
"""
import sys
import os
import argparse

# Put backend/ on the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from config import Config, setup_logging

setup_logging()

import logging
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Wi-Fi Safety Assistant – Waitress server")
    parser.add_argument("--host", default=Config.HOST, help="Bind address (default from .env)")
    parser.add_argument("--port", type=int, default=Config.PORT, help="Port (default from .env)")
    parser.add_argument("--threads", type=int, default=4, help="Waitress thread count")
    args = parser.parse_args()

    from wsgi import application  # backend/ is already on sys.path

    logger.info("=" * 60)
    logger.info("Wi-Fi Safety Assistant – Production Server (Waitress)")
    logger.info(f"  Listening : http://{args.host}:{args.port}")
    logger.info(f"  Threads   : {args.threads}")
    logger.info(f"  Granite   : {'configured' if Config.has_granite() else 'NOT configured'}")
    logger.info(f"  Gemini    : {'configured' if Config.has_gemini() else 'NOT configured'}")
    logger.info("=" * 60)

    from waitress import serve
    serve(
        application,
        host=args.host,
        port=args.port,
        threads=args.threads,
        channel_timeout=60,        # close idle connections after 60s
        cleanup_interval=10,
        connection_limit=100,
        url_scheme="http",
        ident="WifiSafetyAssistant",
    )


if __name__ == "__main__":
    main()
