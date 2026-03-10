"""Script to test API with sample audio."""

import argparse
import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test API with audio file."""
    parser = argparse.ArgumentParser(description="Test API endpoint")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)

    # Check health
    logger.info("Checking API health...")
    try:
        response = requests.get(f"{args.url}/health")
        response.raise_for_status()
        health = response.json()
        logger.info(f"Health status: {health}")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sys.exit(1)

    # Transcribe
    logger.info(f"Transcribing audio: {audio_path}")
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            data = {}
            if args.language:
                data["language"] = args.language

            response = requests.post(
                f"{args.url}/transcribe",
                files=files,
                data=data,
            )
            response.raise_for_status()

        result = response.json()

        logger.info("\n" + "=" * 60)
        logger.info("TRANSCRIPTION RESULT")
        logger.info("=" * 60)
        logger.info(f"Text: {result['text']}")
        logger.info(f"Processing time: {result['processing_time_ms']:.2f} ms")
        logger.info(f"Model: {result['model_name']}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
