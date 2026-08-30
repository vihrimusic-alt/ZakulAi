"""Inline test results or private S3-compatible durable audio, never local URLs."""

import base64
import hashlib
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

# Base64 expansion stays below RunPod /run's 10 MB response limit.
MAX_INLINE_BYTES = 5_000_000
CONTENT_TYPES = {"mp3": "audio/mpeg", "flac": "audio/flac"}


class ResultStore:
    """Publish only to an operator-configured bucket; request input cannot choose a destination."""

    def __init__(self, mode: str, job_id: str):
        """Validate all destination settings before loading any GPU model."""
        self.mode = mode
        self.inline_bytes = 0
        self.client = None
        self.job_key = hashlib.sha256(job_id.encode()).hexdigest()
        self.bucket = os.environ.get("ZAKUL_S3_BUCKET", "").strip()
        self.prefix = os.environ.get("ZAKUL_S3_PREFIX", "zakul/results").strip("/")
        self.expires = 3600
        if mode == "s3":
            self._configure_s3()

    def _configure_s3(self) -> None:
        import boto3
        from botocore.config import Config

        endpoint = os.environ.get("ZAKUL_S3_ENDPOINT_URL", "").strip()
        key = os.environ.get("ZAKUL_S3_ACCESS_KEY_ID", "").strip()
        secret = os.environ.get("ZAKUL_S3_SECRET_ACCESS_KEY", "").strip()
        if not self.bucket or not key or not secret:
            raise ValueError("S3 mode needs ZAKUL_S3_BUCKET, ACCESS_KEY_ID and SECRET_ACCESS_KEY")
        if endpoint:
            parsed = urlsplit(endpoint)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("ZAKUL_S3_ENDPOINT_URL must be a credential-free HTTPS origin")
            if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
                raise ValueError("ZAKUL_S3_ENDPOINT_URL must be an origin, not a file URL")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/-]{0,150}", self.prefix):
            raise ValueError("ZAKUL_S3_PREFIX contains unsupported characters")
        if any(part in {"", ".", ".."} for part in self.prefix.split("/")):
            raise ValueError("ZAKUL_S3_PREFIX contains invalid path segments")
        self.client = boto3.client(
            "s3", endpoint_url=endpoint or None,
            region_name=os.environ.get("ZAKUL_S3_REGION", "us-east-1"),
            aws_access_key_id=key, aws_secret_access_key=secret,
            config=Config(
                signature_version="s3v4", connect_timeout=15, read_timeout=120,
                retries={"mode": "standard", "max_attempts": 3},
                s3={"addressing_style": "path"},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )

    def publish(self, assets: dict[str, Path], index: int, duration: float, seed: int) -> dict:
        """Wait for uploads before returning a track; inline tests contain MP3 only."""
        track = {"index": index, "duration_seconds": duration, "seed": seed}
        for format_name, path in assets.items():
            size = path.stat().st_size
            descriptor = {
                "filename": f"take-{index}.{format_name}",
                "content_type": CONTENT_TYPES[format_name], "size_bytes": size,
            }
            if self.mode == "inline":
                if format_name != "mp3" or self.inline_bytes + size > MAX_INLINE_BYTES:
                    raise ValueError("Inline audio exceeds the safe payload limit; use S3")
                self.inline_bytes += size
                descriptor["base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
            else:
                object_key = f"{self.prefix}/{self.job_key}/take-{index}.{format_name}"
                self.client.upload_file(
                    str(path), self.bucket, object_key,
                    ExtraArgs={"ContentType": CONTENT_TYPES[format_name]},
                )
                descriptor.update({
                    "bucket": self.bucket, "key": object_key,
                    "url": self.client.generate_presigned_url(
                        "get_object", Params={"Bucket": self.bucket, "Key": object_key},
                        ExpiresIn=self.expires,
                    ),
                    "url_expires_at": (
                        datetime.now(timezone.utc) + timedelta(seconds=self.expires)
                    ).isoformat(),
                })
            track[format_name] = descriptor
        return track
