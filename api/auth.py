"""
API Security Module
--------------------
API key authentication, rate limiting, and artifact integrity verification.
"""

from typing import Dict, List, Optional
import hashlib
import hmac
import time
import secrets
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API Key Middleware
# ---------------------------------------------------------------------------

try:
    from fastapi import Request, HTTPException
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class APIKeyMiddleware(BaseHTTPMiddleware):
        """Middleware that validates API keys in the X-API-Key header.

        Skips authentication for excluded paths (e.g., /health, /docs).
        """

        def __init__(
            self,
            app,
            api_keys: Optional[List[str]] = None,
            exclude_paths: Optional[List[str]] = None,
        ):
            super().__init__(app)
            self.api_keys = api_keys or []
            self.exclude_paths = exclude_paths or [
                "/health", "/docs", "/openapi.json", "/redoc", "/",
                "/api/v1/health",
            ]

        async def dispatch(self, request: Request, call_next):
            # Skip excluded paths
            path = request.url.path
            for excluded in self.exclude_paths:
                if path.startswith(excluded):
                    return await call_next(request)

            # Skip if no API keys configured (open access)
            if not self.api_keys:
                return await call_next(request)

            # Check X-API-Key header
            api_key = request.headers.get("X-API-Key", "")
            if not verify_api_key(api_key, self.api_keys):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid or missing API key"},
                )

            return await call_next(request)

except ImportError:
    # FastAPI not installed; provide stub
    class APIKeyMiddleware:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass


def verify_api_key(api_key: str, valid_keys: List[str]) -> bool:
    """Verify an API key against a list of valid keys.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        api_key: The key to verify.
        valid_keys: List of valid API keys.

    Returns:
        True if the key is valid.
    """
    if not api_key or not valid_keys:
        return False

    for valid_key in valid_keys:
        if hmac.compare_digest(api_key.encode("utf-8"), valid_key.encode("utf-8")):
            return True
    return False


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """In-memory token bucket rate limiter.

    Limits requests per client within a sliding time window.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        """
        Args:
            max_requests: Maximum requests per window.
            window_seconds: Time window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets = {}  # type: Dict[str, List[float]]

    def check(self, client_id: str) -> bool:
        """Check if a request is allowed.

        Args:
            client_id: Client identifier (e.g., IP address or API key).

        Returns:
            True if request is allowed, False if rate exceeded.
        """
        now = time.time()
        cutoff = now - self.window_seconds

        if client_id not in self._buckets:
            self._buckets[client_id] = []

        # Remove expired timestamps
        self._buckets[client_id] = [
            t for t in self._buckets[client_id] if t > cutoff
        ]

        if len(self._buckets[client_id]) >= self.max_requests:
            return False

        self._buckets[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for a client."""
        now = time.time()
        cutoff = now - self.window_seconds

        if client_id not in self._buckets:
            return self.max_requests

        active = [t for t in self._buckets[client_id] if t > cutoff]
        return max(self.max_requests - len(active), 0)


# ---------------------------------------------------------------------------
# Artifact Integrity
# ---------------------------------------------------------------------------

def compute_artifact_checksum(filepath: str) -> str:
    """Compute SHA256 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        Hex-encoded SHA256 hash string.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_artifact_checksum(filepath: str, expected_hash: str) -> bool:
    """Verify a file's SHA256 checksum.

    Args:
        filepath: Path to the file.
        expected_hash: Expected SHA256 hex string.

    Returns:
        True if the checksum matches.
    """
    actual = compute_artifact_checksum(filepath)
    return hmac.compare_digest(actual, expected_hash)


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(32)
