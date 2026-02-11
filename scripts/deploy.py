"""
Zero-Downtime Deployment Script
---------------------------------
Downloads new model artifacts, validates them, triggers hot-reload
on the running FastAPI backend, then verifies health.

Usage:
    python -m scripts.deploy --source ./new_artifacts
    python -m scripts.deploy --source gs://bucket/models/latest
    python -m scripts.deploy --source s3://bucket/models/latest
"""

import sys
import os
import argparse
import json
import shutil
import logging
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts_backup")
API_URL = "http://127.0.0.1:8000"


def download_from_gcs(gcs_path: str, local_dir: str):
    """Download artifacts from Google Cloud Storage."""
    try:
        from google.cloud import storage
        # Parse gs://bucket/prefix
        parts = gcs_path.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        os.makedirs(local_dir, exist_ok=True)
        for blob in blobs:
            filename = blob.name.split("/")[-1]
            if filename:
                local_path = os.path.join(local_dir, filename)
                blob.download_to_filename(local_path)
                logger.info(f"Downloaded: {filename}")
    except ImportError:
        logger.error("google-cloud-storage not installed")
        sys.exit(1)


def download_from_s3(s3_path: str, local_dir: str):
    """Download artifacts from AWS S3."""
    try:
        import boto3
        parts = s3_path.replace("s3://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        os.makedirs(local_dir, exist_ok=True)

        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]
                if filename:
                    local_path = os.path.join(local_dir, filename)
                    s3.download_file(bucket_name, key, local_path)
                    logger.info(f"Downloaded: {filename}")
    except ImportError:
        logger.error("boto3 not installed")
        sys.exit(1)


def validate_artifacts(artifact_dir: str) -> bool:
    """Validate that required artifact files exist and are valid."""
    required_files = ["config.json", "feature_columns.json", "ticker_list.json"]
    model_files = ["model.pkl", "model.pt"]

    for f in required_files:
        path = os.path.join(artifact_dir, f)
        if not os.path.exists(path):
            logger.error(f"Missing required file: {f}")
            return False
        # Validate JSON
        try:
            with open(path, "r") as fh:
                json.load(fh)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {f}: {e}")
            return False

    # Check at least one model file exists
    has_model = any(
        os.path.exists(os.path.join(artifact_dir, f)) for f in model_files
    )
    if not has_model:
        logger.error("No model file found (model.pkl or model.pt)")
        return False

    # Validate config
    with open(os.path.join(artifact_dir, "config.json"), "r") as f:
        config = json.load(f)
    if "model_type" not in config:
        logger.error("config.json missing 'model_type'")
        return False

    # Validate sentiment model availability if config uses dual sentiment
    if not validate_sentiment_model(config):
        return False

    # Check sentiment IC if metrics are present
    metrics_path = os.path.join(artifact_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            sentiment_ic = metrics.get("sentiment_ic", None)
            if sentiment_ic is not None and sentiment_ic < 0:
                logger.error(
                    "Sentiment IC is negative (%.4f) -- refusing deployment. "
                    "Retrain with updated sentiment data.",
                    sentiment_ic,
                )
                return False
        except (json.JSONDecodeError, TypeError):
            pass  # No metrics file or invalid -- skip check

    logger.info("Artifact validation passed")
    return True


def validate_sentiment_model(config: dict) -> bool:
    """Validate that sentence-transformers is available if needed.

    Args:
        config: Model config dict from config.json.

    Returns:
        True if validation passes.
    """
    sentiment_config = config.get("sentiment_engine", {})
    uses_sentence_model = sentiment_config.get("use_sentence_embedding", False)

    if uses_sentence_model:
        try:
            import sentence_transformers  # noqa: F401
            logger.info("sentence-transformers dependency verified")
        except ImportError:
            logger.error(
                "Config requires sentence-transformers but it is not installed. "
                "Install with: pip install sentence-transformers"
            )
            return False

    return True


def backup_current(artifact_dir: str, backup_dir: str):
    """Backup current artifacts."""
    if os.path.exists(artifact_dir):
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(artifact_dir, backup_dir)
        logger.info(f"Backed up to {backup_dir}")


def deploy_artifacts(source_dir: str, target_dir: str):
    """Copy new artifacts to the target directory."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    logger.info(f"Deployed to {target_dir}")


def trigger_hot_reload(api_url: str) -> bool:
    """Trigger model hot-reload via API."""
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"{api_url}/api/v1/health",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "healthy"
    except Exception as e:
        logger.warning(f"Hot-reload check failed: {e}")
        return False


def verify_health(api_url: str, max_retries: int = 5) -> bool:
    """Verify the API is healthy after deployment."""
    for i in range(max_retries):
        try:
            import urllib.request
            req = urllib.request.Request(f"{api_url}/api/v1/health")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "healthy":
                    logger.info(f"Health check passed (attempt {i + 1})")
                    return True
        except Exception:
            pass
        logger.info(f"Waiting for API... (attempt {i + 1}/{max_retries})")
        time.sleep(2)
    return False


def rollback(backup_dir: str, artifact_dir: str):
    """Rollback to previous artifacts."""
    if os.path.exists(backup_dir):
        if os.path.exists(artifact_dir):
            shutil.rmtree(artifact_dir)
        shutil.copytree(backup_dir, artifact_dir)
        logger.info("Rolled back to previous version")
    else:
        logger.error("No backup available for rollback")


def main():
    parser = argparse.ArgumentParser(description="Deploy model artifacts")
    parser.add_argument("--source", required=True,
                        help="Source: local dir, gs://..., or s3://...")
    parser.add_argument("--target", default=ARTIFACT_DIR,
                        help="Target artifact directory")
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--skip-health-check", action="store_true")

    args = parser.parse_args()

    # 1. Download if remote
    staging_dir = os.path.join(os.path.dirname(args.target), "artifacts_staging")
    source = args.source

    if source.startswith("gs://"):
        logger.info(f"Downloading from GCS: {source}")
        download_from_gcs(source, staging_dir)
        source = staging_dir
    elif source.startswith("s3://"):
        logger.info(f"Downloading from S3: {source}")
        download_from_s3(source, staging_dir)
        source = staging_dir
    else:
        if not os.path.isdir(source):
            logger.error(f"Source directory not found: {source}")
            sys.exit(1)

    # 2. Validate
    if not args.skip_validate:
        if not validate_artifacts(source):
            logger.error("Validation failed. Aborting deployment.")
            sys.exit(1)

    # 3. Backup current
    backup_current(args.target, BACKUP_DIR)

    # 4. Deploy
    deploy_artifacts(source, args.target)

    # 5. Verify health
    if not args.skip_health_check:
        if verify_health(args.api_url):
            logger.info("Deployment successful!")
        else:
            logger.error("Health check failed after deployment. Rolling back...")
            rollback(BACKUP_DIR, args.target)
            sys.exit(1)
    else:
        logger.info("Deployment complete (health check skipped)")

    # Cleanup staging
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)


if __name__ == "__main__":
    main()
