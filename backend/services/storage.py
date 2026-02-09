"""Cloudflare R2 storage service for file uploads."""
import uuid
from typing import Optional
from contextlib import asynccontextmanager

import aioboto3

from core.config import config


class R2StorageService:
    """Service for interacting with Cloudflare R2 storage."""

    def __init__(self):
        self.session = aioboto3.Session()
        self.bucket = config.r2_bucket_name
        self.endpoint_url = config.r2_endpoint_url

    @asynccontextmanager
    async def _get_client(self):
        """Get an async S3 client configured for R2."""
        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=config.r2_access_key_id,
            aws_secret_access_key=config.r2_secret_access_key,
            region_name="auto",
        ) as client:
            yield client

    async def upload(
        self,
        data: bytes,
        key: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload data to R2.

        :param data: File content as bytes
        :param key: Object key (path in bucket)
        :param content_type: Optional MIME type
        :return: The object key
        """
        async with self._get_client() as client:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            await client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                **extra_args
            )

        return key

    async def download(self, key: str) -> bytes:
        """
        Download data from R2.

        :param key: Object key
        :return: File content as bytes
        """
        async with self._get_client() as client:
            response = await client.get_object(
                Bucket=self.bucket,
                Key=key
            )
            async with response["Body"] as stream:
                return await stream.read()

    async def delete(self, key: str) -> bool:
        """
        Delete an object from R2.

        :param key: Object key
        :return: True if deleted
        """
        async with self._get_client() as client:
            await client.delete_object(
                Bucket=self.bucket,
                Key=key
            )
        return True

    async def exists(self, key: str) -> bool:
        """
        Check if an object exists in R2.

        :param key: Object key
        :return: True if exists
        """
        async with self._get_client() as client:
            try:
                await client.head_object(
                    Bucket=self.bucket,
                    Key=key
                )
                return True
            except client.exceptions.ClientError:
                return False

    def generate_key(self, prefix: str, filename: str) -> str:
        """
        Generate a unique object key.

        :param prefix: Folder prefix (e.g., "datasets")
        :param filename: Original filename
        :return: Unique key like "datasets/uuid.ext"
        """
        ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
        unique_id = str(uuid.uuid4())
        return f"{prefix}/{unique_id}.{ext}" if ext else f"{prefix}/{unique_id}"


# Singleton instance
_storage_service: Optional[R2StorageService] = None


def get_storage_service() -> R2StorageService:
    """Get the singleton R2StorageService instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = R2StorageService()
    return _storage_service
