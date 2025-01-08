import argparse
import os

import boto3
from loguru import logger


def download_from_s3(prefix, local_path, bucket_name, endpoint_url):
    """Downloads files from an S3 bucket to a local directory.

    Args:
        prefix (str): The prefix of the objects to download in the S3 bucket.
        local_path (str): The local directory to download the files to.
        bucket_name (str): The name of the S3 bucket.
        endpoint_url (str): The endpoint URL of the S3 service.
    """

    aws_access_key_id = os.environ.get("ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("SECRET_ACCESS_KEY")

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "ACCESS_KEY_ID and SECRET_ACCESS_KEY must be set as environment variables."
        )

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
    )
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    # logger.info(f"prefix is {prefix}")

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.endswith("/"):
                continue

            local_file_path = os.path.join(
                local_path, prefix, os.path.relpath(key, prefix)
            )
            # logger.info(f"local_file_path is {local_file_path}")

            if os.path.exists(local_file_path):
                logger.info(f"{local_file_path} already exists, skipping...")
                continue

            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3.download_file(bucket_name, key, local_file_path)
            logger.info(f"Downloaded {key} to {local_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from S3.")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix of the objects to download in the S3 bucket.",
    )
    parser.add_argument(
        "--local_path", required=True, help="Local directory to download the files to."
    )
    parser.add_argument("--bucket_name", required=True, help="Name of the S3 bucket.")
    parser.add_argument(
        "--endpoint_url", required=True, help="Endpoint URL of the S3 service."
    )

    args = parser.parse_args()

    download_from_s3(args.prefix, args.local_path, args.bucket_name, args.endpoint_url)
