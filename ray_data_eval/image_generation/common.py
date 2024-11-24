import io
import time
from typing import Any

import boto3

BUCKET_NAME = "ray-data-eval-us-west-2"


def encode_and_upload(batch: dict[str, Any]):
    s3 = boto3.client("s3")
    output_paths = []
    for path, image in zip(batch["path"], batch["image"]):
        filename = path.split("/")[-1]
        key = f"output/{filename}"
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=buf.getvalue(),
        )
        output_paths.append(f"s3://{BUCKET_NAME}/{key}")
    time.sleep(2)
    return {
        "path": output_paths,
    }
