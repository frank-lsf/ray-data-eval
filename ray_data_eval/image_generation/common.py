import io
import os
import time
from typing import Any

from PIL import Image
import boto3
import numpy as np
import pandas as pd

S3_BUCKET_NAME = "ray-data-eval-us-west-2"
S3_DATASOURCE = "s3://ray-data-eval-us-west-2/instructpix2pix/"
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_PROMPTS_DF = pd.read_csv(os.path.join(SCRIPT_DIR, "path2prompt.csv"), index_col=0)


def wait(n_seconds: int, busy: bool):
    if busy:
        start = time.perf_counter()
        while time.perf_counter() - start < n_seconds:
            pass
    else:
        time.sleep(n_seconds)


def get_image_paths(limit: int = 200, s3: bool = True) -> list[str]:
    ret = IMAGE_PROMPTS_DF.index[:limit].tolist()
    if s3:
        ret = [f"s3://{S3_BUCKET_NAME}/instructpix2pix/{path}" for path in ret]
    return ret


def get_image_prompt(path: str) -> str:
    if path not in IMAGE_PROMPTS_DF.index:
        return "make it black and white"
    return IMAGE_PROMPTS_DF.loc[path, "prompt"]


def transform_image(image: Image, resolution=512, busy=False) -> Image:
    image = image.resize((resolution, resolution), resample=Image.BILINEAR)
    image = image.convert("RGB")
    wait(4, busy)
    return image


def encode_and_upload(batch: dict[str, Any], busy=False):
    s3 = boto3.client("s3")
    output_paths = []
    for path, image in zip(batch["path"], batch["image"]):
        filename = path.split("/")[-1]
        key = f"output/{filename}"
        buf = io.BytesIO()
        image = numpy_to_pil(image)[0]
        image.save(buf, format="PNG")
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=buf.getvalue(),
        )
        output_paths.append(f"s3://{S3_BUCKET_NAME}/{key}")
    wait(8, busy)
    return {
        "path": output_paths,
    }


def numpy_to_pil(images: np.ndarray) -> list[Image.Image]:
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
