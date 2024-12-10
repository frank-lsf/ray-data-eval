import gc
import io
import logging
import os

import boto3
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

from ray_data_eval.image_generation.common import IMAGE_PROMPTS_DF, S3_BUCKET_NAME

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)

NUM_BATCHES = 10
BATCH_SIZE = 20
RESOLUTION = 512
NUM_GPUS = 1


def gpu_memory_stats() -> dict[str, float]:
    """Monitor GPU memory usage"""
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
    }


def download_partition(iterator):
    """Stage 1: Download and preprocess images"""
    s3_client = boto3.client("s3")

    for row in iterator:
        # Download
        buffer = io.BytesIO()
        s3_path = "instructpix2pix/" + row.s3_path
        s3_client.download_fileobj(S3_BUCKET_NAME, s3_path, buffer)
        buffer.seek(0)

        # Preprocess
        image = Image.open(buffer)
        image = image.resize((RESOLUTION, RESOLUTION), resample=Image.BILINEAR)
        image = image.convert("RGB")

        # Serialize for storage
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        yield (row.s3_path, row.prompt, image_bytes)


def gpu_inference_partition(iterator):
    """Stage 2: GPU processing"""
    gpu_id = torch.cuda.current_device()
    device = f"cuda:{gpu_id}"

    # Initialize model
    model = AutoPipelineForImage2Image.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)

    batch = []
    batch_images = []
    batch_prompts = []

    try:
        for row in iterator:
            s3_path, prompt, image_bytes = row
            image = Image.open(io.BytesIO(image_bytes))

            batch.append(row)
            batch_images.append(image)
            batch_prompts.append(prompt)

            if len(batch) == BATCH_SIZE:
                # Process batch
                output_batch = model(
                    prompt=batch_prompts,
                    image=batch_images,
                    height=RESOLUTION,
                    width=RESOLUTION,
                    num_inference_steps=2,
                    output_type="np",
                )

                # Yield results
                for orig_row, processed_img in zip(batch, output_batch.images):
                    # Serialize processed image
                    buffer = io.BytesIO()
                    Image.fromarray(processed_img.astype("uint8")).save(buffer, format="PNG")
                    yield (orig_row[0], buffer.getvalue())

                # Clear batch
                batch = []
                batch_images = []
                batch_prompts = []

        # Process remaining items
        if batch:
            output_batch = model(
                prompt=batch_prompts,
                image=batch_images,
                height=RESOLUTION,
                width=RESOLUTION,
                num_inference_steps=2,
                output_type="np",
            )

            for orig_row, processed_img in zip(batch, output_batch.images):
                buffer = io.BytesIO()
                Image.fromarray(processed_img.astype("uint8")).save(buffer, format="PNG")
                yield (orig_row[0], buffer.getvalue())

    finally:
        # Cleanup
        del model
        with torch.cuda.device(gpu_id):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        gc.collect()


def upload_partition(iterator):
    """Stage 3: Upload results"""
    s3_client = boto3.client("s3")

    for row in iterator:
        s3_path, processed_image = row
        output_path = f"output/{os.path.basename(s3_path)}"

        # Upload directly from memory
        s3_client.upload_fileobj(io.BytesIO(processed_image), S3_BUCKET_NAME, output_path)

        yield (s3_path, output_path)


def main():
    spark = (
        SparkSession.builder.appName("GPU Image Pipeline")
        .config("spark.executor.instances", str(NUM_GPUS))
        .config("spark.executor.cores", "1")
        .config("spark.executor.memory", "16g")
        .config("spark.task.cpus", "1")
        .config("spark.locality.wait", "0")
        .config("spark.executor.logs.python.enabled", "true")
        .getOrCreate()
    )

    # Create initial dataframe
    limit = NUM_BATCHES * BATCH_SIZE
    pdf = IMAGE_PROMPTS_DF[:limit].reset_index()
    pdf.columns = ["s3_path", "prompt"]

    df = spark.createDataFrame(
        pdf,
        StructType(
            [StructField("s3_path", StringType(), False), StructField("prompt", StringType(), True)]
        ),
    )

    rdd = (
        df.rdd.repartition(NUM_BATCHES)
        .mapPartitions(download_partition)
        .repartition(1)
        .mapPartitions(gpu_inference_partition)
        .repartition(NUM_BATCHES)
        .mapPartitions(upload_partition)
    )

    results_df = spark.createDataFrame(
        rdd,
        schema=StructType(
            [
                StructField("input_path", StringType(), False),
                StructField("output_path", StringType(), False),
            ]
        ),
    )

    # Force execution and show results
    results_df.show()
    print("Processed", results_df.count(), "images")

    return df


if __name__ == "__main__":
    main()
