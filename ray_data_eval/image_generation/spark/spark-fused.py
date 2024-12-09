import io
import logging
import os
import time

import boto3
import numpy as np
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

from ray_data_eval.image_generation.common import (
    IMAGE_PROMPTS_DF,
    S3_BUCKET_NAME,
    CsvTimerLogger,
    wait,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)

NUM_BATCHES = 50
BATCH_SIZE = 20
RESOLUTION = 512
NUM_GPUS = 1
NUM_CPUS = 8

CSV_FILENAME = "spark_fused_tput.csv"


def get_memory_stats() -> dict[str, float]:
    """Monitor GPU memory usage"""
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
    }


class GPUImageProcessor:
    def __init__(self, gpu_id: int):
        """Initialize the model on a specific GPU"""
        self.device = f"cuda:{gpu_id}"
        self.gpu_id = gpu_id
        torch.cuda.set_device(self.gpu_id)

        self.model = AutoPipelineForImage2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(self.device)

    def process_batch(self, images: list[Image.Image], prompts: list[str]) -> np.ndarray:
        """Process a batch of images"""
        print("GPU processing batch")
        output_batch = self.model(
            prompt=prompts,
            image=images,
            height=RESOLUTION,
            width=RESOLUTION,
            num_inference_steps=2,
            output_type="np",
        )
        return output_batch.images


class S3Handler:
    def __init__(self, bucket_name: str):
        self.s3_client = boto3.client("s3")
        self.bucket = bucket_name

    def download_image(self, s3_path: str) -> Image.Image:
        """Download and preprocess image from S3 directly to memory"""
        buffer = io.BytesIO()
        s3_path = "instructpix2pix/" + s3_path
        self.s3_client.download_fileobj(self.bucket, s3_path, buffer)
        buffer.seek(0)

        image = Image.open(buffer)
        image = image.resize((RESOLUTION, RESOLUTION), resample=Image.BILINEAR)
        image = image.convert("RGB")

        buffer.close()

        return image

    def upload_image(self, image_array: np.ndarray, s3_path: str):
        """Upload processed image to S3 directly from memory"""
        image = Image.fromarray(image_array.astype("uint8"))

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        output_path = f"output/{os.path.basename(s3_path)}"
        self.s3_client.upload_fileobj(buffer, self.bucket, output_path)

        buffer.close()

        return output_path


def process_partition(iterator, gpu_id: int):
    s3_handler = S3Handler(S3_BUCKET_NAME)

    batch = list(iterator)

    # Download images
    images = [s3_handler.download_image(row.s3_path) for row in batch]
    prompts = [row.prompt for row in batch]
    wait(4 * len(batch), busy=True)

    # Process images

    processor = GPUImageProcessor(gpu_id)
    processed_images = processor.process_batch(images, prompts)
    del processor

    # Upload results
    results = []
    for row, processed_image in zip(batch, processed_images):
        output_path = s3_handler.upload_image(processed_image, row.s3_path)
        results.append((row.s3_path, output_path))
    wait(8, busy=True)

    print("Batch done", time.time())
    yield from results


def mapper(iterator):
    csv_logger = CsvTimerLogger(CSV_FILENAME)
    gpu_id = torch.cuda.current_device()
    batch = []

    def run_batch(batch):
        inference_start_time = time.time()
        yield from process_partition(batch, gpu_id)
        inference_end_time = time.time()
        batch_size = len(batch)
        batch = []
        csv_logger.log_batch(
            batch_size,
            inference_end_time - inference_start_time,
        )

    for row in iterator:
        batch.append(row)
        if len(batch) == BATCH_SIZE:
            yield from run_batch(batch)
            batch = []

    if len(batch) > 0:
        yield from run_batch(batch)


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
    limit = NUM_BATCHES * BATCH_SIZE

    start_time = time.time()

    pdf = IMAGE_PROMPTS_DF[:limit].reset_index()
    pdf.columns = ["s3_path", "prompt"]
    df = spark.createDataFrame(
        pdf,
        StructType(
            [StructField("s3_path", StringType(), False), StructField("prompt", StringType(), True)]
        ),
    )
    df = df.repartition(1)
    df.show()

    result_rdd = df.rdd.mapPartitions(mapper)

    # Convert results back to dataframe
    results_df = spark.createDataFrame(
        result_rdd,
        StructType(
            [
                StructField("input_path", StringType(), False),
                StructField("output_path", StringType(), False),
            ]
        ),
    )

    results_df.show()
    print("Processed", results_df.count(), "images")

    end_time = time.time()
    print("Total time:", end_time - start_time)
    return results_df


if __name__ == "__main__":
    main()
