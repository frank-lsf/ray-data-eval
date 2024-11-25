import gc
import io
import os
import time

import boto3
import numpy as np
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

from ray_data_eval.image_generation.common import IMAGE_PROMPTS_DF, S3_BUCKET_NAME

NUM_BATCHES = 10
BATCH_SIZE = 20
RESOLUTION = 512
NUM_GPUS = 1
NUM_CPUS = 8


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

        print("init")
        print(get_memory_stats())
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

    def cleanup(self):
        if hasattr(self, "model"):
            # for param in self.model.parameters():
            #     del param
            del self.model

            with torch.cuda.device(self.gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            gc.collect()

            print("GPU memory cleaned up")
            print(get_memory_stats())

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {e}")


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
    processor = GPUImageProcessor(gpu_id)
    s3_handler = S3Handler(S3_BUCKET_NAME)

    batch = list(iterator)

    # Download images
    images = [s3_handler.download_image(row.s3_path) for row in batch]
    prompts = [row.prompt for row in batch]

    # Process images
    processed_images = processor.process_batch(images, prompts)

    # Upload results
    results = []
    for row, processed_image in zip(batch, processed_images):
        output_path = s3_handler.upload_image(processed_image, row.s3_path)
        results.append((row.s3_path, output_path))

    print(time.time())
    yield from results


def main():
    spark = (
        SparkSession.builder.appName("GPU Image Pipeline")
        .config("spark.task.cpus", "1")
        .config("spark.executor.memory", "2g")
        .config("spark.executor.instances", str(NUM_CPUS))
        .getOrCreate()
    )
    limit = NUM_BATCHES * BATCH_SIZE

    pdf = IMAGE_PROMPTS_DF[:limit].reset_index()
    pdf.columns = ["s3_path", "prompt"]
    df = spark.createDataFrame(
        pdf,
        StructType(
            [StructField("s3_path", StringType(), False), StructField("prompt", StringType(), True)]
        ),
    )
    df = df.repartition(NUM_BATCHES)
    df.show()

    result_rdd = df.rdd.mapPartitionsWithIndex(
        lambda idx, iterator: process_partition(iterator, idx % NUM_GPUS),
    )

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
    return results_df


if __name__ == "__main__":
    main()
