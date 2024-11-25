import time
import os

from diffusers import AutoPipelineForImage2Image
import numpy as np
from PIL import Image
import ray
import torch

from ray_data_pipeline_helpers import (
    ChromeTracer,
    CsvLogger,
    append_gpu_timeline,
)
from ray_data_eval.image_generation.common import (
    encode_and_upload,
    get_image_paths,
    IMAGE_PROMPTS_DF,
    S3_DATASOURCE,
)

NUM_BATCHES = 10
BATCH_SIZE = 20
RESOLUTION = 512
CSV_FILENAME = "log.csv"
GPU_TIMELINE_FILENAME = "gpu_timeline.json"
TIMELINE_FILENAME = "ray_timeline.json"
ACCELERATOR = "NVIDIA_A10G"


def get_image_prompt(path: str) -> str:
    if path not in IMAGE_PROMPTS_DF.index:
        return "make it black and white"
    return IMAGE_PROMPTS_DF.loc[path, "prompt"]


def transform_image(image: Image) -> Image:
    image = image.resize((RESOLUTION, RESOLUTION), resample=Image.BILINEAR)
    image = image.convert("RGB")
    time.sleep(1)
    return image


class Model:
    def __init__(self):
        self.tracer = ChromeTracer(GPU_TIMELINE_FILENAME)
        self.model = AutoPipelineForImage2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")  # StableDiffusionImg2ImgPipeline

        self.start_time = time.time()
        self.last_end_time = self.start_time
        self.total_num_rows = 0

        self.csv_logger = CsvLogger(CSV_FILENAME)

    def __call__(self, batch: dict[str, np.ndarray]):
        inference_start_time = time.time()

        images = batch["image"]
        keys = [os.path.basename(path) for path in batch["path"]]
        prompts = [get_image_prompt(key) for key in keys]

        with self.tracer.profile("task:gpu_execution"):
            output_batch = self.model(
                prompt=prompts,
                image=images,
                height=RESOLUTION,
                width=RESOLUTION,
                num_inference_steps=5,
                output_type="np",
            )

        inference_end_time = time.time()
        num_rows = len(images)
        self.total_num_rows += num_rows
        self.csv_logger.write_csv_row(
            [
                inference_end_time - self.start_time,
                self.total_num_rows,
                self.total_num_rows / (inference_end_time - self.start_time),
                num_rows,
                inference_end_time - inference_start_time,
                num_rows / (inference_end_time - inference_start_time),
                inference_end_time - self.last_end_time,
                num_rows / (inference_end_time - self.last_end_time),
            ]
        )
        self.last_end_time = inference_end_time
        self.tracer.save()

        # print(ray._private.internal_api.memory_summary(stats_only=True))
        return {
            "image": output_batch.images,
            "path": batch["path"],
        }


def main():
    image_paths = get_image_paths(limit=NUM_BATCHES * BATCH_SIZE)
    ds = ray.data.read_images(
        image_paths,
        include_paths=True,
        transform=transform_image,
        override_num_blocks=NUM_BATCHES,
    )
    ds = ds.map_batches(
        Model,
        concurrency=1,
        num_gpus=1,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
        max_concurrency=2,
    )
    ds = ds.map_batches(
        encode_and_upload,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
    )
    ds.take_all()
    print(ds.stats())

    print(f"Ray timeline saved to {TIMELINE_FILENAME}")
    ray.timeline(TIMELINE_FILENAME)
    append_gpu_timeline(TIMELINE_FILENAME, GPU_TIMELINE_FILENAME)
    print("Timeline log saved to: ", TIMELINE_FILENAME)


if __name__ == "__main__":
    ray.init(object_store_memory=8e9)
    main()
    ray.shutdown()
