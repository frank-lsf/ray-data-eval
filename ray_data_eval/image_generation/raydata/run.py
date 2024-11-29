import argparse
import time
import os

from diffusers import AutoPipelineForImage2Image
import numpy as np
from PIL import Image
import ray
import torch

from ray_data_eval.image_generation.raydata import timeline_utils
from ray_data_pipeline_helpers import (
    ChromeTracer,
)
from ray_data_eval.image_generation.common import (
    CsvLogger,
    encode_and_upload,
    get_image_paths,
    IMAGE_PROMPTS_DF,
)

NUM_BATCHES = 10
BATCH_SIZE = 20
RESOLUTION = 512
CSV_FILENAME = "log.csv"
GPU_TIMELINE_FILENAME = "gpu_timeline.json"
TIMELINE_FILENAME = "ray_timeline.json"
ACCELERATOR = "NVIDIA_A10G"
NUM_CPUS = 12
NUM_GPUS = 1


def get_image_prompt(path: str) -> str:
    if path not in IMAGE_PROMPTS_DF.index:
        return "make it black and white"
    return IMAGE_PROMPTS_DF.loc[path, "prompt"]


def transform_image(image: Image) -> Image:
    image = image.resize((RESOLUTION, RESOLUTION), resample=Image.BILINEAR)
    image = image.convert("RGB")
    time.sleep(4)
    return image


class Model:
    def __init__(self, postprocess: bool = False):
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
        self.postprocess = postprocess

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
                num_inference_steps=2,
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

        ret = {
            "image": output_batch.images,
            "path": batch["path"],
        }

        if self.postprocess:
            return encode_and_upload(ret)

        # print(ray._private.internal_api.memory_summary(stats_only=True))
        return ret


def ray_data_pipeline(image_paths: list[str]):
    ds = ray.data.read_images(
        image_paths,
        include_paths=True,
        transform=transform_image,
        override_num_blocks=NUM_BATCHES,
    )  # 12s per batch
    ds = ds.map_batches(
        Model,
        concurrency=NUM_GPUS,
        num_gpus=1,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
    )  # 5s per batch
    ds = ds.map_batches(
        encode_and_upload,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
    )  # 10s per batch
    return ds


def fused_pipeline(image_paths: list[str]):
    ds = ray.data.read_images(
        image_paths,
        include_paths=True,
        transform=transform_image,
        override_num_blocks=NUM_BATCHES,
        concurrency=NUM_GPUS,
    )  # 12s per batch
    ds = ds.map_batches(
        Model,
        concurrency=NUM_GPUS,
        num_gpus=1,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
        fn_constructor_kwargs={"postprocess": True},
    )  # 5s per batch
    ds = ds.map_batches(
        lambda batch: {"path": batch["path"]},
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
        concurrency=NUM_GPUS,
    )  # 10s per batch
    return ds


def concurrency_1_pipeline(image_paths: list[str]):
    ds = ray.data.read_images(
        image_paths,
        include_paths=True,
        transform=transform_image,
        override_num_blocks=NUM_BATCHES,
        concurrency=NUM_GPUS,
    )  # 12s per batch
    ds = ds.map_batches(
        Model,
        concurrency=NUM_GPUS,
        num_gpus=1,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
    )  # 5s per batch
    ds = ds.map_batches(
        encode_and_upload,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
        concurrency=NUM_GPUS,
    )  # 10s per batch
    return ds


def staged_pipeline(image_paths: list[str]):
    ds = ray.data.read_images(
        image_paths,
        include_paths=True,
        transform=transform_image,
        override_num_blocks=NUM_BATCHES,
        concurrency=NUM_GPUS,
    )  # 12s per batch
    ds = ds.materialize()
    ds = ds.map_batches(
        Model,
        concurrency=NUM_GPUS,
        num_gpus=1,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
    )  # 5s per batch
    ds = ds.materialize()
    ds = ds.map_batches(
        encode_and_upload,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
        concurrency=NUM_GPUS,
    )  # 10s per batch
    return ds


def static_pipeline(image_paths: list[str]):
    ds = ray.data.read_images(
        image_paths,
        include_paths=True,
        transform=transform_image,
        override_num_blocks=NUM_BATCHES,
        concurrency=4,
    )  # 12s per batch
    ds = ds.map_batches(
        Model,
        concurrency=NUM_GPUS,
        num_gpus=1,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
    )  # 5s per batch
    ds = ds.map_batches(
        encode_and_upload,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
        concurrency=4,
    )  # 10s per batch
    return ds


def main(args):
    image_paths = get_image_paths(limit=NUM_BATCHES * BATCH_SIZE)
    if args.mode == "raydata":
        ds = ray_data_pipeline(image_paths)
    elif args.mode == "fused":
        ds = fused_pipeline(image_paths)
    elif args.mode == "staged":
        ds = staged_pipeline(image_paths)
    elif args.mode == "static":
        ds = static_pipeline(image_paths)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    ds.take_all()
    print(ds.stats())

    timeline_utils.save_timeline_with_cpus_gpus("ray_combined.json", NUM_CPUS, NUM_GPUS)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "raydata",
            "fused",
            "staged",
            "static",
        ],
        default="raydata",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ray.init(num_cpus=NUM_CPUS, object_store_memory=8e9)

    data_context = ray.data.DataContext.get_current()
    data_context.op_resource_reservation_ratio = 0
    data_context.execution_options.verbose_progress = True
    data_context.is_budget_policy = True
    main(args)
