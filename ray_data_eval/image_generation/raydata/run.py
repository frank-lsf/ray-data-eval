import time

from diffusers import AutoPipelineForImage2Image
import numpy as np
import ray
import torch

from ray_data_pipeline_helpers import (
    ChromeTracer,
    CsvLogger,
    append_gpu_timeline,
)

NUM_BATCHES = 20
BATCH_SIZE = 10
RESOLUTION = 512
CSV_FILENAME = "log.csv"
GPU_TIMELINE_FILENAME = "gpu_timeline.json"
TIMELINE_FILENAME = "ray_timeline.json"
ACCELERATOR = "NVIDIA_A10G"


prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"


class Model:
    def __init__(self):
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
        output_batch = self.model(
            prompt=[prompt] * len(images),
            image=images,
            height=RESOLUTION,
            width=RESOLUTION,
            num_inference_steps=10,
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

        # print(ray._private.internal_api.memory_summary(stats_only=True))
        return {
            "image": output_batch.images,
        }


def postprocess(batch):
    print(batch["image"][0])
    time.sleep(1)
    return {
        "path": ["ok"] * len(batch["image"]),
    }


def main():
    ds = ray.data.read_images(
        ["./mountain.png"] * BATCH_SIZE * NUM_BATCHES,
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
        postprocess,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
    )

    tracer = ChromeTracer(GPU_TIMELINE_FILENAME, ACCELERATOR)
    ds.take_all()

    print(ds.stats())

    # Save and combine cpu, gpu timeline view
    tracer.save()
    ray.timeline(TIMELINE_FILENAME)
    append_gpu_timeline(TIMELINE_FILENAME, GPU_TIMELINE_FILENAME)
    print("Timeline log saved to: ", TIMELINE_FILENAME)


if __name__ == "__main__":
    ray.init(num_cpus=5, object_store_memory=8e9)
    main()
    ray.shutdown()
