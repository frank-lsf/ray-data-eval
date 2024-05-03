import argparse
import contextlib
import functools
import io
import time

import humanize
import numpy as np
import ray
from ray.data.block import DataBatch

import torch
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)


from ray_data_pipeline_helpers import postprocess, download_train_directories

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--source",
    default="local",
    help="local or S3",
)

args = parser.parse_args()

DEVICE = "cuda"
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
IMAGE_SIZE = 224
NUM_FRAMES = 16
MODEL_INPUT_SHAPE = (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
BATCH_SIZE = 32

if args.source == "local":
    print("Using local data.")
    INPUT_PATH = "/mnt/data/ray-data-eval/kinetics/Kinetics700-2020-test"
else:
    print("Using S3 data.")
    try:
        with open("kinetics-train-1-percent.txt", "r") as f:
            INPUT_PATH = eval(f.read())
    except FileNotFoundError:
        INPUT_PATH = download_train_directories(
            bucket_name="ray-data-eval-us-west-2", prefix="kinetics/k700-2020/train/", percentage=1
        )


def timeit(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"{name or func.__name__}: {time.time() - start:.2f} seconds")
            return result

        return wrapper

    if callable(name):
        return decorator(name)
    else:
        return decorator


@contextlib.contextmanager
def timer(description: str = ""):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{description} took {end - start:.2f} seconds")


def tensor_size(t: torch.Tensor) -> str:
    return humanize.naturalsize(t.element_size() * t.nelement())


def print_gpu_memory_usage():
    print(
        f"Total GPU memory: {humanize.naturalsize(torch.cuda.get_device_properties(0).total_memory)}"
    )
    print(f"Reserved GPU memory: {humanize.naturalsize(torch.cuda.memory_reserved(0))}")
    print(f"Allocated GPU memory: {humanize.naturalsize(torch.cuda.memory_allocated(0))}")


class Classifier:
    def __init__(self):
        start_time = time.time()
        self.model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID).eval().to(DEVICE)
        print(f"Time to initialize model: {time.time() - start_time}")
        self.last_batch_time = start_time

    @timeit("Inference")
    @torch.no_grad
    def __call__(self, batch: DataBatch) -> DataBatch:
        inference_start_time = time.time()
        batch = batch["video"]
        batch = batch[: (batch.shape[0] // 16) * 16]  # align to 16 frames
        batch = batch.reshape(-1, 16, 3, 224, 224)
        model_input = torch.from_numpy(batch).to(DEVICE)
        print(f"Input tensor size: {tensor_size(model_input)}, shape {model_input.shape}")
        model_output = self.model(model_input)
        logits = model_output.logits
        preds = logits.argmax(-1)
        result = [self.model.config.id2label[pred.item()] for pred in preds]
        print_gpu_memory_usage()

        inference_end_time = time.time()
        print(
            "[Completed Batch]",
            inference_end_time,
            len(batch),
            "[Batch Tput]",
            len(batch) / (inference_end_time - self.last_batch_time),
            "[Inference Tput]",
            len(batch) / (inference_end_time - inference_start_time),
        )
        self.last_batch_time = inference_end_time
        print(result)
        return {"result": result}


def produce_video_slices(row: DataBatch):
    from decord import VideoReader, DECORDError

    path = row["item"][0]

    # start_time = time.time()
    # s3 = boto3.client("s3")
    # response = s3.get_object(Bucket="ray-data-eval-us-west-2", Key=path)
    # video_bytes = response["Body"].read()
    # print(
    #     f"Time to download video: {time.time() - start_time:.2f} seconds, size {humanize.naturalsize(len(video_bytes))}"
    # )
    with open(path, "rb") as fin:
        video_bytes = fin.read()

    try:
        vr = VideoReader(
            io.BytesIO(video_bytes),
            num_threads=1,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
        )
        total_num_frames = len(vr)
        print(f"Total number of frames: {total_num_frames}")
        for iteration in range(15):
            for start in range(0, total_num_frames, NUM_FRAMES):
                if start + NUM_FRAMES > total_num_frames:
                    break
                with timer("decode 16 frames"):
                    frames = vr.get_batch(range(start, start + NUM_FRAMES)).asnumpy()
                print(
                    f"[Iteration {iteration}] Yielded frames {start}-{start + NUM_FRAMES} for video, shape {frames.shape}"
                )
                yield {"frames": frames}
    except DECORDError as e:
        print(f"Failed to process video: {e}")


def preprocess_video(row: DataBatch) -> DataBatch:
    print("preprocess video")
    time.sleep(0.5)
    frames = row["frames"]
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    ret = processor(list(frames), return_tensors="np")
    arr = ret.data["pixel_values"][0]
    return {"video": arr}


def collate_video_frames(batch: DataBatch) -> DataBatch:
    return {"video": np.concatenate(batch["video"], axis=0)}


@timeit
def main():
    INSTANCE = "g5_xlarge"
    TIMELINE_FILENAME = f"video_inference_{args.source}_{INSTANCE}_batch_{BATCH_SIZE}.json"
    OUTPUT_FILENAME = f"video_inference_{args.source}_{INSTANCE}_batch_{BATCH_SIZE}.out"

    data_context = ray.data.DataContext.get_current()
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = (
        np.prod(MODEL_INPUT_SHAPE) * np.dtype(np.float32).itemsize * 1.001
    )

    ds = ray.data.from_items(
        ["/mnt/data/ray-data-eval/kinetics/Kinetics700-2020-test/-LK7TeL2DNg_000027_000037.mp4"]
    )
    ds = ds.map_batches(produce_video_slices, batch_size=1)
    ds = ds.map_batches(
        preprocess_video,
        batch_size=NUM_FRAMES,
        # num_cpus=0.99,
    )
    ds = ds.map_batches(
        Classifier,
        batch_size=BATCH_SIZE * NUM_FRAMES,
        num_gpus=1,
        concurrency=1,
        zero_copy_batch=True,
        max_concurrency=2,
    )

    ds.take_all()
    print(ds.stats())

    ray.timeline(TIMELINE_FILENAME)

    postprocess(OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
