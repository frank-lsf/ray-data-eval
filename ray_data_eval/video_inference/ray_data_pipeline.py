import functools
import io
import time

import humanize
import numpy as np
import ray
from ray.data.block import DataBatch
from ray._private import profiling
import torch
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)

DEVICE = "cuda"
MODEL_ID = "MCG-NJU/videomae-huge-finetuned-kinetics"
IMAGE_SIZE = 224
NUM_FRAMES = 16
MODEL_INPUT_SHAPE = (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
BATCH_SIZE = 64


DATA_PATH = "/data/ray-data-eval/kinetics"
INPUT_PATH = f"{DATA_PATH}/Kinetics700-2020-test"


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

    @timeit("Inference")
    @torch.no_grad
    def __call__(self, batch: DataBatch) -> DataBatch:
        model_input = torch.from_numpy(batch["video"]).to(DEVICE)
        print(f"Input tensor size: {tensor_size(model_input)}")
        model_output = self.model(model_input)
        logits = model_output.logits
        preds = logits.argmax(-1)
        result = [self.model.config.id2label[pred.item()] for pred in preds]
        print_gpu_memory_usage()
        return {"result": result}


def preprocess_video(row: DataBatch) -> DataBatch:
    from decord import VideoReader

    video_bytes = row["bytes"]
    vr = VideoReader(
        io.BytesIO(video_bytes),
        num_threads=1,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
    )
    frames = vr.get_batch(range(min(NUM_FRAMES, len(vr)))).asnumpy()
    if frames.shape[0] < NUM_FRAMES:
        last_frame = frames[-2:-1]
        last_frame_repeated = np.repeat(last_frame, NUM_FRAMES - len(frames), axis=0)
        frames = np.concatenate([frames, last_frame_repeated], axis=0)

    frames = list(frames)
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    ret = processor(frames, return_tensors="np")
    arr = ret.data["pixel_values"]
    # time.sleep(1)
    return {"video": arr}


def collate_video_frames(batch: DataBatch) -> DataBatch:
    return {"video": np.concatenate(batch["video"], axis=0)}


@timeit
def main():
    classifier = Classifier()
    ds = ray.data.read_binary_files(
        INPUT_PATH,
        override_num_blocks=1291,
    )
    ds = ds.map(preprocess_video)

    last_batch_time = time.time()
    for batch in ds.iter_batches(batch_size=BATCH_SIZE, _collate_fn=collate_video_frames):
        print(f"Time to read batch: {time.time() - last_batch_time}")
        with profiling.profile("Inference"):
            print(classifier(batch))
        last_batch_time = time.time()

    print(ds.stats())
    ray.timeline("video_inference.json")


if __name__ == "__main__":
    main()
