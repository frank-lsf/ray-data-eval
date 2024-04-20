import argparse
import csv
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


from ray_data_pipeline_helpers import ChromeTracer, append_gpu_timeline, download_train_directories

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
    INPUT_PATH = "/home/ubuntu/kinetics/kinetics/k700-2020/train"
else:
    print("Using S3 data.")
    try:
        with open("kinetics-train-10-percent.txt", "r") as f:
            INPUT_PATH = eval(f.read())
            print(len(INPUT_PATH))
    except FileNotFoundError:
        INPUT_PATH = download_train_directories(
            bucket_name="ray-data-eval-us-west-2", prefix="kinetics/k700-2020/train/", percentage=10
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


last_good_row = None


def preprocess_video(row: DataBatch) -> DataBatch:
    from decord import VideoReader, DECORDError

    global last_good_row

    video_bytes = row["bytes"]
    try:
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
    except DECORDError as e:
        print(f"Failed to process video: {e}")
        return last_good_row

    frames = list(frames)
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    ret = processor(frames, return_tensors="np")
    arr = ret.data["pixel_values"]
    # time.sleep(1)

    if last_good_row is None:
        last_good_row = {"video": arr}

    return {"video": arr}


def collate_video_frames(batch: DataBatch) -> DataBatch:
    return {"video": np.concatenate(batch["video"], axis=0)}


@timeit
def main():
    classifier = Classifier()

    ACCELERATOR = "NVIDIA_A10G"
    TIMELINE_FILENAME = f"video_inference_{args.source}_{ACCELERATOR}_batch_{BATCH_SIZE}.json"
    GPU_TIMELINE_FILENAME = (
        f"video_inference_{args.source}_{ACCELERATOR}_batch_{BATCH_SIZE}_gpu.json"
    )
    CSV_FILENAME = f"video_inference_{args.source}_{ACCELERATOR}_batch_{BATCH_SIZE}.csv"
    start_time = time.time()
    rows_read = 0

    # Initialize tput csv file writer
    csv_file_obj = open(CSV_FILENAME, "w")
    writer = csv.writer(csv_file_obj)
    writer.writerow(["time_from_start", "number_of_rows_finished"])
    writer.writerow([0, 0])

    ds = ray.data.read_binary_files(
        INPUT_PATH,
        override_num_blocks=1291,
    )

    ds = ds.map(preprocess_video)

    tracer = ChromeTracer(GPU_TIMELINE_FILENAME, ACCELERATOR)
    last_batch_time = time.time()
    for batch in ds.iter_batches(batch_size=BATCH_SIZE, _collate_fn=collate_video_frames):
        print(f"Time to read batch: {time.time() - last_batch_time}")
        # with profiling.profile("Inference"):
        with tracer.profile("task:gpu_execution"):
            print(classifier(batch))

        last_batch_time = time.time()

        # Write time_from_start, number_of_rows_finished to csv file
        rows_read += len(batch["video"])
        writer.writerow([last_batch_time - start_time, rows_read])

    print(ds.stats())

    # Save and combine cpu, gpu timeline view
    tracer.save()
    ray.timeline(TIMELINE_FILENAME)
    append_gpu_timeline(TIMELINE_FILENAME, GPU_TIMELINE_FILENAME)


if __name__ == "__main__":
    main()
