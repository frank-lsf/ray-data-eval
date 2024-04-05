import io
import time

import torch
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)
import ray

MODEL_ID = "MCG-NJU/videomae-huge-finetuned-kinetics"

NUM_FRAMES = 16


def iterate(dataset, label, output_file=None):
    PRINT_EVERY = 100
    start = time.time()
    it = iter(dataset)
    num_rows = 0
    print_at = PRINT_EVERY
    last_print_time = start
    for batch in it:
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = batch[0]
        else:
            batch = batch["image"]
        if hasattr(batch, "shape"):
            num_rows += batch.shape[0]
        else:
            num_rows += len(batch)
        if num_rows >= print_at:
            print(f"Read {num_rows} rows in {time.time() - last_print_time} seconds.")
            print_at = ((num_rows // PRINT_EVERY) + 1) * PRINT_EVERY
            last_print_time = time.time()
    end = time.time()
    print(label, end - start, "epoch")

    tput = num_rows / (end - start)
    print(label, "tput", tput, "epoch")

    if output_file is None:
        output_file = "output.csv"
    with open(output_file, "a+") as f:
        f.write(f"{label},{tput}\n")


def video_preprocessing(row):
    from decord import VideoReader

    video_bytes = row["bytes"]
    vr = VideoReader(io.BytesIO(video_bytes))
    frames = vr.get_batch(range(min(NUM_FRAMES, len(vr)))).asnumpy()
    frames = list(frames)
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    ret = processor(frames, return_tensors="pt")
    arr = ret.data["pixel_values"]
    return {"image": arr}


# ray.init(num_cpus=1)
ds = ray.data.read_binary_files(
    "/mnt/data/ray-data-eval/kinetics/Kinetics700-2020-test",
    override_num_blocks=1291,
).map(video_preprocessing)
iterate(ds.iter_batches(batch_size=32), "ray_data")
print(ds.stats())
