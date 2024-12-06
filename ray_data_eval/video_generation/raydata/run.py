import time
from typing import Any, Iterator
import io

import cv2
import mmcv
from mmagic.apis.mmagic_inferencer import MMagicInferencer
import numpy as np
import ray
import torch


INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

DECORD_NUM_FRAMES_PER_BATCH = 8  # Adjust to avoid CUDA OOM

BATCH_SIZE_LIMIT = 1024 * 1024 * 32  # Adjust to avoid CUDA OOM
RGB_MAX = 255.0


def _batch_to_tensor(batch: list[np.ndarray]) -> torch.Tensor:
    batch_np = np.stack(batch, axis=0)  # (frames, width, height, channels)
    tensor = (
        torch.from_numpy(batch_np).permute(0, 3, 1, 2) / RGB_MAX
    )  # (frames, channels, width, height)
    return tensor


def preprocess(input_path: str) -> Iterator[torch.Tensor]:
    reader = mmcv.VideoReader(input_path)
    batch = []
    batch_total_size = 0
    last_batch_time = time.time()
    for frame in reader:
        batch.append(frame)
        batch_total_size += frame.nbytes
        if batch_total_size > BATCH_SIZE_LIMIT:
            yield _batch_to_tensor(batch)
            print(f"Time to decode {len(batch)} frames: {time.time() - last_batch_time:.3f}s")
            batch = []
            batch_total_size = 0
            last_batch_time = time.time()
    if batch:
        yield _batch_to_tensor(batch)
        print(f"Time to decode {len(batch)} frames: {time.time() - last_batch_time:.3f}s")


class VideoWriter:
    def __init__(self, width: int, height: int, output_path: str):
        self.width = width
        self.height = height
        self.output_path = output_path

        # Define the codec using VideoWriter_fourcc
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Create VideoWriter object
        self.writer = cv2.VideoWriter(output_path, fourcc, 30, (self.width, self.height))

        self.num_frames_written = 0

    def write_batch(self, preds: np.ndarray):
        start_time = time.time()
        # Convert to uint8 and correct format
        preds = np.clip(np.round(preds * RGB_MAX), 0, RGB_MAX).astype(np.uint8)
        preds = np.transpose(preds, (0, 2, 3, 1))  # (frames, width, height, channels)

        # OpenCV expects BGR format
        preds = preds[..., ::-1]  # Convert RGB to BGR

        # Write each frame
        for img in preds:
            self.writer.write(img)
            self.num_frames_written += 1

        print(
            f"Written {self.num_frames_written} frames, last batch time {time.time() - start_time:.3f}s"
        )

    def finish(self):
        self.writer.release()


class VideoProcessor:
    def __init__(self):
        self.engine = MMagicInferencer("real_basicvsr", seed=None).inferencer.inferencer
        self.upscaling_factor = 2

    def __call__(self, batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
        start_time = time.time()
        frames_processed = 0
        tensor = torch.from_numpy(batch["bytes"]).float()
        # tensor = tensor.to("cuda", non_blocking=True)

        batch_start_time = time.time()
        # with torch.autocast("cuda"):
        preds = self.engine.forward(tensor)  # (batch, frames, channels, width, height)
        frames_processed += preds.size(1)
        print(
            f"Processed {frames_processed} frames, "
            f"last batch time {time.time() - batch_start_time:.3f}s, "
            f"total time {time.time() - start_time:.3f}s"
        )
        preds_np = preds.detach().cpu().numpy()
        yield {"bytes": preds_np, "path": batch["path"]}


def preprocess_operator(batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
    # print(batch.keys(), flush=True)
    # paths = batch["path"]
    # for path in paths:
    #     for tensor in preprocess(path):
    #         yield {"bytes": tensor.unsqueeze(0), "path": [path]}

    path = batch["path"]
    # print(type(path))
    video_bytes = batch["bytes"][0]

    from decord import VideoReader, DECORDError

    reader = VideoReader(
        io.BytesIO(video_bytes),
        num_threads=1,
        # width=320,
        # height=240,
    )
    batch = []
    batch_total_size = 0
    last_batch_time = time.time()

    start, end = 0, DECORD_NUM_FRAMES_PER_BATCH
    while start < 128:
        for frame in reader.get_batch(range(start, end)).asnumpy():
            frame = np.array(frame)
            batch.append(frame)
            batch_total_size += frame.nbytes
            if batch_total_size > BATCH_SIZE_LIMIT:
                yield {"bytes": _batch_to_tensor(batch).unsqueeze(0), "path": path}
                print(f"Time to decode {len(batch)} frames: {time.time() - last_batch_time:.3f}s")
                batch = []
                batch_total_size = 0
                last_batch_time = time.time()
        start = end
        end = min(end + DECORD_NUM_FRAMES_PER_BATCH, len(reader))
    if batch:
        yield {"bytes": _batch_to_tensor(batch).unsqueeze(0), "path": path}
        print(f"Time to decode {len(batch)} frames: {time.time() - last_batch_time:.3f}s")


def postprocess_operator(batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
    # print(batch.keys())
    # print(type(batch["path"]))
    # print(batch["path"][0])
    input_path = batch["path"][0]
    # output_path = input_path.replace("input", "output")
    output_path = input_path.replace(
        "ray-data-eval-us-west-2/youtube-8m-sample-sampled/", "data/output/"
    )
    output_shape = batch["bytes"].shape
    width = output_shape[-1]
    height = output_shape[-2]
    writer = VideoWriter(width, height, output_path)
    writer.write_batch(batch["bytes"].squeeze(0))
    writer.finish()
    yield {"path": [output_path]}


def print_path(batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
    # print(batch["path"])
    yield {"path": batch["path"]}


# ds = ray.data.from_items(["data/input/a.mp4"] * 10)
# ds = ray.data.from_items(["data/input/a.mp4"])

ds = ray.data.read_binary_files(
    "s3://ray-data-eval-us-west-2/youtube-8m-sample-sampled/", include_paths=True
)

# ds_high_res = ray.data.read_binary_files(
#     "s3://ray-data-eval-us-west-2/youtube-8m-sample-sampled/high-res/", include_paths=True
# )
# ds_low_res = ray.data.read_binary_files(
#     "s3://ray-data-eval-us-west-2/youtube-8m-sample-sampled/low-res/", include_paths=True
# )

# ds = ds_high_res.union(ds_low_res)

# ds = ds.map_batches(print_path, batch_size=1, zero_copy_batch=True)

ds = ds.map_batches(
    preprocess_operator,
    batch_size=1,
    zero_copy_batch=True,
)
ds = ds.map_batches(
    VideoProcessor,
    concurrency=1,
    num_gpus=1,
    batch_size=1,
    zero_copy_batch=True,
)
ds = ds.map_batches(
    postprocess_operator,
    batch_size=1,
    zero_copy_batch=True,
)
print(ds.take_all())

ray.timeline("timeline.json")
