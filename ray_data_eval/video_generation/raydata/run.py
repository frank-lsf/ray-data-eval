import time
from typing import Any, Iterator
import io

import boto3
import cv2
from decord import VideoReader

# from mmagic.apis.mmagic_inferencer import MMagicInferencer
import numpy as np
import ray
import torch
from ray_data_eval.image_generation.common import CsvTimerLogger
from ray_data_eval.video_generation.raydata.data import VIDEO_SEGMENTS

BUCKET = "ray-data-eval-us-west-2"
CSV_FILENAME = "radar_tput.csv"

DECORD_NUM_FRAMES_PER_BATCH = 8  # Adjust to avoid CUDA OOM

BATCH_SIZE_LIMIT = 1024 * 1024 * 32  # Adjust to avoid CUDA OOM
RGB_MAX = 255.0
NUM_FRAMES_PER_BATCH = 16


def sleep(mean: float, std_pct: float = 0.2):
    std = mean * std_pct
    duration = np.random.normal(mean, std)
    time.sleep(duration)


def _batch_to_tensor(batch: list[np.ndarray]) -> torch.Tensor:
    batch_np = np.stack(batch, axis=0)  # (frames, width, height, channels)
    tensor = (
        torch.from_numpy(batch_np).permute(0, 3, 1, 2) / RGB_MAX
    )  # (frames, channels, width, height)
    return tensor


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
        self.csv_logger = CsvTimerLogger(CSV_FILENAME)

    def __call__(self, batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
        start_time = time.time()
        tensor = torch.from_numpy(batch["bytes"]).float()
        # tensor = tensor.to("cuda", non_blocking=True)

        # with torch.autocast("cuda"):
        preds = self.engine.forward(tensor)  # (batch, frames, channels, width, height)
        preds_np = preds.detach().cpu().numpy()

        self.csv_logger.log_batch(NUM_FRAMES_PER_BATCH, time.time() - start_time)
        yield {"bytes": preds_np, "path": batch["path"]}


class DummyVideoProcessor:
    def __init__(self):
        self.csv_logger = CsvTimerLogger(CSV_FILENAME)

    def __call__(self, batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
        # path = batch["path"][0]
        start_time = time.time()
        sleep(0.3)
        self.csv_logger.log_batch(NUM_FRAMES_PER_BATCH, time.time() - start_time)
        yield batch


def preprocess_operator(batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
    item = batch["item"][0]
    path, start_frame = item.split("#")
    start_frame = int(start_frame)
    end_frame = start_frame + NUM_FRAMES_PER_BATCH

    s3_client = boto3.client("s3")
    video_bytes = io.BytesIO()
    s3_client.download_fileobj(BUCKET, path, video_bytes)

    start_time = time.time()
    video_bytes.seek(0)
    reader = VideoReader(
        video_bytes,
        num_threads=1,
    )
    batch = []
    for frame in reader.get_batch(range(start_frame, end_frame)).asnumpy():
        frame = np.array(frame)
        batch.append(frame)

    if "high-res" in path:
        time.sleep(3)
    yield {"bytes": _batch_to_tensor(batch).unsqueeze(0), "path": [path]}
    print(f"Time to decode {len(batch)} frames from {path}: {time.time() - start_time:.3f}s")


def postprocess_operator(batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
    input_path = batch["path"][0]
    filename = input_path.split("/")[-1]
    output_path = f"/tmp/{filename}"
    output_shape = batch["bytes"].shape
    if len(output_shape) < 5:
        yield {"path": [output_path]}
        return

    width = output_shape[-1]
    height = output_shape[-2]
    writer = VideoWriter(width, height, output_path)
    writer.write_batch(batch["bytes"].squeeze(0))
    writer.finish()

    if "low-res" in input_path:
        sleep(0.3)

    yield {"path": [output_path]}


ray.init(object_store_memory=1024 * 1024 * 1024 * 12, num_cpus=10)

ds = ray.data.from_items(VIDEO_SEGMENTS)

ds = ds.map_batches(
    preprocess_operator,
    batch_size=1,
    zero_copy_batch=True,
    concurrency=4,
)
ds = ds.map_batches(
    DummyVideoProcessor,
    concurrency=1,
    num_gpus=1,
    batch_size=1,
    zero_copy_batch=True,
)
ds = ds.map_batches(
    postprocess_operator,
    batch_size=1,
    zero_copy_batch=True,
    concurrency=4,
)
print(ds.take_all())

ray.timeline("timeline.json")
