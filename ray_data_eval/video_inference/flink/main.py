from typing import List
import time

import boto3
import io
import numpy as np

from pyflink.common import Configuration
from pyflink.common.typeinfo import Types
from pyflink.datastream import (
    KeyedProcessFunction,
    MapFunction,
    RuntimeContext,
    StreamExecutionEnvironment,
)

import torch
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

LOAD_PARALLELISM = 4
PREPROCESS_PARALLELISM = 4
INFERENCE_PARALLELISM = 1

USE_LOCAL = False
VIDEO_FILE_PATH = "/home/ubuntu/dev/kinetics-manifest.txt"  # This file should contain all S3/local filepaths of the video files
BUCKET = "ray-data-eval-us-west-2"

DEVICE = "cuda"
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
IMAGE_SIZE = 224
NUM_FRAMES = 16
MODEL_INPUT_SHAPE = (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
BATCH_SIZE = 32

PREPROCESSOR_CACHE = "/tmp/videomae_processor"
MODEL_CACHE = "/tmp/videomae_model"


def get_config() -> Configuration:
    """Get Flink configuration. Note that here we set python execution mode to thread. The process mode doesn't work"""
    config = Configuration()
    config.set_string("python.execution-mode", "thread")
    return config


def initialize_models():
    """Pre-load models and save to shared location"""
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(PREPROCESSOR_CACHE)

    model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
    model.save_pretrained(MODEL_CACHE)


def get_video_file_paths(limit: int = -1) -> List[str]:
    """Get video file paths from the file"""
    ret = []
    with open(VIDEO_FILE_PATH, "r") as fin:
        for line in fin:
            try:
                ret.append(line.strip().replace(f"s3://{BUCKET}/", ""))
                if limit > 0 and len(ret) >= limit:
                    break
            except ValueError as e:
                print(line.strip())
                raise e
    return ret


class Loader(MapFunction):
    def map(self, file_path: str) -> torch.Tensor:
        if USE_LOCAL:
            with open(file_path, "rb") as f:
                video_bytes = f.read()
        else:
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=BUCKET, Key=file_path)
            video_bytes = response["Body"].read()

        print(len(video_bytes) / 1024 / 1024, "MB")
        return video_bytes


class VideoPreprocessor(MapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.processor = VideoMAEImageProcessor.from_pretrained(
            PREPROCESSOR_CACHE, local_files_only=True
        )

    def map(self, row):
        from decord import VideoReader, DECORDError

        video_bytes = row
        try:
            vr = VideoReader(
                io.BytesIO(video_bytes),
                num_threads=1,
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
            )
            frames = vr.get_batch(range(min(NUM_FRAMES, len(vr)))).asnumpy()

            # Pad frames if needed
            if frames.shape[0] < NUM_FRAMES:
                last_frame = frames[-1:]
                last_frame_repeated = np.repeat(last_frame, NUM_FRAMES - len(frames), axis=0)
                frames = np.concatenate([frames, last_frame_repeated], axis=0)
        except DECORDError as e:
            print(f"Failed to process video: {e}")
            return np.zeros((NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        # Process frames
        frames = list(frames)
        ret = self.processor(frames, return_tensors="np")
        arr = ret.data["pixel_values"]
        print("done")
        return arr[0]


class VideoMAEModel:
    """VideoMAE model wrapper with performance logging"""

    def __init__(self):
        self.device = torch.device(DEVICE)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            MODEL_CACHE, local_files_only=True
        )
        self.model = self.model.eval().to(self.device)

        self.start_time = time.time()
        self.last_batch_completion_time = self.start_time
        self.total_num_videos = 0

    @torch.no_grad()
    def __call__(self, batch: np.ndarray) -> List[dict]:
        """Run inference on a batch of videos"""
        inference_start_time = time.time()

        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        batch = batch.to(self.device)

        with torch.inference_mode():
            outputs = self.model(batch)
            logits = outputs.logits

            predicted_classes = logits.argmax(dim=-1).detach().cpu()
            confidences = torch.softmax(logits, dim=-1).max(dim=-1).values.detach().cpu()

            predicted_labels = []
            for pred_class, conf in zip(predicted_classes, confidences):
                predicted_labels.append(
                    {
                        "label": self.model.config.id2label[pred_class.item()],
                        "confidence": conf.item(),
                    }
                )

        inference_end_time = time.time()
        self.total_num_videos += len(batch)
        self.last_batch_completion_time = inference_end_time

        return predicted_labels


class VideoModelProcessor(KeyedProcessFunction):
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.current_batch = []
        self.model = None

    def open(self, runtime_context: RuntimeContext):
        print(f"Initializing VideoModelProcessor on {DEVICE}")

        self.model = VideoMAEModel()
        self.current_batch = []

    def process_element(self, value, ctx: "KeyedProcessFunction.Context"):
        video_tensor = value
        self.current_batch.append(video_tensor)

        if len(self.current_batch) < self.batch_size:
            return

        print(f"Processing batch of size {len(self.current_batch)}")

        batch = np.stack(self.current_batch, axis=0)
        predictions = self.model(batch)

        labels = []
        for pred_info in predictions:
            labels.append(pred_info["label"])

        self.current_batch = []
        return labels


def run_flink(env: StreamExecutionEnvironment):
    file_paths = get_video_file_paths(2560)
    initialize_models()

    ds = env.from_collection(file_paths, type_info=Types.STRING())

    loader = Loader()
    ds = ds.map(loader, output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(LOAD_PARALLELISM)

    video_preprocessor = VideoPreprocessor()
    ds = ds.map(video_preprocessor, output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(
        PREPROCESS_PARALLELISM
    )

    ds = (
        ds.key_by(lambda x: 0).process(VideoModelProcessor()).set_parallelism(INFERENCE_PARALLELISM)
    )
    ds.print()
    env.execute()


def main():
    config = Configuration()
    config.set_string("python.execution-mode", "thread")
    env = StreamExecutionEnvironment.get_execution_environment(config)

    run_flink(env)


if __name__ == "__main__":
    main()
