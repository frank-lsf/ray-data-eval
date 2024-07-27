import concurrent.futures as cf
import csv
import io
import logging
import time

import boto3
import numpy as np
from PIL import Image
from pyspark.sql import SparkSession
import torch
from torchvision.models import ResNet152_Weights
from torchvision import models
import torchvision.transforms.functional as F

BUCKET = "ray-data-eval-us-west-2"
BATCH_SIZE = 128


def get_image_file_paths(limit: int = -1) -> list[str]:
    ret = []
    with open("../manifests/imagenet-manifest.txt", "r") as fin:
        for line in fin:
            try:
                _, _, _, path = line.strip().split(maxsplit=3)
                ret.append(path)
                if len(ret) % 100_000 == 0:
                    print(len(ret))
                if limit > 0 and len(ret) >= limit:
                    break
            except ValueError as e:
                print(line.strip().split(maxsplit=3))
                raise e
    return ret


class CsvLogger:
    def __init__(self, filename: str):
        self.filename = filename
        with open(self.filename, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "time_from_start",
                    "total_rows",
                    "cumulative_throughput",
                    "batch_rows",
                    "batch_inference_time",
                    "batch_inference_throughput",
                    "batch_time",
                    "batch_throughput",
                ]
            )
            writer.writerow([0, 0, 0, 0, 0, 0, 0, 0])

    def write_csv_row(self, row):
        with open(self.filename, mode="a") as file:
            writer = csv.writer(file)
            writer.writerow(row)


class ResnetModel:
    def __init__(self):
        weights = ResNet152_Weights.IMAGENET1K_V2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet152(weights=weights).to(self.device)
        self.model.eval()
        self.categories = weights.meta["categories"]

        self.start_time = time.time()
        self.last_end_time = self.start_time
        self.total_num_rows = 0

        self.csv_logger = CsvLogger("spark_inference.csv")

    def __call__(self, batch: torch.Tensor):
        logging.warning("Inference")
        inference_start_time = time.time()
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        batch = batch.to(self.device)
        with torch.inference_mode():
            prediction = self.model(batch)
            predicted_classes = prediction.argmax(dim=1).detach().cpu()
            predicted_labels = [self.categories[i] for i in predicted_classes]

        inference_end_time = time.time()
        num_rows = len(batch)
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
        return predicted_labels


spark = (
    SparkSession.builder.appName("Image Batch Inference")
    .config("spark.executor.memory", "2g")
    .config("spark.executor.instances", "12")
    .getOrCreate()
)
sc = spark.sparkContext


def transform_image(image: Image) -> torch.Tensor:
    image = image.resize((232, 232), resample=Image.BILINEAR)
    image = image.convert("RGB")
    image = F.pil_to_tensor(image)
    image = F.center_crop(image, 224)
    image = F.convert_image_dtype(image, torch.float)
    image = F.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return image


def decode_and_transform_image(content: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(content))
    image = transform_image(image)
    return image


def download_image(file_path):
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=BUCKET, Key=file_path)
    return response["Body"].read()


def load_images(partition_iterator):
    with cf.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_image = {executor.submit(download_image, f): f for f in partition_iterator}

        for future in cf.as_completed(future_to_image):
            image_bytes = future.result()
            image = decode_and_transform_image(image_bytes)
            yield image


limit = 12800
file_paths = get_image_file_paths(limit)
print(len(file_paths))

rdd = sc.parallelize(file_paths, limit // BATCH_SIZE)
rdd = rdd.mapPartitions(load_images)

model = ResnetModel()
rdd_batch = []
for image in rdd.toLocalIterator(prefetchPartitions=True):
    rdd_batch.append(image)
    if len(rdd_batch) < BATCH_SIZE:
        continue
    model_batch = torch.stack(rdd_batch, dim=0)
    result = model(model_batch)
    rdd_batch = []

# in_flight_future = None
# with cf.ThreadPoolExecutor(max_workers=1) as executor:
#     for image in rdd.toLocalIterator(prefetchPartitions=True):
#         rdd_batch.append(image)
#         if len(rdd_batch) < BATCH_SIZE:
#             continue
#         model_batch = torch.stack(rdd_batch, dim=0)
#         if in_flight_future:
#             in_flight_future.result()
#         in_flight_future = executor.submit(model, model_batch)
#         rdd_batch = []
