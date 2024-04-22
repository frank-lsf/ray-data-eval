import argparse
import csv
import time

import humanize
import numpy as np
import ray
import torch
from torchvision.models import ResNet152_Weights
from torchvision import transforms
from torchvision import models
from typing import Dict

from ray_data_pipeline_helpers import (
    ChromeTracer,
    append_gpu_timeline,
    download_train_directories,
)

DATA_PERCENTAGE = 1
BUCKET_NAME = "ray-data-eval-us-west-2"
PREFIX = "imagenet/ILSVRC/Data/CLS-LOC/train/"
IMAGENET_LOCAL_DIR = f"/mnt/data/ray-data-eval/ILSVRC/Data/CLS-LOC/10k/"
IMAGENET_S3_FILELIST = f"imagenet-train-{DATA_PERCENTAGE}-percent.txt"
DEFAULT_IMAGE_SIZE = 256  # transformed size in ResNet152_Weights.IMAGENET1K_V1.transforms
BATCH_SIZE = 1024
NUM_CPUS = 4

imagenet_transforms = ResNet152_Weights.IMAGENET1K_V1.transforms
transform = transforms.Compose([transforms.ToTensor(), imagenet_transforms()])

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--source",
    default="local",
    help="local or S3",
)
parser.add_argument(
    "--iter",
    action="store_const",
    const="iter_batches",
    dest="mode",
)
parser.add_argument(
    "--map",
    action="store_const",
    const="map_batches",
    dest="mode",
)
parser.set_defaults(mode="iter_batches")
args = parser.parse_args()

if args.source == "local":
    print("Using local data.")
    INPUT_PATH = IMAGENET_LOCAL_DIR
else:
    print("Using S3 data.")
    try:
        with open(IMAGENET_S3_FILELIST, "r") as f:
            INPUT_PATH = eval(f.read())
            print(len(INPUT_PATH))
    except FileNotFoundError:
        INPUT_PATH = download_train_directories(
            bucket_name=BUCKET_NAME,
            prefix=PREFIX,
            percentage=DATA_PERCENTAGE,
        )

ACCELERATOR = "NVIDIA_A10G"
TIMELINE_FILENAME = (
    f"logs/ray_log/image_inference_{args.source}_batch_{BATCH_SIZE}_{DATA_PERCENTAGE}pct.json"
)
GPU_TIMELINE_FILENAME = (
    f"logs/gpu/image_inference_{args.source}_batch_{BATCH_SIZE}_{DATA_PERCENTAGE}pct_gpu.json"
)
CSV_FILENAME = f"logs/csv/image_inference_{args.source}_batch_{BATCH_SIZE}_{DATA_PERCENTAGE}pct.csv"


def tensor_size(t: torch.Tensor) -> str:
    return humanize.naturalsize(t.element_size() * t.nelement())


def print_gpu_memory_usage():
    print(
        f"Total GPU memory: {humanize.naturalsize(torch.cuda.get_device_properties(0).total_memory)}"
    )
    print(f"Reserved GPU memory: {humanize.naturalsize(torch.cuda.memory_reserved(0))}")
    print(f"Allocated GPU memory: {humanize.naturalsize(torch.cuda.memory_allocated(0))}")


def preprocess_image(row: Dict[str, np.ndarray]):
    return {
        "transformed_image": transform(row["image"]),
    }


class ResnetModel:
    def __init__(self):
        self.weights = ResNet152_Weights.IMAGENET1K_V1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet152(weights=self.weights).to(self.device)
        self.model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]):
        # Convert the numpy array of images into a PyTorch tensor.
        # Move the tensor batch to GPU if available.
        inference_start_time = time.time()
        torch_batch = torch.from_numpy(batch["transformed_image"]).to(self.device)
        with torch.inference_mode():
            prediction = self.model(torch_batch)
            predicted_classes = prediction.argmax(dim=1).detach().cpu()
            predicted_labels = [self.weights.meta["categories"][i] for i in predicted_classes]
            print(f"Inference time: {time.time() - inference_start_time:.4f}")
            return {
                "predicted_label": predicted_labels,
            }


def main():
    start_time = time.time()
    rows_read = 0

    # Initialize tput csv file writer
    csv_file_obj = open(CSV_FILENAME, "w")
    writer = csv.writer(csv_file_obj)
    writer.writerow(["time_from_start", "number_of_rows_finished"])
    writer.writerow([0, 0])

    ds = ray.data.read_images(
        INPUT_PATH,
        mode="RGB",
    )

    ds = ds.map(preprocess_image)

    model = None
    if args.mode == "map_batches":
        ds = ds.map_batches(
            ResnetModel,
            concurrency=1,  # Use 1 GPU (number of GPUs in your cluster)
            num_gpus=1,  # number of GPUs needed for each ImageClassifier instance
            batch_size=BATCH_SIZE,  # Use the largest batch size that can fit on our GPUs
            zero_copy_batch=True,
        )
    else:  # iter_batches
        model = ResnetModel()

    last_batch_time = time.time()
    tracer = ChromeTracer(GPU_TIMELINE_FILENAME, ACCELERATOR)
    for batch in ds.iter_batches(batch_size=BATCH_SIZE):
        print(f"Time to read batch: {time.time() - last_batch_time:.4f}")
        if args.mode == "iter_batches":
            with tracer.profile("task:gpu_execution"):
                batch = model(batch)

        print(f"Total batch time: {time.time() - last_batch_time:.4f}")
        last_batch_time = time.time()

        # Write time_from_start, number_of_rows_finished to csv file
        rows_read += len(batch)
        writer.writerow([last_batch_time - start_time, rows_read])

    print(ds.stats())
    print("Total images processed: ", rows_read)

    # Save and combine cpu, gpu timeline view
    tracer.save()
    ray.timeline(TIMELINE_FILENAME)
    append_gpu_timeline(TIMELINE_FILENAME, GPU_TIMELINE_FILENAME)
    print("Timeline log saved to: ", TIMELINE_FILENAME)


if __name__ == "__main__":
    # ray.init(object_store_memory=7e9)
    ray.init(num_cpus=NUM_CPUS)
    main()
    ray.shutdown()
