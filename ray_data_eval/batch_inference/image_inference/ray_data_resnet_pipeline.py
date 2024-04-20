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

DATA_PERCENTAGE = 5
BUCKET_NAME = "ray-data-eval-us-west-2"
PREFIX = "imagenet/ILSVRC/Data/CLS-LOC/train/"
IMAGENET_LOCAL_DIR = f"/home/ubuntu/image-data-{DATA_PERCENTAGE}-percent/ILSVRC/Data/CLS-LOC/train/"
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
    transformed_img = transform(row["image"])
    return {
        "original_image": row["image"],
        "transformed_image": transformed_img,
    }


def main():
    start_time = time.time()
    rows_read = 0

    # Initialize tput csv file writer
    csv_file_obj = open(CSV_FILENAME, "w")
    writer = csv.writer(csv_file_obj)
    writer.writerow(["time_from_start", "number_of_rows_finished"])
    writer.writerow([0, 0])

    # Load the pretrained resnet model and move to GPU if one is available.
    weights = ResNet152_Weights.IMAGENET1K_V1
    device = torch.device("cuda")
    model = models.resnet152(weights=weights).to(device)
    model.eval()

    ds = ray.data.read_images(
        INPUT_PATH,
        mode="RGB",
    )

    transformed_ds = ds.map(preprocess_image)

    tracer = ChromeTracer(GPU_TIMELINE_FILENAME, ACCELERATOR)
    last_batch_time = time.time()
    for batch in transformed_ds.iter_batches(batch_size=BATCH_SIZE):
        print(f"Time to read batch: {time.time() - last_batch_time}")
        with tracer.profile("task:gpu_execution"):
            with torch.inference_mode():
                torch_batch = torch.from_numpy(batch["transformed_image"]).to(device)
                prediction = model(torch_batch)
                predicted_classes = prediction.argmax(dim=1).detach().cpu()
            labels = [weights.meta["categories"][i] for i in predicted_classes]
            print(labels)
            # print_gpu_memory_usage()}

        last_batch_time = time.time()

        # Write time_from_start, number_of_rows_finished to csv file
        rows_read += batch["transformed_image"].shape[0]
        writer.writerow([last_batch_time - start_time, rows_read])

    print(transformed_ds.stats())
    print("Total images processed: ", rows_read)

    # Save and combine cpu, gpu timeline view
    tracer.save()
    ray.timeline(TIMELINE_FILENAME)
    append_gpu_timeline(TIMELINE_FILENAME, GPU_TIMELINE_FILENAME)
    print("Timeline log saved to: ", TIMELINE_FILENAME)


if __name__ == "__main__":
    ray.init(num_cpus=NUM_CPUS)
    main()
    ray.shutdown()
