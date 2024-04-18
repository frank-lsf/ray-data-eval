import boto3
from boto3.s3.transfer import TransferConfig
import os
import time

import ray

# Download settings
# Replace `s3_downloaded_images` with intended storage path.
DEST_DIR = os.path.join(
    os.getenv("HOME"),
    "s3_downloaded_images",
)
BUCKET_NAME = "ray-data-eval-us-west-2"
PREFIX = "imagenet/ILSVRC/Data/CLS-LOC"

# Ray + S3 client settings
RAY_NUM_CPU = 512
S3_CLIENT_MAX_CONCURRENCY = 8192
NUM_PARTITIONS = 10000


def get_file_list(bucket=BUCKET_NAME):
    s3 = boto3.client("s3", region_name="us-west-2")
    paginator = s3.get_paginator(
        "list_objects_v2"
    )  # paginator abstracts the logic of handling continuation tokens

    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=PREFIX):
        if "Contents" in page:
            for file in page["Contents"]:
                files.append(f"s3://{bucket}/{file['Key']}")

    return files


@ray.remote
def download(file_list):
    s3 = boto3.client("s3", region_name="us-west-2")
    transfer_config = TransferConfig(max_concurrency=S3_CLIENT_MAX_CONCURRENCY)

    for uri in file_list:
        bucket, key = uri.replace("s3://", "").split("/", 1)
        local_path = os.path.join(DEST_DIR, key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path, Config=transfer_config)


start = time.time()
file_list = get_file_list()
end = time.time()
print("get_file_list wall time:", end - start)  # For Imagenet, this takes around 320s.

# with open("temp.txt", "w") as f:
#     f.write(repr(file_list))

# with open("temp.txt", "r") as f:
#     file_list = eval(f.read())

start = time.time()
num_partitions = NUM_PARTITIONS
chunk_size = len(file_list) // num_partitions + (len(file_list) % num_partitions > 0)
partitions = [file_list[i : i + chunk_size] for i in range(0, len(file_list), chunk_size)]

print("num items:", len(file_list))
print("num partitions:", num_partitions)
print("chunk size:", chunk_size)

ray.init(num_cpus=RAY_NUM_CPU)
results = ray.get([download.remote(partition) for partition in partitions])
end = time.time()
print("download wall time:", end - start)
