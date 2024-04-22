import boto3
import json
import ray
import time
import csv
import os


class ChromeTracer:
    """
    A simple custom profiler that records event start and end time, to replace Ray's profiler due to observed issues with profiling gpu workload.
    https://github.com/ray-project/ray/blob/master/python/ray/_private/profiling.py#L84

    """

    def __init__(self, log_file, device_name="NVIDIA_A10G"):
        self.log_file = log_file
        self.device_name = device_name
        self.events = []

    def _add_event(self, name, phase, timestamp, cname="rail_load", extra_data=None):
        event = {
            "name": name,
            "ph": phase,
            "ts": timestamp,
            "pid": ray._private.services.get_node_ip_address(),
            "tid": "gpu:" + "NVIDIA_A10G",
            "cname": cname,
            "args": extra_data or {},
        }
        self.events.append(event)

    def __enter__(self):
        self.start_time = time.time() * 1000000
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time() * 1000000
        self._add_event(self.name, "B", self.start_time, self.extra_data)
        self._add_event(self.name, "E", self.end_time)

    def profile(self, name, extra_data=None):
        self.name = name
        self.extra_data = extra_data
        return self

    def save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.events, f)


def append_gpu_timeline(timeline_filename, gpu_timeline_filename):
    """
    Append GPU events log to the main log.

    """
    try:
        with open(timeline_filename, "r") as file:
            timeline = json.load(file)
            assert isinstance(timeline, list)

        with open(gpu_timeline_filename, "r") as gpu_file:
            gpu_timeline = json.load(gpu_file)
            assert isinstance(gpu_timeline, list)

        timeline += gpu_timeline

        with open(timeline_filename, "w") as file:
            json.dump(timeline, file)
    except Exception as e:
        print(f"Error occurred when appending GPU timeline: {e}")


def get_prefixes(bucket_name, prefix):
    """
    Each bucket_name, prefix combination creates a path that leads to a folder,
    which contains training data of the same label.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

    prefixes = []
    for response in response_iterator:
        if "CommonPrefixes" in response:
            prefixes.extend(
                [f"s3://{bucket_name}/" + cp["Prefix"] for cp in response["CommonPrefixes"]]
            )

    return prefixes, len(prefixes)


def postprocess(timeline_file, start_time, total_size, batch_size, csv_filename):
    with open(timeline_file, "r") as f:
        timeline = json.load(f)

    batch_finish_times = []
    for idx, event in enumerate(timeline):
        if event["cat"].startswith("task::Classifier.__call__"):
            event["cname"] = "rail_load"  # modify color
            elapsed_time = (event["ts"] + event["dur"]) / 1e6 - start_time
            batch_finish_times.append(elapsed_time)
            timeline[idx] = event
    json.dump(timeline, open(timeline_file, "w"))

    batch_finish_times = sorted(batch_finish_times)

    csv_ref = open(csv_filename, "w")
    writer = csv.writer(csv_ref)
    writer.writerow(["time_from_start", "number_of_rows_finished"])
    writer.writerow([0, 0])
    for i, finish_time in enumerate(batch_finish_times):
        num_rows_read = (i + 1) * batch_size
        if num_rows_read > total_size:
            num_rows_read = total_size
        writer.writerow([finish_time, num_rows_read])
    csv_ref.close()
    return


def get_size(input_path):
    count = 0
    for path in input_path:
        count += len(os.listdir(path))
    return count


def download_train_directories(
    bucket_name,
    prefix,
    percentage=1,
    output_file="kinetics-train-1-percent.txt",
):
    directories, count = get_prefixes(bucket_name, prefix)
    num_samples = len(directories) * percentage // 100
    directories = directories[:num_samples]

    with open(output_file, "w") as f:
        f.write(repr(directories))
    print(f"Downloaded {num_samples} directories to {output_file}")
    return directories


if __name__ == "__main__":
    print(None)
    # bucket_name = "ray-data-eval-us-west-2"
    # prefix = "kinetics/k700-2020/train/"
    # print(download_train_directories(bucket_name, prefix)[0])
    # postprocess(
    #     "/home/ubuntu/ray-data-eval/ray_data_eval/video_inference/video_inference_local_NVIDIA_A10G_batch_64.json",
    #     1713767961,
    #     640,
    #     64,
    #     "temp.csv",
    # )
    # print(get_size(["/home/ubuntu/kinetics/kinetics/k700-2020/train/abseiling"]))
