import os
import pandas as pd

producer_task_name = "task::MapBatches(produce_video_slices)"
consumer_task_name = "task::MapBatches(preprocess_video)"
no_parallel_task_name = "task::MapBatches(produce_video_slices)->MapBatches(preprocess_video)"
gpu_task_name = "task::MapBatches(Classifier)"

COLORS = {
    producer_task_name: "rail_response",
    consumer_task_name: "cq_build_passed",
    no_parallel_task_name: "rail_load",
    gpu_task_name: "cq_build_failed",
}


DIRECTORY = "/home/ubuntu/ray-data-eval/ray_data_eval/video_inference/long_video/logs/"
file_paths = [
    os.path.join(DIRECTORY, "Dynamic parallelism.json"),
    os.path.join(DIRECTORY, "$N=4, M=4$.json"),
    os.path.join(DIRECTORY, "$N=1, M=7$.json"),
    os.path.join(DIRECTORY, "$N=7, M=1$.json"),
]


def process_file(file_path):
    df = pd.read_json(file_path)

    filtered_df = df[df["cat"].isin(COLORS.keys())]

    # Modify the event title
    filtered_df.loc[:, "name"] = filtered_df["cat"].apply(
        lambda x: x.replace("task::MapBatches(", "").replace(")", "")
    )

    # Modify the color
    filtered_df.loc[:, "cname"] = filtered_df["cat"].apply(lambda x: COLORS[x])

    new_file_name = os.path.basename(file_path).replace(".json", "_modified.json")
    new_file_path = os.path.join(DIRECTORY, new_file_name)
    filtered_df.to_json(new_file_path, orient="records")

    print(f"Processed and saved modified data to {new_file_path}")


# Process each file
for file_path in file_paths:
    process_file(file_path)


if __name__ == "__main__":
    for file_path in file_paths:
        process_file(file_path)
