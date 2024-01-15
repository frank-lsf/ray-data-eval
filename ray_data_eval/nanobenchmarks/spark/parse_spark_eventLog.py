import os
import json


def spark_timeline():
    # logs/spark-events/app-20240109133439-0022

    log_directory = os.getenv("SPARK_EVENTS_PATH")
    app_id = "app-20240115141650-0146"

    job_info_dict = {}

    file = os.path.join(log_directory, app_id)
    with open(file, "r") as f:
        for line in f:
            event = json.loads(line)

            if event["Event"] == "SparkListenerJobStart":
                job_id = event["Job ID"]
                job_info_dict[job_id] = {
                    "start_time": (event["Submission Time"]),
                    "tasks": [],
                }

            elif event["Event"] == "SparkListenerTaskEnd":
                task_info = event["Task End Reason"]
                start_time = event["Task Info"]["Launch Time"]
                end_time = event["Task Info"]["Finish Time"]
                task_id = event["Task Info"]["Task ID"]

                job_info_dict[job_id]["tasks"].append(
                    {
                        "task_id": task_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "task_info": task_info,
                    }
                )

    for job_id, job_info in job_info_dict.items():
        print(f"Job ID: {job_id}, Start Time: {job_info['start_time']}")
        for task_info in job_info["tasks"]:
            print(
                f"  Task ID: {task_info['task_id']}, Start Time: {task_info['start_time']}, End Time: {task_info['end_time']}, Task Info: {task_info['task_info']}"
            )


def main():
    spark_timeline()


if __name__ == "__main__":
    main()
