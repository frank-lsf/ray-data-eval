import os
import json


def make_trace_event_spark():
    # logs/spark-events/app-20240109133439-0022
    app_id = "app-20240127211827-0054"

    log_directory = os.getenv("SPARK_EVENTS_PATH")
    output_directory = os.getenv("SPARK_TRACE_EVENT_PATH")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    events = []

    file = os.path.join(log_directory, app_id)
    with open(file, "r") as f:
        job_id = None
        for line in f:
            event = json.loads(line)

            if event["Event"] == "SparkListenerJobStart":
                job_id = event["Job ID"]

            if event["Event"] == "SparkListenerTaskEnd":
                start_time = event["Task Info"]["Launch Time"]
                end_time = event["Task Info"]["Finish Time"]
                task_id = event["Task Info"]["Task ID"]
                executor_id = event["Task Info"]["Executor ID"]
                host = event["Task Info"]["Host"]

                event = {
                    "cat": f"job:{job_id}",
                    "name": f"task:{task_id}",
                    "pid": host,
                    "tid": f"worker:{executor_id}",
                    "ts": start_time,  # start time (ms)
                    "dur": (end_time - start_time),  # duration (ms)
                    "ph": "X",
                    "args": {},
                }

                # if job_id == 2:  # producer
                #     event["cname"] = "generic_work"
                # elif job_id == 4:  # consumer
                #     event["cname"] = "rail_animation"

                events.append(event)

    output_file = os.path.join(output_directory, f"timeline_{app_id}.json")
    with open(output_file, "w") as output_f:
        json.dump(events, output_f, indent=2)
        print(f"Output saved to: {output_file}")


def main():
    make_trace_event_spark()


if __name__ == "__main__":
    main()
