import json


def postprocess_logs(file_path: str = "flink_logs.log"):
    data = []
    with open(file_path) as f:
        for line in f:
            line = line.replace("'", '"')
            line = json.loads(line)
            tid = line["cat"][-1]
            if "inference" in line["cat"]:
                line["tid"] = "GPU:" + tid
            else:
                line["tid"] = "CPU:" + tid
            data.append(line)

    json.dump(data, open("flink_logs_parsed.json", "w"), indent=2)


if __name__ == "__main__":
    postprocess_logs()
