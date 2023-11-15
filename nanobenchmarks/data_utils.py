DATA_ROOT = "/mnt/data/ray-data-eval"
DATA_INDEX = "/mnt/data/ray-data-eval/index/train.txt"
TOTAL_COUNT = 1281167


def load_index(*, limit: int, prefix: str = "") -> list[str]:
    with open(DATA_INDEX, "r") as f:
        paths = f.read().splitlines()
    ret = [f"{prefix}{DATA_ROOT}/{path}" for path in paths]
    if limit > 0:
        return ret[:limit]
    return ret
