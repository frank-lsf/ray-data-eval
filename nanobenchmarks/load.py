import time

import numpy as np
import ray

DATA_ROOT = "/mnt/data/ray-data-eval"
DATA_INDEX = "/mnt/data/ray-data-eval/index/train.txt"
TOTAL_COUNT = 1281167


def load_index(*, limit: int) -> list[str]:
    with open(DATA_INDEX, "r") as f:
        paths = f.read().splitlines()
    ret = [f"{DATA_ROOT}/{path}" for path in paths]
    if limit > 0:
        return ret[:limit]
    return ret


def get_image_size(row: dict[str, np.ndarray]) -> dict[str, int]:
    return {"size": row["image"].size}


def test_load(*, limit: int = -1):
    filenames = load_index(limit=limit)

    start = time.perf_counter()
    ret = ray.data.read_images(filenames).map(get_image_size).sum("size")
    end = time.perf_counter()
    print(ret)
    print(f"Time: {end - start:.4f}s")
    return ret


def main():
    ray.init()
    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    test_load(limit=10000)


if __name__ == "__main__":
    main()
