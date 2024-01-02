import time

import ray

from data_utils import load_index


def get_bytecount(row):
    return {"size": len(row["bytes"])}


def test_load(*, limit: int = -1):
    filenames = load_index(limit=limit)

    start = time.perf_counter()

    ds = ray.data.read_binary_files(filenames, parallelism=100)
    ds = ds.map(get_bytecount)
    ret = ds.sum("size")

    end = time.perf_counter()
    print()
    print(ret)
    print(f"Time: {end - start:.4f}s")
    return ret


def main():
    ray.init()
    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    test_load(limit=10000)


if __name__ == "__main__":
    main()
