import datetime
import subprocess
import time

import ray
import wandb

from ray_data_eval.common.types import SchedulingProblem, test_problem

DATA_SIZE_BYTES = 1000 * 1000 * 100  # 100 MB
TIME_UNIT = 1  # seconds


def start_ray(cfg: SchedulingProblem):
    subprocess.run("ray stop -f", shell=True, check=True)
    ray.init(
        num_cpus=cfg.num_execution_slots,
        num_gpus=cfg.num_execution_slots,
    )
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.resource_limits.cpu = cfg.num_execution_slots
    ctx.execution_options.resource_limits.gpu = cfg.num_execution_slots
    ctx.execution_options.resource_limits.object_store_memory = (
        cfg.buffer_size_limit * DATA_SIZE_BYTES
    )


def producer(row, *, cfg: SchedulingProblem):
    i = row["item"]
    data = b"1" * (DATA_SIZE_BYTES * cfg.producer_output_size[i])
    time.sleep(TIME_UNIT * cfg.producer_time[i])
    return {"data": data, "idx": i}


def consumer(row, *, cfg: SchedulingProblem):
    data = row["data"]
    time.sleep(TIME_UNIT * cfg.consumer_time[row["idx"]])
    return {"result": len(data)}


def save_ray_timeline():
    timestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"/tmp/ray-timeline-{timestr}.json"
    ray.timeline(filename=filename)
    wandb.save(filename)


def run_ray_data(cfg: SchedulingProblem):
    if cfg.num_producers != cfg.num_consumers:
        raise NotImplementedError(f"num_producers != num_consumers: {cfg}")
    start = time.perf_counter()

    items = list(range(cfg.num_producers))
    ds = ray.data.from_items(items, parallelism=cfg.num_producers)

    ds = ds.map(producer, fn_kwargs={"cfg": cfg})
    ds = ds.map(consumer, fn_kwargs={"cfg": cfg}, num_gpus=1)

    ret = 0
    for row in ds.iter_rows():
        ret += row["result"]

    run_time = time.perf_counter() - start
    print(f"\n{ret:,}")
    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))
    save_ray_timeline()
    wandb.log({"run_time": run_time})
    return ret


def run_experiment(cfg: SchedulingProblem):
    wandb.init(project="ray-data-eval", entity="raysort")
    wandb.config.update(cfg)

    start_ray(cfg)
    run_ray_data(cfg)


def main():
    run_experiment(test_problem)


if __name__ == "__main__":
    main()
