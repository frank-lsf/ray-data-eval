import datetime
import time

import wandb
import ray
from ray_data_eval.common.types import SchedulingProblem
from ray.data._internal.execution.backpressure_policy import StreamingOutputBackpressurePolicy
import numpy as np

NUM_ROWS_PER_TASK = 10
STREMING_GEN_BUFFER_SIZE = 1
OP_OUTPUT_QUEUE_SIZE = 1

DATA_SIZE_BYTES = 1000 * 1000 * 100  # 100 MB
BLOCK_SIZE = int(DATA_SIZE_BYTES / NUM_ROWS_PER_TASK)  # 10 MB.
ENABLED_BACKPRESSURE_POLICIES_CONFIG_KEY = "backpressure_policies.enabled"


def start_ray(cfg: SchedulingProblem):
    ray.init(num_cpus=cfg.num_execution_slots)
    data_context = ray.data.DataContext.get_current()
    data_context.set_config(
        ENABLED_BACKPRESSURE_POLICIES_CONFIG_KEY,
        [
            StreamingOutputBackpressurePolicy,
        ],
    )
    data_context.set_config(
        StreamingOutputBackpressurePolicy.MAX_BLOCKS_IN_OP_OUTPUT_QUEUE_CONFIG_KEY,
        OP_OUTPUT_QUEUE_SIZE,
    )
    data_context.set_config(
        StreamingOutputBackpressurePolicy.MAX_BLOCKS_IN_GENERATOR_BUFFER_CONFIG_KEY,
        STREMING_GEN_BUFFER_SIZE,
    )
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = BLOCK_SIZE
    data_context.execution_options.resource_limits.object_store_memory = (
        cfg.buffer_size_limit * DATA_SIZE_BYTES
    )


def produce(batch):
    print("Produce task started", batch["id"])
    time.sleep(0.1)
    for id in batch["id"]:
        print("Producing", id)
        yield {
            "id": [id],
            "image": [np.zeros(BLOCK_SIZE, dtype=np.uint8)],
        }


def consume(batch):
    print("Consume task started", batch["id"])
    time.sleep(0.01)
    return {"id": batch["id"], "result": [0 for _ in batch["id"]]}


def save_ray_timeline():
    timestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"/tmp/ray-timeline-{timestr}.json"
    print(f"Save Ray timeline to {filename}")
    ray.timeline(filename=filename)
    wandb.save(filename)


def run_ray_data(cfg: SchedulingProblem):
    start = time.perf_counter()

    ds = ray.data.range(cfg.num_producers * NUM_ROWS_PER_TASK, parallelism=cfg.num_producers)
    ds = ds.map_batches(produce, batch_size=NUM_ROWS_PER_TASK)
    ds = ds.map_batches(consume, batch_size=None, num_cpus=0.9)

    for _ in ds.iter_batches(batch_size=NUM_ROWS_PER_TASK):
        pass

    run_time = time.perf_counter() - start
    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))
    save_ray_timeline()
    wandb.log({"run_time": run_time})


def run_experiment(cfg: SchedulingProblem):
    wandb.init(project="ray-data-eval", entity="raysort")
    wandb.config.update(cfg)
    start_ray(cfg)
    run_ray_data(cfg)


def main():
    backpressure_problem = SchedulingProblem(
        num_producers=20,
        num_consumers=20,  # Unused.
        num_execution_slots=6,
        buffer_size_limit=1,  # 100MB
    )

    run_experiment(backpressure_problem)


if __name__ == "__main__":
    main()
