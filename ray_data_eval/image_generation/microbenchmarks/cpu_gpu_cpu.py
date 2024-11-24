import argparse
import time

import numpy as np
import ray

from ray_data_eval.microbenchmarks.setting import (
    GB,
    TIME_UNIT,
    NUM_GPUS,
    FRAMES_PER_VIDEO,
    NUM_VIDEOS,
    NUM_FRAMES_TOTAL,
    FRAME_SIZE_B,
)
from ray_data_eval.microbenchmarks.raydata import timeline_utils


def bench(mem_limit):
    def produce(batch):
        time.sleep(TIME_UNIT * 8)
        for id in batch["id"]:
            yield {
                "id": [id],
                "image": [np.zeros(FRAME_SIZE_B, dtype=np.uint8)],
            }

    def consume(batch):
        time.sleep(TIME_UNIT * 4)
        return {"id": batch["id"]}

    def inference(batch):
        time.sleep(TIME_UNIT)
        return {"id": batch["id"], "image": [np.ones(FRAME_SIZE_B, dtype=np.uint8)]}

    NUM_CPUS = 10
    ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS, object_store_memory=mem_limit * GB)

    data_context = ray.data.DataContext.get_current()
    data_context.op_resource_reservation_ratio = 0
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = FRAME_SIZE_B
    data_context.is_budget_policy = True
    # data_context.is_conservative_policy = True

    ################################################################################
    # Radar-fused
    ################################################################################
    ds = ray.data.range(NUM_FRAMES_TOTAL, override_num_blocks=NUM_VIDEOS)
    ds = ds.map_batches(produce, batch_size=FRAMES_PER_VIDEO, concurrency=NUM_GPUS)
    ds = ds.map_batches(inference, batch_size=1, num_cpus=0, num_gpus=1)
    ds = ds.map_batches(consume, batch_size=2, concurrency=NUM_GPUS)
    ################################################################################

    # ################################################################################
    # # Radar-static
    # ################################################################################
    # ds = ray.data.range(NUM_FRAMES_TOTAL, override_num_blocks=NUM_VIDEOS)
    # ds = ds.map_batches(produce, batch_size=FRAMES_PER_VIDEO, concurrency=2)
    # ds = ds.map_batches(inference, batch_size=1, num_cpus=0, num_gpus=1)
    # ds = ds.map_batches(consume, batch_size=2, concurrency=6)
    # ################################################################################

    # ################################################################################
    # # Radar-full
    # ################################################################################
    # ds = ray.data.range(NUM_FRAMES_TOTAL, override_num_blocks=NUM_VIDEOS)
    # ds = ds.map_batches(produce, batch_size=FRAMES_PER_VIDEO)
    # ds = ds.map_batches(inference, batch_size=1, num_cpus=0, num_gpus=1)
    # ds = ds.map_batches(consume, batch_size=2)
    # ################################################################################

    start_time = time.time()
    for _ in ds.iter_batches(batch_size=FRAMES_PER_VIDEO):
        pass
    end_time = time.time()
    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))
    print(f"Total time: {end_time - start_time:.4f}s")
    timeline_utils.save_timeline_with_cpus_gpus(
        f"timeline_ray_data_{mem_limit}.json", NUM_CPUS, NUM_GPUS
    )
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=12
    )
    args = parser.parse_args()
    bench(args.mem_limit)
