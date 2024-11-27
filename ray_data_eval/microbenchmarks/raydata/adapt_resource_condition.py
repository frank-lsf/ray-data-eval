import argparse
import os
import time
import subprocess

import numpy as np
import ray

from ray_data_eval.microbenchmarks.setting import (
    GB,
    TIME_UNIT,
    NUM_CPUS,
    NUM_GPUS,
    FRAMES_PER_VIDEO,
    NUM_VIDEOS,
    NUM_FRAMES_TOTAL,
    FRAME_SIZE_B,
)
import timeline_utils

def expand_cluster():
    additional_num_pids = 4
    ray_start_command = f"ray start --address=10.0.33.210:6379 --num-cpus {additional_num_pids} --num-gpus 1"

    try:
        subprocess.run(ray_start_command, shell=True, check=True)
        print(
            f"Successfully started {additional_num_pids} workers. Run `ray status` to check the current cluster status."
        )
        time.sleep(5)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start {additional_num_pids} workers. Error: {e}")
            
    print(ray.available_resources())
    

def bench(mem_limit):
    os.environ["RAY_DATA_OP_RESERVATION_RATIO"] = "0"

    def produce(batch):
        time.sleep(TIME_UNIT * 10)
        for id in batch["id"]:
            yield {
                "id": [id],
                "image": [np.zeros(FRAME_SIZE_B, dtype=np.uint8)],
            }

    def consume(batch):
        time.sleep(TIME_UNIT * 3)
        return {"id": batch["id"], "image": [np.ones(FRAME_SIZE_B, dtype=np.uint8)]}

    def inference(batch):
        return {"id": batch["id"]}

    data_context = ray.data.DataContext.get_current()
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = FRAME_SIZE_B
    data_context.is_budget_policy = True
    # data_context.is_conservative_policy = True

    ray_start_head_command = f"ray start --head --num-cpus 4"
    try:
        subprocess.run(ray_start_head_command, shell=True, check=True)
        time.sleep(5)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start head node. Error: {e}")
            
        
    ray.init("auto")
    print(ray.available_resources())
    
    ds = ray.data.range(NUM_FRAMES_TOTAL, override_num_blocks=NUM_VIDEOS)
    ds = ds.map_batches(produce, batch_size=FRAMES_PER_VIDEO)
    ds = ds.map_batches(consume, batch_size=1, num_cpus=0.99)
    ds = ds.map_batches(inference, batch_size=1, num_cpus=0, num_gpus=1)

    start_time = time.time()
    seen_batch = 0
    expanded_cluster = False
    for _ in ds.iter_batches(batch_size=FRAMES_PER_VIDEO):
        seen_batch += 1
        if seen_batch > (NUM_FRAMES_TOTAL / FRAMES_PER_VIDEO / 3) and not expanded_cluster:
            expand_cluster()
            expanded_cluster = True
        print(f"Seen batch: {seen_batch}/{NUM_FRAMES_TOTAL / FRAMES_PER_VIDEO}")
        
    end_time = time.time()
    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))
    print(f"Total time: {end_time - start_time:.4f}s")
    
    ray.timeline(f"timeline_ray_data_{mem_limit}_original.json")
    timeline_utils.save_timeline_with_cpus_gpus(
        f"timeline_ray_data_{mem_limit}.json", NUM_CPUS, NUM_GPUS
    )
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=10
    )
    args = parser.parse_args()
    bench(args.mem_limit)
