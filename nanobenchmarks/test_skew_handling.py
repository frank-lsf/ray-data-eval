import logging
import time

import numpy as np
import ray
from ray._raylet import StreamingObjectRefGenerator

"""
Proposed solution:
Central controller schedules executor tasks, each of which yields nano-batches.
The central controller estimates progress by extrapolating time taken to return these nano-batches.
If it detects a straggler, for example, if the straggler's average return time is >2x average, it marks this task as STRAGGLER.
If some other task is finished, we launch a work-stealing task which runs in reverse order and yields nano-batches of the straggler task from behind.
The central controller merges the output nano-batches, until we have the full output, then we cancel the outstanding task(s).
This way we don't have to send messages (except cancellation messages) to the task workers.
Requirement for task: the task must be non-iterative, i.e. output X does not depend on output X-1. This is true for all map tasks; may not be true for reducer tasks.
Open questions:
What if we want to use more than two workers? How should we split the input?
"""


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)


@ray.remote(num_returns="streaming")
def executor(inputs: list[float]):
    # Every X seconds, ask a coordinator to get the latest inputs
    for input in inputs:
        time.sleep(input)
        yield input


def run_experiment(size: int):
    inputs = np.random.rand(size).tolist()
    gen = executor.remote(inputs)
    not_ready = [gen]
    while len(not_ready) > 0:
        ready, not_ready = ray.wait(not_ready, timeout=0.5)
        if len(ready) == 0:
            logging.info("Nothing was ready in 0.5 seconds")
            continue
        for r in ready:
            if isinstance(r, StreamingObjectRefGenerator):
                try:
                    ref = next(r)
                    logging.info("%s", ray.get(ref))
                except StopIteration:
                    pass
                else:
                    not_ready.append(r)
            else:
                logging.info("%s", ray.get(r))

    logging.info("The End")


def main():
    ray.init("auto")

    run_experiment(size=10)


if __name__ == "__main__":
    main()
