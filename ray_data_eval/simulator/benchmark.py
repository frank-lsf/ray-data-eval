import logging

from ray_data_eval.common.pipeline import problems
from ray_data_eval.simulator.environment import ExecutionEnvironment
from ray_data_eval.simulator.policies import (
    GreedyPolicy,
    GreedyWithBufferPolicy,
    GreedyOracleProducerFirstPolicy,
    GreedyOracleConsumerFirstPolicy,
    RatesEqualizingPolicy,
    ConcurrencyCapPolicy,
    DelayPolicy,
)


def main():
    logging.disable(logging.CRITICAL)
    for problem in problems:
        print("Problem:", problem.name)
        for policy in [
            GreedyPolicy(problem),
            GreedyWithBufferPolicy(problem),
            GreedyOracleProducerFirstPolicy(problem),
            GreedyOracleConsumerFirstPolicy(problem),
            RatesEqualizingPolicy(problem),
            ConcurrencyCapPolicy(problem),
            DelayPolicy(problem),
        ]:
            env = ExecutionEnvironment(
                resources=problem.resources,
                buffer_size=problem.buffer_size_limit,
                tasks=problem.tasks,
                scheduling_policy=policy,
            )

            used_time = 0
            is_finished = False
            for _ in range(problem.time_limit):
                env.tick()
                used_time += 1
                if env.check_all_tasks_finished():
                    is_finished = True
                    break

            print(str(policy), "Finished" if is_finished else "Not finished", used_time)
        print("---")


if __name__ == "__main__":
    main()
