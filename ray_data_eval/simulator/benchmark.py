import logging
from ray_data_eval.common.pipeline import SchedulingProblem, problems
from ray_data_eval.simulator.environment import ExecutionEnvironment
from ray_data_eval.simulator.policies import (  # noqa F401
    GreedySchedulingPolicy,
    GreedyWithBufferSchedulingPolicy,
    GreedyAndAnticipatingSchedulingPolicy,
    SchedulingPolicy,
    RatesEqualizingSchedulingPolicy,
)

logging.disable(logging.CRITICAL)


def test_scheduling_policy(problem: SchedulingProblem, policy: SchedulingPolicy) -> bool:
    env = ExecutionEnvironment(
        num_executors=problem.num_execution_slots,
        buffer_size=problem.buffer_size_limit,
        tasks=problem.tasks,
        scheduling_policy=policy,
    )

    for tick in range(problem.time_limit):
        logging.info("-" * 60)
        logging.info(f"Tick {tick}")
        env.tick()

    env.print_timeline()
    ret = env.check_all_tasks_finished()
    logging.info(f"All tasks finished? {ret}")


def main():
    for problem in problems:
        for policy in [
            GreedySchedulingPolicy(problem),
            GreedyWithBufferSchedulingPolicy(problem),
            GreedyAndAnticipatingSchedulingPolicy(problem),
            RatesEqualizingSchedulingPolicy(problem),
        ]:
            env = ExecutionEnvironment(
                num_executors=problem.num_execution_slots,
                buffer_size=problem.buffer_size_limit,
                tasks=problem.tasks,
                scheduling_policy=policy,
            )

            used_time = 0
            is_finished = False
            for _ in range(problem.time_limit):
                env.tick()
                used_time += 1
                is_finished = env.check_all_tasks_finished()
                if is_finished:
                    break

            print(str(policy), "Finished" if is_finished else "Not finished", used_time)


if __name__ == "__main__":
    main()
