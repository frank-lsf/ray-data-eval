import logging

from ray_data_eval.common.types import SchedulingProblem, TaskSpec
from ray_data_eval.simulator.environment import (
    ExecutionEnvironment,
    SchedulingPolicy,
    TaskState,
    TaskStateType,
)


class DoNothingSchedulingPolicy(SchedulingPolicy):
    pass


class GreedySchedulingPolicy(SchedulingPolicy):
    """
    A greedy policy that tries to start tasks as soon as possible
    on the first executor that has capacity.
    """

    def __repr__(self):
        return "GreedySchedulingPolicy"

    def tick(self, env: ExecutionEnvironment):
        super().tick(env)
        for tid, task_state in env.task_states.items():
            if task_state.state == TaskStateType.PENDING:
                logging.debug(f"[{self}] Trying to start {tid}")
                task = env.task_specs[tid]
                if not env.start_task_on_any_executor(task):
                    logging.debug(f"[{self}] Cannot not start {tid}")


class GreedyWithBufferSchedulingPolicy(SchedulingPolicy):
    """
    A greedy policy, except that it will not schedule more producers
    than the buffer size.
    """

    def __init__(self, problem: SchedulingProblem):
        super().__init__(problem)
        self.buffer_size_limit = problem.buffer_size_limit
        self.cumulative_output_size = 0  # Tracks the total output size of all tasks scheduled

    def __repr__(self):
        return "GreedyWithBufferSchedulingPolicy"

    def tick(self, env: ExecutionEnvironment):
        super().tick(env)
        for tid, task_state in env.task_states.items():
            if task_state.state == TaskStateType.PENDING:
                output_size = env.task_specs[tid].output_size
                logging.debug(
                    f"output_size={output_size}, cumulative_output_size={self.cumulative_output_size}, limit={self.buffer_size_limit}"
                )
                if (
                    output_size > 0
                    and self.cumulative_output_size + output_size > self.buffer_size_limit
                ):
                    logging.info(f"[{self}] Not starting {tid} to avoid buffer overflow")
                    continue
                logging.debug(f"[{self}] Trying to start {tid}")
                task = env.task_specs[tid]
                if not env.start_task_on_any_executor(task):
                    logging.debug(f"[{self}] Cannot not start {tid}")

    def on_task_state_change(self, task: TaskSpec, task_state: TaskState):
        super().on_task_state_change(task, task_state)
        if task_state.state == TaskStateType.RUNNING:
            self.cumulative_output_size += task.output_size - task.input_size
