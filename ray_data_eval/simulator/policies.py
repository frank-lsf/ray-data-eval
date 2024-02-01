import logging

import numpy as np

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
        self._cumulative_output_size = 0  # Tracks the total output size of all tasks scheduled

    def __repr__(self):
        return "GreedyWithBufferSchedulingPolicy"

    def tick(self, env: ExecutionEnvironment):
        super().tick(env)
        for tid, task_state in env.task_states.items():
            if task_state.state == TaskStateType.PENDING:
                net_output_size = env.task_specs[tid].output_size - env.task_specs[tid].input_size
                logging.debug(
                    f"net_output_size={net_output_size}, "
                    f"cumulative_output_size={self._cumulative_output_size}, "
                    f"limit={self.buffer_size_limit}"
                )
                if net_output_size > 0:
                    if self._cumulative_output_size + net_output_size > self.buffer_size_limit:
                        logging.info(f"[{self}] Not starting {tid} to avoid buffer overflow")
                        continue
                logging.debug(f"[{self}] Trying to start {tid}")
                task = env.task_specs[tid]
                if not env.start_task_on_any_executor(task):
                    logging.debug(f"[{self}] Cannot not start {tid}")

    def on_task_state_change(self, task: TaskSpec, task_state: TaskState):
        super().on_task_state_change(task, task_state)
        logging.info(f"{task.id} changed state to {task_state.state}")
        if task_state.state == TaskStateType.RUNNING:
            self._cumulative_output_size += task.output_size
        elif task_state.state == TaskStateType.FINISHED:
            self._cumulative_output_size -= task.input_size


class GreedyAndAnticipatingSchedulingPolicy(SchedulingPolicy):
    """
    The perfect policy that knows when a consumer is about to finish, and times the next producer
    so that they finish at the same time.
    """

    def __init__(self, problem: SchedulingProblem):
        super().__init__(problem)
        self.buffer_size_limit = problem.buffer_size_limit
        self._buffer_diff = np.zeros(problem.time_limit * 2)
        self._current_tick = 0

    def __repr__(self):
        return "GreedyAndAnticipatingSchedulingPolicy"

    def _buffer_size_at(self, tick: int):
        return np.sum(self._buffer_diff[: tick + 1])

    def tick(self, env: ExecutionEnvironment):
        super().tick(env)
        for tid, task_state in env.task_states.items():
            if task_state.state == TaskStateType.PENDING:
                net_output_size = env.task_specs[tid].output_size - env.task_specs[tid].input_size
                logging.debug(
                    f"net_output_size={net_output_size}, "
                    f"buffer_diff={self._buffer_diff}, "
                    f"limit={self.buffer_size_limit}"
                )
                if net_output_size > 0:
                    finish_at = self._current_tick + env.task_specs[tid].duration
                    if self._buffer_size_at(finish_at) + net_output_size > self.buffer_size_limit:
                        logging.info(
                            f"[{self}] Not starting {tid} to due to anticipated buffer overflow at {finish_at}"
                        )
                        continue
                logging.debug(f"[{self}] Trying to start {tid}")
                task = env.task_specs[tid]
                if not env.start_task_on_any_executor(task):
                    logging.debug(f"[{self}] Cannot not start {tid}")
        self._current_tick += 1

    def on_task_state_change(self, task: TaskSpec, task_state: TaskState):
        super().on_task_state_change(task, task_state)
        logging.info(f"{task.id} changed state to {task_state.state}")
        if task_state.state == TaskStateType.RUNNING:
            self._buffer_diff[self._current_tick + task.duration] += (
                task.output_size - task.input_size
            )


class RatesEqualizingSchedulingPolicy(SchedulingPolicy):
    """
    A policy where consumer's rate of consuming input data to be the same as the producer's rate of producing output data.
    """

    def __init__(self, problem: SchedulingProblem):
        super().__init__(problem)
        self.buffer_size_limit = problem.buffer_size_limit
        self._current_tick = 0
        self._cumulative_output_size = 0
        producer_rate = problem.producer_output_size[0] / problem.producer_time[0]
        consumer_rate = problem.consumer_input_size[0] / problem.consumer_time[0]
        self.producer_consumer_ratio = producer_rate / consumer_rate

    def __repr__(self):
        return "RatesEqualizingSchedulingPolicy"

    def _is_producer_task(self, env: ExecutionEnvironment, tid: int):
        return env.task_specs[tid].output_size > 0

    def _count_scheduled_tasks(self, env: ExecutionEnvironment):
        num_producers = 0
        num_consumers = 0

        for tid, task_state in env.task_states.items():
            if (
                task_state.state == TaskStateType.RUNNING
                or task_state.state == TaskStateType.FINISHED
            ):
                if not self._is_producer_task(env, tid):
                    num_consumers += 1
                else:
                    num_producers += 1
        return num_producers, num_consumers

    def tick(self, env: ExecutionEnvironment):
        super().tick(env)

        num_producers, num_consumers = self._count_scheduled_tasks(env)
        # Iterate consumer first.
        # env.task_states by default stores consumers first.
        for tid, task_state in env.task_states.items():
            if task_state.state == TaskStateType.PENDING:
                if (
                    self._is_producer_task(env, tid)
                    and num_consumers > 0
                    and (num_producers + 1) / num_consumers > self.producer_consumer_ratio
                ):
                    logging.info(
                        f"[{self}] Not starting producer {tid} to keep ratio. num_producers: {num_producers}, num_consumers: {num_consumers}, ratio: {self.producer_consumer_ratio}"
                    )
                    continue

                logging.debug(f"[{self}] Trying to start {tid}")
                task = env.task_specs[tid]
                if not env.start_task_on_any_executor(task):
                    logging.debug(f"[{self}] Cannot not start {tid}")

    def on_task_state_change(self, task: TaskSpec, task_state: TaskState):
        super().on_task_state_change(task, task_state)
        logging.info(f"{task.id} changed state to {task_state.state}")
        if task_state.state == TaskStateType.RUNNING:
            self._cumulative_output_size += task.output_size
        elif task_state.state == TaskStateType.FINISHED:
            self._cumulative_output_size -= task.input_size
