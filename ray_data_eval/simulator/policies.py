import logging

import numpy as np

from ray_data_eval.common.pipeline import SchedulingProblem, TaskSpec
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
        self.operator_ratios = []
        for idx, operator in enumerate(problem.operator_list):
            if idx == 0:
                self.operator_ratios.append(1)
            else:
                input_rate = operator.input_size / operator.duration
                output_rate = (
                    problem.operator_list[idx - 1].output_size
                    / problem.operator_list[idx - 1].duration
                )
                self.operator_ratios.append(
                    output_rate / input_rate * self.operator_ratios[idx - 1]
                )
        logging.info(f"[{self}] Operator ratios: {self.operator_ratios}")

    def __repr__(self):
        return "RatesEqualizingSchedulingPolicy"

    def _count_num_tasks_per_operator(self, env: ExecutionEnvironment, operator_idx: int):
        num_tasks = 0
        for tid, task_state in env.task_states.items():
            if (
                task_state.state == TaskStateType.RUNNING
                or task_state.state == TaskStateType.FINISHED
            ):
                if env.task_specs[tid].operator_idx == operator_idx:
                    num_tasks += 1
        return num_tasks

    def _try_start_task(self, env: ExecutionEnvironment, tid: int):
        task = env.task_specs[tid]
        logging.debug(f"[{self}] Trying to start {tid}")
        if not env.start_task_on_any_executor(task):
            logging.debug(f"[{self}] Cannot not start {tid}")
            return False
        return True

    def tick(self, env: ExecutionEnvironment):
        super().tick(env)

        # env.task_states by default stores later operators first.
        logging.info(
            f"[{self}] {[self._count_num_tasks_per_operator(env, operator_idx) for operator_idx in range(4)]} {self.operator_ratios}"
        )
        task_started = False
        for tid, task_state in env.task_states.items():
            if task_state.state == TaskStateType.PENDING:
                operator_idx = env.task_specs[tid].operator_idx
                current_operator_num = self._count_num_tasks_per_operator(env, operator_idx)
                next_operator_num = self._count_num_tasks_per_operator(env, operator_idx + 1)

                # Liveness condition
                if operator_idx == 0 and not task_started:
                    task_started = self._try_start_task(env, tid)

                # If the operator has pending intput
                elif env.can_get_task_input(env.task_specs[tid]):
                    task_started = self._try_start_task(env, tid)

                # Maintain ratio.
                elif (
                    # Not the last operator
                    operator_idx + 1 < len(self.operator_ratios)
                    # Next operator has already launched tasks
                    and next_operator_num > 0
                    # Exceed the ratio.
                    and (current_operator_num + 1) / next_operator_num
                    > self.operator_ratios[operator_idx] / self.operator_ratios[operator_idx + 1]
                ):
                    logging.debug(
                        f"[{self}] Not starting producer {tid} to keep ratio. num_producers: {current_operator_num}, num_consumers: {next_operator_num}, ratio: {self.operator_ratios[operator_idx - 1]}"
                    )
                else:
                    task_started = self._try_start_task(env, tid)
