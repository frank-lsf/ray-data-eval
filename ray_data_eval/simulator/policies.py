import logging

import numpy as np

from ray_data_eval.common.pipeline import SchedulingProblem, TaskSpec
from ray_data_eval.simulator.environment import (
    ExecutionEnvironment,
    SchedulingPolicy,
    TaskState,
    TaskStateType,
)


class DoNothingPolicy(SchedulingPolicy):
    pass


class GreedyPolicy(SchedulingPolicy):
    """
    A greedy policy that tries to start tasks as soon as possible
    on the first executor that has capacity.
    """

    def __repr__(self):
        return "GreedyPolicy"

    def tick(self, env: ExecutionEnvironment):
        super().tick(env)
        for tid, task_state in env.task_states.items():
            if task_state.state == TaskStateType.PENDING:
                logging.debug(f"[{self}] Trying to start {tid}")
                task = env.task_specs[tid]
                if not env.start_task_on_any_executor(task):
                    logging.debug(f"[{self}] Cannot not start {tid}")


class GreedyWithBufferPolicy(SchedulingPolicy):
    """
    A greedy policy, except that it will not schedule more producers
    than the buffer size.
    """

    def __init__(self, problem: SchedulingProblem):
        super().__init__(problem)
        self.buffer_size_limit = problem.buffer_size_limit
        self._cumulative_output_size = 0  # Tracks the total output size of all tasks scheduled

    def __repr__(self):
        return "GreedyWithBufferPolicy"

    def tick(self, env: ExecutionEnvironment):
        super().tick(env)
        for tid, task_state in env.task_states.items():
            if task_state.state != TaskStateType.PENDING:
                continue
            task = env.task_specs[tid]
            net_output_size = task.output_size - task.input_size
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
            if not env.start_task_on_any_executor(task):
                logging.debug(f"[{self}] Cannot not start {tid}")

    def on_task_state_change(self, task: TaskSpec, task_state: TaskState):
        super().on_task_state_change(task, task_state)
        logging.info(f"{task.id} changed state to {task_state.state}")
        if task_state.state == TaskStateType.RUNNING:
            self._cumulative_output_size += task.output_size
        elif task_state.state == TaskStateType.FINISHED:
            self._cumulative_output_size -= task.input_size


class GreedyOracleProducerFirstPolicy(SchedulingPolicy):
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
        return "GreedyOracleProducerFirstPolicy"

    def _buffer_size_at(self, tick: int):
        return np.sum(self._buffer_diff[: tick + 1])

    def _get_task_states(self, env: ExecutionEnvironment) -> list[tuple[str, TaskState]]:
        return env.task_states.items()

    def tick(self, env: ExecutionEnvironment):
        super().tick(env)
        for tid, task_state in self._get_task_states(env):
            if task_state.state != TaskStateType.PENDING:
                continue
            task = env.task_specs[tid]
            net_output_size = task.output_size - task.input_size
            logging.debug(
                f"net_output_size={net_output_size}, "
                f"buffer_diff={self._buffer_diff}, "
                f"limit={self.buffer_size_limit}"
            )
            if net_output_size > 0:
                finish_at = self._current_tick + task.duration
                if self._buffer_size_at(finish_at) + net_output_size > self.buffer_size_limit:
                    logging.info(
                        f"[{self}] Not starting {tid} to due to anticipated buffer overflow at {finish_at}"
                    )
                    continue
            logging.debug(f"[{self}] Trying to start {tid}")
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


class GreedyOracleConsumerFirstPolicy(GreedyOracleProducerFirstPolicy):
    """
    Same as GreedyOracleProducerFirstPolicy except that it schedules consumers first.
    """

    def __repr__(self):
        return "GreedyOracleConsumerFirstPolicy"

    def _get_task_states(self, env: ExecutionEnvironment) -> list[tuple[str, TaskState]]:
        ret = super()._get_task_states(env)
        return sorted(ret, key=lambda x: x[0].startswith("C"), reverse=True)


class RatesEqualizingPolicy(SchedulingPolicy):
    """
    A policy where consumer's rate of consuming input data to be the same as the producer's rate of producing output data.
    """

    def __init__(self, problem: SchedulingProblem):
        super().__init__(problem)
        self.operator_ratios = []
        self.operator_running_duration = [0] * len(problem.operators)
        self.task_finished = dict()
        for idx, operator in enumerate(problem.operators):
            if idx == 0:
                self.operator_ratios.append(1)
            else:
                input_rate = operator.input_size / operator.duration
                output_rate = (
                    problem.operators[idx - 1].output_size / problem.operators[idx - 1].duration
                )
                self.operator_ratios.append(
                    output_rate / input_rate * self.operator_ratios[idx - 1]
                )
        logging.info(f"[{self}] Operator ratios: {self.operator_ratios}")

    def __repr__(self):
        return "RatesEqualizingPolicy"

    def _update_operator_running_duration(self, env: ExecutionEnvironment):
        for tid, task_state in env.task_states.items():
            operator_idx = env.task_specs[tid].operator_idx
            if task_state.state == TaskStateType.RUNNING:
                self.operator_running_duration[operator_idx] += 1
            # Add the last time tick to running_duration.
            if task_state.state == TaskStateType.FINISHED and tid not in self.task_finished:
                self.operator_running_duration[operator_idx] += 1
                self.task_finished[tid] = True

    def _try_start_task(self, env: ExecutionEnvironment, tid: int):
        task = env.task_specs[tid]
        logging.debug(f"[{self}] Trying to start {tid}")
        if not env.start_task_on_any_executor(task):
            logging.debug(f"[{self}] Cannot not start {tid}")
            return False
        return True

    def _get_most_upstream_pending_operator(self, env: ExecutionEnvironment):
        most_upstream_pending_operator = len(self.operator_ratios)
        for tid, task_state in env.task_states.items():
            if task_state.state == TaskStateType.PENDING:
                most_upstream_pending_operator = min(
                    most_upstream_pending_operator, env.task_specs[tid].operator_idx
                )
        return most_upstream_pending_operator

    def tick(self, env: ExecutionEnvironment):
        super().tick(env)
        self._update_operator_running_duration(env)
        most_upstream_pending_operator = self._get_most_upstream_pending_operator(env)
        logging.info(
            f"[{self}] {[self.operator_running_duration[operator_idx] for operator_idx in range(len(self.operator_ratios))]}, {self.operator_ratios}, "
            f"most_upstream_pending_operator: {most_upstream_pending_operator}"
        )
        makes_progress = True
        while makes_progress:
            makes_progress = False
            for tid, task_state in env.task_states.items():
                if task_state.state == TaskStateType.PENDING:
                    operator_idx = env.task_specs[tid].operator_idx

                    # Liveness condition.
                    # Tasks are sorted in descending order of operator index.
                    # Prioritized downstreaming tasks.
                    if operator_idx == most_upstream_pending_operator:
                        makes_progress = makes_progress or self._try_start_task(env, tid)
                    # Maintain ratio with the successor.
                    elif (
                        # Not the last operator.
                        operator_idx + 1 < len(self.operator_ratios)
                        # Next operator has already launched tasks.
                        and self.operator_running_duration[operator_idx + 1] > 0
                        # Exceed the ratio.
                        and (self.operator_running_duration[operator_idx] + 1)
                        / self.operator_running_duration[operator_idx + 1]
                        > self.operator_ratios[operator_idx]
                        / self.operator_ratios[operator_idx + 1]
                    ):
                        logging.info(f"[{self}] Not starting task {tid} to keep ratio.")
                    # Maintain ratio with the predecessor.
                    elif (
                        # Not the first operator.
                        operator_idx >= 1
                        and self.operator_running_duration[operator_idx - 1]
                        / (self.operator_running_duration[operator_idx] + 1)
                        < self.operator_ratios[operator_idx - 1]
                        / self.operator_ratios[operator_idx]
                    ):
                        logging.info(f"[{self}] Not starting task {tid} to keep ratio.")
                    else:
                        makes_progress = makes_progress or self._try_start_task(env, tid)

            most_upstream_pending_operator = self._get_most_upstream_pending_operator(env)
