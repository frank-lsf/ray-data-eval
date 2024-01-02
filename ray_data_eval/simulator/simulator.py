from dataclasses import dataclass
from enum import Enum
import logging

from ray_data_eval.common.types import SchedulingProblem, TaskSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname).1s] %(message)s",
    handlers=[logging.StreamHandler()],
)

Tick = int


class TaskStateType(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class TaskState:
    state: TaskStateType = TaskStateType.PENDING
    started_at: Tick = -1
    finished_at: Tick = -1


TaskStateMap = dict[TaskSpec, TaskState]


@dataclass
class RunningTask:
    spec: TaskSpec
    started_at: Tick
    remaining_ticks: int


class HistoryEventType(Enum):
    TASK_STARTED = "TASK_STARTED"
    TASK_FINISHED = "TASK_FINISHED"


@dataclass
class HistoryEvent:
    tick: Tick
    type: HistoryEventType
    task: TaskSpec


class Executor:
    def __init__(self, id: int):
        self.id = id
        self.running_task: RunningTask | None = None
        self._events = []
        self._timeline = []

    def __repr__(self):
        return f"Executor<{self.id}>"

    def tick(self) -> RunningTask | None:
        self._timeline.append(self.running_task.spec.id if self.running_task else None)
        if self.running_task is None:
            return None
        self.running_task.remaining_ticks -= 1
        if self.running_task.remaining_ticks == 0:
            logging.info(f"[{self}] Finished {self.running_task.spec.id}")
            self._events.append(
                HistoryEvent(
                    tick=self.running_task.started_at,
                    type=HistoryEventType.TASK_STARTED,
                    task=self.running_task.spec,
                )
            )
            ret = self.running_task
            self.running_task = None
            return ret

    def start_task(self, task: TaskSpec, at_tick: Tick) -> bool:
        """
        Tries to start a task on this executor.
        Returns true if the task was started, false if it was not.
        """
        if self.running_task is not None:
            logging.debug(
                f"[{self}] Cannot start {task.id}: {self.running_task.spec.id} is running"
            )
            return False
        self.running_task = RunningTask(
            spec=task,
            started_at=at_tick,
            remaining_ticks=task.duration,
        )
        logging.info(f"[{self}] Started {task.id}")
        self._events.append(
            HistoryEvent(
                tick=at_tick,
                type=HistoryEventType.TASK_STARTED,
                task=task,
            )
        )
        return True

    def cancel_task(self):
        self.running_task = None

    def print_timeline(self):
        print(f"|| {self.id:4} ||", end="")
        for item in self._timeline:
            if item is None:
                print("     |", end="")
            else:
                print(f" {item:<3} |", end="")
        print("|")


class Buffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer: list[TaskSpec] = []

    def __repr__(self):
        return f"Buffer<size={self.capacity}>"

    def tick(self):
        pass


class ExecutionEnvironment:
    def __init__(self, *, num_executors: int, buffer_size: int, tasks: list[TaskSpec]):
        self.task_specs = {t.id: t for t in tasks}
        self.task_states = {t.id: TaskState() for t in tasks}
        self._executors = [Executor(i) for i in range(num_executors)]
        self._buffer = Buffer(capacity=buffer_size)
        self._current_tick = 0

    def __repr__(self):
        return f"ExecutionEnvironment@{self._current_tick}"

    def tick(self):
        logging.debug(f"[{self}] Tick")
        self._current_tick += 1
        for executor in self._executors:
            task = executor.tick()
            if task is not None:
                self.task_states[task.spec.id].state = TaskStateType.FINISHED
                self.task_states[task.spec.id].finished_at = self._current_tick

    def start_task(self, task: TaskSpec, executor_id: int) -> bool:
        started = self._executors[executor_id].start_task(task, self._current_tick)
        if started:
            self.task_states[task.id].state = TaskStateType.RUNNING
            self.task_states[task.id].started_at = self._current_tick
        return started

    def start_task_on_any_executor(self, task: TaskSpec) -> bool:
        for exec_id in range(len(self._executors)):
            if self.start_task(task, exec_id):
                self.task_states[task.id].state = TaskStateType.RUNNING
                self.task_states[task.id].started_at = self._current_tick
                return True
        return False

    def cancel_task(self, task: TaskSpec):
        raise NotImplementedError

    def print_timeline(self):
        max_time = self._current_tick
        separator_line = "++" + "-" * (max_time * 6 + 7) + "++"
        print(separator_line)
        for executor in self._executors:
            executor.print_timeline()
        print(separator_line)
        print("|| time ||", end="")
        for t in range(max_time):
            print(f" {t:<3} |", end="")
        print("|")
        print(separator_line)

    def check_all_tasks_finished(self):
        all_finished = True
        for tid, state in self.task_states.items():
            logging.info(f"{tid}: {state.state}")
            if state.state != TaskStateType.FINISHED:
                all_finished = False

        return all_finished


class SchedulingPolicy:
    def tick(self, _env: ExecutionEnvironment):
        logging.debug(f"[{self}] Tick")


class DoNothingSchedulingPolicy(SchedulingPolicy):
    pass


class GreedySchedulingPolicy(SchedulingPolicy):
    """
    A greedy policy that tries to start tasks as soon as possible,
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


def test_scheduling_policy(problem: SchedulingProblem, policy: SchedulingPolicy) -> bool:
    env = ExecutionEnvironment(
        num_executors=problem.num_execution_slots,
        buffer_size=problem.buffer_size_limit,
        tasks=problem.tasks,
    )

    for tick in range(problem.time_limit):
        logging.info(f"Tick {tick}")
        policy.tick(env)
        env.tick()
        print()

    env.print_timeline()
    ret = env.check_all_tasks_finished()
    logging.info(f"All tasks finished? {ret}")


def main():
    problem = SchedulingProblem(
        num_producers=5,
        num_consumers=5,
        producer_time=1,
        consumer_time=2,
        time_limit=15,
        num_execution_slots=2,
    )
    policy = GreedySchedulingPolicy()
    test_scheduling_policy(problem, policy)


if __name__ == "__main__":
    main()

"""
TODO:
- What should happen when a producer returns and the buffer is full?
- What should happen when a consumer starts but there is no data in the buffer?
"""
