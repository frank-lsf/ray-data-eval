from dataclasses import dataclass
from enum import Enum
import logging

from ray_data_eval.common.types import TaskSpec

Tick = int


@dataclass
class DataItem:
    id: str
    block_id: int = 0
    producer: TaskSpec | None = None
    produced_at: Tick = -1
    consumer: TaskSpec | None = None
    consumed_at: Tick = -1


class Buffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items: list[DataItem] = []
        self._timeline: list[int] = [0]

    def __repr__(self):
        return f"Buffer({len(self._items)}/{self.capacity})"

    def tick(self):
        logging.info(f"[{self}] Tick")
        self._timeline.append(len(self._items))

    def push(self, at_tick: Tick, task: TaskSpec, size: int) -> DataItem | None:
        assert size > 0, (task, size)
        if len(self._items) + size > self.capacity:
            logging.debug(f"[{self}] Cannot push {task.id}: buffer full")
            return None
        for i in range(size):
            item = DataItem(
                id=task.id,
                block_id=i,
                producer=task,
                produced_at=at_tick,
                consumer=None,
                consumed_at=-1,
            )
            self._items.append(item)
        logging.debug(f"[{self}] Pushed {task.id}")
        return item

    def pop(self, at_tick: Tick, size: int) -> list[DataItem]:
        if len(self._items) < size:
            raise ValueError(f"Cannot pop {size} items: only {len(self._items)} in buffer")
        ret = []
        for _ in range(size):
            item = self._items.pop(0)
            item.consumer = item.producer
            item.consumed_at = at_tick
            ret.append(item)
        logging.info(f"[{self}] Popped {size} items")
        return ret

    def peek(self, size: int) -> list[DataItem]:
        if len(self._items) < size:
            return []
        return self._items[:size]

    def print_timeline(self):
        print("|| buf  ||", end="")
        for item in self._timeline:
            print(f" {item:<3} |", end="")
        print("|")


class TaskStateType(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PENDING_OUTPUT = "PENDING_OUTPUT"
    FINISHED = "FINISHED"


class TaskState:
    state: TaskStateType = TaskStateType.PENDING
    started_at: Tick = -1
    execution_started_at: Tick = -1
    execution_finished_at: Tick = -1
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
    def __init__(self, id: int, env: "ExecutionEnvironment"):
        self.id = id
        self.running_task: RunningTask | None = None
        self._env = env
        self._events: list[HistoryEvent] = []
        self._timeline: list[str] = []

    def __repr__(self):
        return f"Executor#{self.id}"

    def _try_finishing_running_task(self) -> bool:
        """
        Tries to add the running task's output to the buffer.
        """
        if self.running_task is None:
            return True
        if self.running_task.spec.output_size > 0:
            item = self._env.buffer.push(
                at_tick=self.running_task.started_at,
                task=self.running_task.spec,
                size=self.running_task.spec.output_size,
            )
            if item is None:
                logging.debug(f"[{self}] Cannot finish {self.running_task.spec.id}: buffer is full")
                return False
        return True

    def _finish_running_task(self) -> RunningTask:
        """
        Sets the running task to None and returns the finished task.
        """
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

    def _get_timeline_item(self):
        if self.running_task is None:
            return ""
        if self.running_task.remaining_ticks <= 0:
            return "!"
        return self.running_task.spec.id

    def tick(self) -> RunningTask | None:
        """
        Advances a tick. Returns the task that finished, if any.
        """
        self._timeline.append(self._get_timeline_item())
        if self.running_task is None:
            return None
        self.running_task.remaining_ticks -= 1
        if self.running_task.remaining_ticks <= 0:
            if self._try_finishing_running_task():
                self._env.update_task_state(self.running_task.spec.id, TaskStateType.FINISHED)
                return self._finish_running_task()
            else:
                self._env.update_task_state(self.running_task.spec.id, TaskStateType.PENDING_OUTPUT)
                return None

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
        logging.info(f"[{self}] Started {task}")
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
            print(f" {item:<3} |", end="")
        print("|")


class ExecutionEnvironment:
    def __init__(self, *, num_executors: int, buffer_size: int, tasks: list[TaskSpec]):
        self.task_specs = {t.id: t for t in tasks}
        self.task_states = {t.id: TaskState() for t in tasks}
        self.buffer = Buffer(capacity=buffer_size)
        self._executors = [Executor(i, self) for i in range(num_executors)]
        self._current_tick = 0

    def __repr__(self):
        return f"ExecutionEnvironment@{self._current_tick}"

    def update_task_state(self, tid: str, state: TaskStateType):
        self.task_states[tid].state = state
        if state == TaskStateType.RUNNING:
            self.task_states[tid].started_at = self._current_tick
        elif state == TaskStateType.PENDING_OUTPUT:
            self.task_states[tid].execution_finished_at = self._current_tick
        elif state == TaskStateType.FINISHED:
            self.task_states[tid].finished_at = self._current_tick

    def tick(self):
        logging.debug(f"[{self}] Tick")
        self._current_tick += 1
        for executor in self._executors:
            executor.tick()
        self.buffer.tick()

    def start_task(self, task: TaskSpec, executor_id: int) -> bool:
        if task.input_size > 0:
            inp = self.buffer.peek(task.input_size)
            if len(inp) < task.input_size:
                logging.info(f"[{self}] Cannot start {task.id}: input requirement not satisfied")
                return False
        started = self._executors[executor_id].start_task(task, self._current_tick)
        if started:
            self.buffer.pop(self._current_tick, task.input_size)
            self.update_task_state(task.id, TaskStateType.RUNNING)
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
        self.buffer.print_timeline()
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
