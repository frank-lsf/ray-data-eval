from dataclasses import dataclass


@dataclass
class OperatorSpec:
    name: str
    operator_idx: int
    num_tasks: int
    duration: int
    input_size: int
    output_size: int
    num_cpus: int


@dataclass
class TaskSpec:
    id: str
    operator_idx: int
    duration: int
    input_size: int
    output_size: int
    num_cpus: int


def _get_tasks(operator_list: list[OperatorSpec]):
    tasks = []
    # Reversed so that downstream tasks are prioritized when
    # there are items in the buffer.
    for _, operator in enumerate(reversed(operator_list)):
        tasks.extend(
            [
                TaskSpec(
                    f"{operator.name}{i}",
                    operator.operator_idx,
                    operator.duration,
                    operator.input_size,
                    operator.output_size,
                    operator.num_cpus,
                )
                for i in range(operator.num_tasks)
            ]
        )
    return tasks


@dataclass
class SchedulingProblem:
    operator_list: list[OperatorSpec]
    name: str
    num_execution_slots: int
    time_limit: int
    buffer_size_limit: int

    def __post_init__(self):
        self.tasks = _get_tasks(self.operator_list)


test_problem = SchedulingProblem(
    [
        OperatorSpec(
            name="P",
            operator_idx=0,
            num_tasks=8,
            duration=1,
            input_size=0,
            output_size=1,
            num_cpus=1,
        ),
        OperatorSpec(
            name="C",
            operator_idx=1,
            num_tasks=8,
            duration=2,
            input_size=1,
            output_size=0,
            num_cpus=1,
        ),
    ],
    name="test_problem",
    time_limit=12,
    num_execution_slots=4,
    buffer_size_limit=2,
)

multi_stage_problem = SchedulingProblem(
    [
        OperatorSpec(
            name="A",
            operator_idx=0,
            num_tasks=8,
            duration=1,
            input_size=0,
            output_size=1,
            num_cpus=1,
        ),
        OperatorSpec(
            name="B",
            operator_idx=1,
            num_tasks=8,
            duration=2,
            input_size=1,
            output_size=2,
            num_cpus=1,
        ),
        OperatorSpec(
            name="C",
            operator_idx=2,
            num_tasks=4,
            duration=1,
            input_size=4,
            output_size=10,
            num_cpus=1,
        ),
        OperatorSpec(
            name="D",
            operator_idx=3,
            num_tasks=2,
            duration=2,
            input_size=20,
            output_size=0,
            num_cpus=1,
        ),
    ],
    name="multi_stage_problem",
    time_limit=15,
    num_execution_slots=4,
    buffer_size_limit=100,
)

producer_consumer_problem = SchedulingProblem(
    [
        OperatorSpec(
            name="P",
            operator_idx=0,
            num_tasks=10,
            duration=1,
            input_size=0,
            output_size=1,
            num_cpus=1,
        ),
        OperatorSpec(
            name="C",
            operator_idx=1,
            num_tasks=10,
            duration=2,
            input_size=1,
            output_size=0,
            num_cpus=1,
        ),
    ],
    name="producer_consumer_problem",
    time_limit=15,
    buffer_size_limit=2,
    num_execution_slots=3,
)

long_problem = SchedulingProblem(
    [
        OperatorSpec(
            name="A",
            operator_idx=0,
            num_tasks=50,
            duration=1,
            input_size=0,
            output_size=1,
            num_cpus=1,
        ),
        OperatorSpec(
            name="B",
            operator_idx=1,
            num_tasks=50,
            duration=2,
            input_size=1,
            output_size=2,
            num_cpus=1,
        ),
        OperatorSpec(
            name="C",
            operator_idx=2,
            num_tasks=25,
            duration=1,
            input_size=4,
            output_size=0,
            num_cpus=1,
        ),
    ],
    name="long_problem",
    time_limit=300,
    buffer_size_limit=5000,
    num_execution_slots=3,
)

problems = [test_problem, multi_stage_problem, producer_consumer_problem, long_problem]
