import os

import pulp as pl

from ray_data_eval.common.pipeline import (
    ResourcesSpec,
    TaskSpec,
    SchedulingProblem,
    test_problem,
    training_problem,
)


def _get_executors_for_task(
    resources: ResourcesSpec, task: TaskSpec
) -> tuple[list[int], list[int]]:
    cpu_slots = [i for i in range(resources.cpu)]
    gpu_slots = [i + resources.cpu for i in range(resources.gpu)]
    if task.resources.cpu > 0:
        return cpu_slots, gpu_slots
    if task.resources.gpu > 0:
        return gpu_slots, cpu_slots
    return [], cpu_slots + gpu_slots


def solve(cfg: SchedulingProblem, *, solver=None, tidy=False) -> int:
    """
    Solve the scheduling problem using integer linear programming.

    :param cfg: Scheduling problem configuration.
    :param solver: PuLP solver to use. If None, use CPLEX with the number of threads equal to the
        number of CPUs.
    :param tidy: If True, add tidiness constraints. These do not affect the solution but make it
        easier to read.

    :return: The total time taken to execute all tasks.
    """
    if solver is None:
        solver = pl.CPLEX_CMD(threads=os.cpu_count())

    model = pl.LpProblem(cfg.name, pl.LpMinimize)

    schedule = pl.LpVariable.dicts(
        "x",
        [
            (i, j, t)
            for i in range(cfg.num_total_tasks)
            for j in range(cfg.resources.num_executors)
            for t in range(cfg.time_limit)
        ],
        cat="Binary",
    )
    buffer = pl.LpVariable.dicts(
        "b",
        [(op, t) for op in range(cfg.num_operators - 1) for t in range(cfg.time_limit + 1)],
        lowBound=0,
        cat="Integer",
    )
    buffer_to_consume = pl.LpVariable.dicts(
        "bc",
        [(op, t) for op in range(cfg.num_operators - 1) for t in range(cfg.time_limit + 1)],
        lowBound=0,
        cat="Integer",
    )

    # start[i, j, t] = 1 if task i starts at time t on slot j
    start = pl.LpVariable.dicts(
        "s",
        [
            (i, j, t)
            for i in range(cfg.num_total_tasks)
            for j in range(cfg.resources.num_executors)
            for t in range(cfg.time_limit)
        ],
        cat="Binary",
    )
    for i in range(cfg.num_total_tasks):
        for j in range(cfg.resources.num_executors):
            model += start[(i, j, 0)] == schedule[(i, j, 0)]
            for t in range(1, cfg.time_limit):
                model += start[(i, j, t)] >= schedule[(i, j, t)] - schedule[(i, j, t - 1)]
                model += start[(i, j, t)] <= schedule[(i, j, t)]
        # Ensure that each task starts exactly once
        model += (
            pl.lpSum(
                [
                    start[(i, j, t)]
                    for j in range(cfg.resources.num_executors)
                    for t in range(cfg.time_limit)
                ]
            )
            == 1
        )

    # finish[i, j, t] = 1 if task i finishes at time t on CPU slot j
    finish = pl.LpVariable.dicts(
        "f",
        [
            (i, j, t)
            for i in range(cfg.num_total_tasks)
            for j in range(cfg.resources.num_executors)
            for t in range(cfg.time_limit)
        ],
        cat="Binary",
    )
    for i in range(cfg.num_total_tasks):
        for j in range(cfg.resources.num_executors):
            model += finish[(i, j, cfg.time_limit - 1)] == schedule[(i, j, cfg.time_limit - 1)]
            for t in range(cfg.time_limit - 1):
                model += finish[(i, j, t)] >= schedule[(i, j, t)] - schedule[(i, j, t + 1)]
                model += finish[(i, j, t)] <= schedule[(i, j, t)]
        # Ensure that each task finishes exactly once
        model += (
            pl.lpSum(
                [
                    finish[(i, j, t)]
                    for j in range(cfg.resources.num_executors)
                    for t in range(cfg.time_limit)
                ]
            )
            == 1
        )

    # Constraint: One slot can run only one task at a time
    for j in range(cfg.resources.num_executors):
        for t in range(cfg.time_limit):
            model += pl.lpSum([schedule[(i, j, t)] for i in range(cfg.num_total_tasks)]) <= 1

    # Tidiness Constraint: Lower-indexed executors should be used first
    if tidy:
        for t in range(cfg.time_limit):
            for j in range(cfg.resources.num_executors - 1):
                model += pl.lpSum(
                    [schedule[(i, j, t)] for i in range(cfg.num_total_tasks)]
                ) >= pl.lpSum([schedule[(i, j + 1, t)] for i in range(cfg.num_total_tasks)])

    # Constraint: All tasks are assigned to exactly one slot and complete
    for i, task in enumerate(cfg.tasks):
        executors, non_executors = _get_executors_for_task(cfg.resources, task)
        model += (
            pl.lpSum([schedule[(i, j, t)] for j in executors for t in range(cfg.time_limit)])
            == task.duration
        )
        model += (
            pl.lpSum([schedule[(i, j, t)] for j in non_executors for t in range(cfg.time_limit)])
            == 0
        )

    # Constraint: All tasks must run contiguously for their entire duration
    for i in range(cfg.num_total_tasks):
        for j in range(cfg.resources.num_executors):
            # Ensure that the task either starts and runs for its entire duration or doesn't start
            for t in range(cfg.time_limit - cfg.tasks[i].duration + 1):
                # Task starts at time 't' and runs for 'task_time[i]' time units
                model += (
                    pl.lpSum([schedule[(i, j, t + k)] for k in range(cfg.tasks[i].duration)])
                    >= cfg.tasks[i].duration * start[i, j, t]
                )

    def _task_started_at(tid, tick):
        return pl.lpSum([start[(tid, j, tick)] for j in range(cfg.resources.num_executors)])

    def _task_finished_at(tid, tick):
        return pl.lpSum([finish[(tid, j, tick)] for j in range(cfg.resources.num_executors)])

    # Constraint: Buffer size is the total size of data in memory buffer. Buffer to consume size
    # is the total size of producer output not yet consumed.
    # When a producer finishes, it increases both buffer and buffer to consume sizes.
    # When a consumer starts, it decreases the buffer to consume size.
    # When a consumer finishes, it decreases the buffer size.
    for t in range(cfg.time_limit):
        buffer_increase = [[] for _ in range(cfg.num_operators - 1)]
        buffer_to_consume_decrease = [[] for _ in range(cfg.num_operators - 1)]
        buffer_decrease = [[] for _ in range(cfg.num_operators - 1)]
        for tid, task in enumerate(cfg.tasks):
            o = task.operator_idx
            if task.output_size > 0:
                buffer_increase[o].append(task.output_size * _task_finished_at(tid, t))
            if task.input_size > 0 and o > 0:
                buffer_to_consume_decrease[o - 1].append(task.input_size * _task_started_at(tid, t))
                buffer_decrease[o - 1].append(task.input_size * _task_finished_at(tid, t))
        for o in range(cfg.num_operators - 1):
            model += buffer[(o, t)] >= pl.lpSum(buffer_decrease[o])
            model += buffer[(o, t + 1)] == buffer[(o, t)] + pl.lpSum(buffer_increase[o]) - pl.lpSum(
                buffer_decrease[o]
            )
            model += buffer_to_consume[(o, t)] >= pl.lpSum(buffer_to_consume_decrease[o])
            model += buffer_to_consume[(o, t + 1)] == buffer_to_consume[(o, t)] + pl.lpSum(
                buffer_increase[o]
            ) - pl.lpSum(buffer_to_consume_decrease[o])

    # Constraint: Buffer size is bounded
    for t in range(cfg.time_limit):
        model += (
            pl.lpSum([buffer[(op, t)] for op in range(cfg.num_operators - 1)])
            <= cfg.buffer_size_limit
        )
        model += (
            pl.lpSum([buffer_to_consume[(op, t)] for op in range(cfg.num_operators - 1)])
            <= cfg.buffer_size_limit
        )

    # Constraint: Buffer must be empty at the beginning and end of the schedule
    for op in range(cfg.num_operators - 1):
        model += buffer[(op, 0)] == 0
        model += buffer[(op, cfg.time_limit)] == 0
        model += buffer_to_consume[(op, 0)] == 0
        model += buffer_to_consume[(op, cfg.time_limit)] == 0

    # Objective function: Minimize the latest finish time
    latest_finish_time = pl.LpVariable("lf", cat="Integer")
    for i in range(cfg.num_total_tasks):
        for j in range(cfg.resources.num_executors):
            for t in range(cfg.time_limit):
                model += latest_finish_time >= finish[(i, j, t)] * t

    model += latest_finish_time

    # Write down the problem
    model.writeLP(f"{cfg.name}.lp")

    # Solve the problem
    model.solve(solver=solver)

    # Print all variables
    # for v in model.variables():
    #     print(v.name, "=", v.varValue)

    # Output results
    print(">>> Status:", pl.LpStatus[model.status])

    max_time = int(pl.value(model.objective)) + 1
    separator_line = "++" + "-" * (max_time * 6 + 7) + "++"
    print(separator_line)
    for j in range(cfg.resources.num_executors):
        label = f"CPU{j}" if j < cfg.resources.cpu else f"GPU{j - cfg.resources.cpu}"
        print(f"|| {label} ||", end="")
        for t in range(max_time):
            idle = True
            for i in range(cfg.num_total_tasks):
                if pl.value(schedule[(i, j, t)]) == 1:
                    label = cfg.tasks[i].id
                    print(f" {label:<3} |", end="")
                    idle = False
            if idle:
                print("     |", end="")
        print("|")
    print(separator_line)
    for o in range(cfg.num_operators - 1):
        op = cfg.operators[o]
        print(f"|| buf{op.name} ||", end="")
        for t in range(max_time + 1):
            print(f" {int(pl.value(buffer[(o, t)])):<3} |", end="")
        print()
    print(separator_line)
    for o in range(cfg.num_operators - 1):
        op = cfg.operators[o]
        print(f"|| btc{op.name} ||", end="")
        for t in range(max_time + 1):
            print(f" {int(pl.value(buffer_to_consume[(o, t)])):<3} |", end="")
        print()
    print(separator_line)
    print("|| time ||", end="")
    for t in range(max_time):
        print(f" {t:<3} |", end="")
    print("|")
    print(separator_line)
    print("Total Run Time =", max_time)

    return max_time


def main():
    solve(training_problem)


if __name__ == "__main__":
    main()
