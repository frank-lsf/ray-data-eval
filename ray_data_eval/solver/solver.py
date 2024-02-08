import os

import pulp as pl

from ray_data_eval.common.pipeline import SchedulingProblem, training_problem


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
            for j in range(cfg.num_execution_slots)
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

    # schedule_flat[i, t] = 1 if task i is running at time t on any CPU slot
    schedule_flat = pl.LpVariable.dicts(
        "xf",
        [(i, t) for i in range(cfg.num_total_tasks) for t in range(cfg.time_limit)],
        cat="Binary",
    )
    for i in range(cfg.num_total_tasks):
        for t in range(cfg.time_limit):
            model += schedule_flat[(i, t)] == pl.lpSum(
                [schedule[(i, j, t)] for j in range(cfg.num_execution_slots)]
            )

    # start[i, j, t] = 1 if task i starts at time t on CPU slot j
    start = pl.LpVariable.dicts(
        "s",
        [
            (i, j, t)
            for i in range(cfg.num_total_tasks)
            for j in range(cfg.num_execution_slots)
            for t in range(cfg.time_limit)
        ],
        cat="Binary",
    )
    for i in range(cfg.num_total_tasks):
        for j in range(cfg.num_execution_slots):
            model += start[(i, j, 0)] == schedule[(i, j, 0)]
            for t in range(1, cfg.time_limit):
                model += start[(i, j, t)] >= schedule[(i, j, t)] - schedule[(i, j, t - 1)]
                model += start[(i, j, t)] <= schedule[(i, j, t)]
        # Ensure that each task starts at most once
        model += (
            pl.lpSum(
                [
                    start[(i, j, t)]
                    for j in range(cfg.num_execution_slots)
                    for t in range(cfg.time_limit)
                ]
            )
            == 1
        )

    def _task_started_at_or_before(tid, tick):
        """Returns 1 if task tid starts at or before time tick, 0 otherwise."""
        return pl.lpSum(
            [start[(tid, j, t)] for j in range(cfg.num_execution_slots) for t in range(tick)]
        )

    # finish[i, j, t] = 1 if task i finishes at time t on CPU slot j
    finish = pl.LpVariable.dicts(
        "f",
        [
            (i, j, t)
            for i in range(cfg.num_total_tasks)
            for j in range(cfg.num_execution_slots)
            for t in range(cfg.time_limit)
        ],
        cat="Binary",
    )
    for i in range(cfg.num_total_tasks):
        for j in range(cfg.num_execution_slots):
            model += finish[(i, j, cfg.time_limit - 1)] == schedule[(i, j, cfg.time_limit - 1)]
            for t in range(cfg.time_limit - 1):
                model += finish[(i, j, t)] >= schedule[(i, j, t)] - schedule[(i, j, t + 1)]
                model += finish[(i, j, t)] <= schedule[(i, j, t)]
        # Ensure that each task finishes at most once
        model += (
            pl.lpSum(
                [
                    finish[(i, j, t)]
                    for j in range(cfg.num_execution_slots)
                    for t in range(cfg.time_limit)
                ]
            )
            == 1
        )

    # Constraint: One CPU slot can run only one task at a time
    for j in range(cfg.num_execution_slots):
        for t in range(cfg.time_limit):
            model += pl.lpSum([schedule[(i, j, t)] for i in range(cfg.num_total_tasks)]) <= 1

    # Tidiness Constraint: Lower-indexed CPUs should be used first
    if tidy:
        for t in range(cfg.time_limit):
            for j in range(cfg.num_execution_slots - 1):
                model += pl.lpSum(
                    [schedule[(i, j, t)] for i in range(cfg.num_total_tasks)]
                ) >= pl.lpSum([schedule[(i, j + 1, t)] for i in range(cfg.num_total_tasks)])

    # Constraint: All tasks are assigned to exactly one CPU slot and complete
    for i in range(cfg.num_total_tasks):
        model += (
            pl.lpSum([schedule_flat[(i, t)] for t in range(cfg.time_limit)])
            == cfg.tasks[i].duration
        )

    # Constraint: All tasks must run contiguously for their entire duration
    for i in range(cfg.num_total_tasks):
        for j in range(cfg.num_execution_slots):
            # Ensure that the task either starts and runs for its entire duration or doesn't start
            for t in range(cfg.time_limit - cfg.tasks[i].duration + 1):
                # Task starts at time 't' and runs for 'task_time[i]' time units
                model += (
                    pl.lpSum([schedule[(i, j, t + k)] for k in range(cfg.tasks[i].duration)])
                    >= cfg.tasks[i].duration * start[i, j, t]
                )

    def _task_started_at(tid, tick):
        return pl.lpSum([start[(tid, j, tick)] for j in range(cfg.num_execution_slots)])

    def _task_finished_at(tid, tick):
        return pl.lpSum([finish[(tid, j, tick)] for j in range(cfg.num_execution_slots)])

    # Constraint: Buffer size is the total size of data in memory buffer. Buffer to consume size
    # is the total size of producer output not yet consumed.
    # When a producer finishes, it increases both buffer and buffer to consume sizes.
    # When a consumer starts, it decreases the buffer to consume size.
    # When a consumer finishes, it decreases the buffer size.
    for t in range(cfg.time_limit):
        buffer_increase = [[]] * (cfg.num_operators - 1)
        buffer_to_consume_decrease = [[]] * (cfg.num_operators - 1)
        buffer_decrease = [[]] * (cfg.num_operators - 1)
        for tid, task in enumerate(cfg.tasks):
            op = task.operator_idx
            if task.output_size > 0:
                buffer_increase[op].append(task.output_size * _task_finished_at(tid, t))
            if task.input_size > 0 and op > 0:
                buffer_to_consume_decrease[op - 1].append(
                    task.input_size * _task_started_at(tid, t)
                )
                buffer_decrease[op - 1].append(task.input_size * _task_finished_at(tid, t))
        for op in range(cfg.num_operators - 1):
            model += buffer[(op, t)] >= pl.lpSum(buffer_decrease[op])
            model += buffer[(op, t + 1)] == buffer[(op, t)] + pl.lpSum(
                buffer_increase[op]
            ) - pl.lpSum(buffer_decrease[op])
            model += buffer_to_consume[(op, t)] >= pl.lpSum(buffer_to_consume_decrease[op])
            model += buffer_to_consume[(op, t + 1)] == buffer_to_consume[(op, t)] + pl.lpSum(
                buffer_increase[op]
            ) - pl.lpSum(buffer_to_consume_decrease[op])

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
        for j in range(cfg.num_execution_slots):
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
    for j in range(cfg.num_execution_slots):
        print(f"|| {j:4} ||", end="")
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
    for op in range(cfg.num_operators - 1):
        print(f"|| buf{op} ||", end="")
        for t in range(max_time + 1):
            print(f" {int(pl.value(buffer[(op, t)])):<3} |", end="")
        print()
    for op in range(cfg.num_operators - 1):
        print(f"|| btc{op} ||", end="")
        for t in range(max_time + 1):
            print(f" {int(pl.value(buffer_to_consume[(op, t)])):<3} |", end="")
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
