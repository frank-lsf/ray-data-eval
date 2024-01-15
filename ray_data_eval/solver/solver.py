import os

import pulp as pl

from ray_data_eval.common.types import SchedulingProblem, test_problem


def solve(cfg: SchedulingProblem, *, solver=None, tidy=True) -> int:
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
    buffer = pl.LpVariable.dicts("b", range(cfg.time_limit + 1), lowBound=0, cat="Integer")
    buffer_to_consume = pl.LpVariable.dicts(
        "bc", range(cfg.time_limit + 1), lowBound=0, cat="Integer"
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

    # Tidiness Constraint: Tasks with lower index should start no later than tasks with higher index
    # NOTE: This seems to be buggy for CPLEX.
    if tidy and not isinstance(solver, pl.CPLEX_CMD):
        for i in range(cfg.num_total_tasks - 1):
            if i == cfg.num_producers - 1:
                # C0 should not need to start before Pn starts
                continue
            for t in range(cfg.time_limit):
                model += _task_started_at_or_before(i, t) >= _task_started_at_or_before(i + 1, t)

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
            pl.lpSum([schedule_flat[(i, t)] for t in range(cfg.time_limit)]) == cfg.task_time[i]
        )

    # Constraint: All tasks must run contiguously for their entire duration
    for i in range(cfg.num_total_tasks):
        for j in range(cfg.num_execution_slots):
            # Ensure that the task either starts and runs for its entire duration or doesn't start
            for t in range(cfg.time_limit - cfg.task_time[i] + 1):
                # Task starts at time 't' and runs for 'task_time[i]' time units
                model += (
                    pl.lpSum([schedule[(i, j, t + k)] for k in range(cfg.task_time[i])])
                    >= cfg.task_time[i] * start[i, j, t]
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
        # Increase buffer when a producer finishes
        buffer_increase = pl.lpSum(
            [
                cfg.producer_output_size[i] * _task_finished_at(i, t)
                for i in range(cfg.num_producers)
            ],
        )
        # Decrease buffer to consume in use when a consumer starts
        buffer_to_consume_decrease = pl.lpSum(
            [
                cfg.consumer_input_size[i] * _task_started_at(i + cfg.num_producers, t)
                for i in range(cfg.num_consumers)
            ],
        )
        # Decrease buffer when a consumer finishes
        buffer_decrease = pl.lpSum(
            [
                cfg.consumer_input_size[i] * _task_finished_at(i + cfg.num_producers, t)
                for i in range(cfg.num_consumers)
            ],
        )

        model += buffer[t] >= buffer_decrease
        model += buffer[t + 1] == buffer[t] + buffer_increase - buffer_decrease
        model += buffer_to_consume[t] >= buffer_to_consume_decrease
        model += (
            buffer_to_consume[t + 1]
            == buffer_to_consume[t] + buffer_increase - buffer_to_consume_decrease
        )

    # Constraint: Buffer size is bounded
    for t in range(cfg.time_limit):
        model += buffer[t] <= cfg.buffer_size_limit
        model += buffer_to_consume[t] <= cfg.buffer_size_limit

    # Constraint: Buffer must be empty at the beginning and end of the schedule
    model += buffer[0] == 0
    model += buffer[cfg.time_limit] == 0
    model += buffer_to_consume[0] == 0
    model += buffer_to_consume[cfg.time_limit] == 0

    # Objective function: Minimize the latest finish time
    latest_finish_time = pl.LpVariable("lf", lowBound=0, cat="Integer")
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
                    label = f"P{i}" if i < cfg.num_producers else f"C{i - cfg.num_producers}"
                    print(f" {label:<3} |", end="")
                    idle = False
            if idle:
                print("     |", end="")
        print("|")
    print(separator_line)
    print("||  buf ||", end="")
    for t in range(max_time + 1):
        print(f" {int(pl.value(buffer[t])):<3} |", end="")
    print()
    print("||  btc ||", end="")
    for t in range(max_time + 1):
        print(f" {int(pl.value(buffer_to_consume[t])):<3} |", end="")
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
    # solve(test_problem)
    solve(SchedulingProblem(num_producers=4, num_consumers=4, time_limit=10), tidy=True)


if __name__ == "__main__":
    main()
