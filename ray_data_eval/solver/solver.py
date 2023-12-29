import os

import pulp as pl

from ray_data_eval.solver.config import SchedulingProblem


def solve(cfg: SchedulingProblem, *, solver=None) -> int:
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

    # schedule_flat[i, t] = 1 if task i is running at time t on any CPU slot
    schedule_flat = pl.LpVariable.dicts(
        "xf",
        [(i, t) for i in range(cfg.num_total_tasks) for t in range(cfg.time_limit)],
        cat="Binary",
    )
    for i in range(cfg.num_total_tasks):
        for t in range(cfg.time_limit):
            model += schedule_flat[(i, t)] == pl.lpSum([schedule[(i, j, t)] for j in range(cfg.num_execution_slots)])

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
            pl.lpSum([start[(i, j, t)] for j in range(cfg.num_execution_slots) for t in range(cfg.time_limit)]) == 1
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
            pl.lpSum([finish[(i, j, t)] for j in range(cfg.num_execution_slots) for t in range(cfg.time_limit)]) == 1
        )

    # Constraint: One CPU slot can run only one task at a time
    for j in range(cfg.num_execution_slots):
        for t in range(cfg.time_limit):
            model += pl.lpSum([schedule[(i, j, t)] for i in range(cfg.num_total_tasks)]) <= 1

    # Constraint: All tasks are assigned to exactly one CPU slot and complete
    for i in range(cfg.num_total_tasks):
        model += pl.lpSum([schedule_flat[(i, t)] for t in range(cfg.time_limit)]) == cfg.task_time[i]

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

    # Constraint: Buffer size is the total size of producer output not yet consumed
    # TODO: should a consumer release the buffer at start or finish? currently it's at start
    for t in range(cfg.time_limit):
        buffer_increase = pl.lpSum(
            [
                cfg.producer_output_size[i] * pl.lpSum([finish[(i, j, t)] for j in range(cfg.num_execution_slots)])
                for i in range(cfg.num_producers)
            ],
        )
        buffer_decrease = pl.lpSum(
            [
                cfg.consumer_input_size[i]
                * pl.lpSum([start[(i + cfg.num_producers, j, t)] for j in range(cfg.num_execution_slots)])
                for i in range(cfg.num_consumers)
            ],
        )
        model += buffer[t] >= buffer_decrease
        model += buffer[t + 1] == buffer[t] + buffer_increase - buffer_decrease

    # Constraint: Buffer size is bounded
    for t in range(cfg.time_limit):
        model += buffer[t] <= cfg.buffer_size_limit

    # Constraint: Buffer must be empty at the beginning and end of the schedule
    model += buffer[0] == 0
    model += buffer[cfg.time_limit] == 0

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
    print("||  buf ||", end="")
    for t in range(max_time):
        print(f" {int(pl.value(buffer[t])):<3} |", end="")
    print(f"| ({int(pl.value(buffer[max_time]))})")
    print(separator_line)
    print("|| time ||", end="")
    for t in range(max_time):
        print(f" {t:<3} |", end="")
    print("|")
    print(separator_line)
    print("Total Run Time =", max_time)

    return max_time


def main():
    solve(
        SchedulingProblem(
            num_producers=20,
            num_consumers=20,
            # producer_time=3,
            consumer_time=3,
            # producer_output_size=2,
            # consumer_input_size=2,
            time_limit=22,
            num_execution_slots=4,
            buffer_size_limit=10,
        ),
    )


if __name__ == "__main__":
    main()

# TODO: schedule looks wrong for p=1, c=2
