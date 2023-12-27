import os

import pulp as pl


def solve(
    *,
    problem_title: str = "TaskScheduling",
    num_producers: int = 1,
    num_consumers: int = 1,
    producer_time: int | list[int] = 1,
    consumer_time: int | list[int] = 1,
    producer_output_size: int | list[int] = 1,
    consumer_input_size: int | list[int] = 1,
    num_execution_slots: int = 1,
    time_limit: int = 4,
    buffer_size_limit: int = 1,
) -> int:
    producer_time = [producer_time] * num_producers if isinstance(producer_time, int) else producer_time
    consumer_time = [consumer_time] * num_consumers if isinstance(consumer_time, int) else consumer_time
    producer_output_size = (
        [producer_output_size] * num_producers if isinstance(producer_output_size, int) else producer_output_size
    )
    consumer_input_size = (
        [consumer_input_size] * num_consumers if isinstance(consumer_input_size, int) else consumer_input_size
    )
    assert len(producer_time) == num_producers, (producer_time, num_producers)
    assert len(consumer_time) == num_consumers, (consumer_time, num_consumers)
    assert len(producer_output_size) == num_producers, (producer_output_size, num_producers)
    assert len(consumer_input_size) == num_consumers, (consumer_input_size, num_consumers)

    num_total_tasks = num_producers + num_consumers
    task_time = producer_time + consumer_time

    model = pl.LpProblem(problem_title, pl.LpMinimize)

    schedule = pl.LpVariable.dicts(
        "x",
        [(i, j, t) for i in range(num_total_tasks) for j in range(num_execution_slots) for t in range(time_limit)],
        cat="Binary",
    )
    buffer = pl.LpVariable.dicts("b", range(time_limit + 1), lowBound=0, cat="Integer")

    # schedule_flat[i, t] = 1 if task i is running at time t on any CPU slot
    schedule_flat = pl.LpVariable.dicts(
        "xf",
        [(i, t) for i in range(num_total_tasks) for t in range(time_limit)],
        cat="Binary",
    )
    for i in range(num_total_tasks):
        for t in range(time_limit):
            model += schedule_flat[(i, t)] == pl.lpSum([schedule[(i, j, t)] for j in range(num_execution_slots)])

    # start[i, j, t] = 1 if task i starts at time t on CPU slot j
    start = pl.LpVariable.dicts(
        "s",
        [(i, j, t) for i in range(num_total_tasks) for j in range(num_execution_slots) for t in range(time_limit)],
        cat="Binary",
    )
    for i in range(num_total_tasks):
        for j in range(num_execution_slots):
            model += start[(i, j, 0)] == schedule[(i, j, 0)]
            for t in range(1, time_limit):
                model += start[(i, j, t)] >= schedule[(i, j, t)] - schedule[(i, j, t - 1)]
                model += start[(i, j, t)] <= schedule[(i, j, t)]
        # Ensure that each task starts at most once
        model += pl.lpSum([start[(i, j, t)] for j in range(num_execution_slots) for t in range(time_limit)]) == 1

    # finish[i, j, t] = 1 if task i finishes at time t on CPU slot j
    finish = pl.LpVariable.dicts(
        "f",
        [(i, j, t) for i in range(num_total_tasks) for j in range(num_execution_slots) for t in range(time_limit)],
        cat="Binary",
    )
    for i in range(num_total_tasks):
        for j in range(num_execution_slots):
            model += finish[(i, j, time_limit - 1)] == schedule[(i, j, time_limit - 1)]
            for t in range(time_limit - 1):
                model += finish[(i, j, t)] >= schedule[(i, j, t)] - schedule[(i, j, t + 1)]
                model += finish[(i, j, t)] <= schedule[(i, j, t)]
        # Ensure that each task finishes at most once
        model += pl.lpSum([finish[(i, j, t)] for j in range(num_execution_slots) for t in range(time_limit)]) == 1

    # Constraint: One CPU slot can run only one task at a time
    for j in range(num_execution_slots):
        for t in range(time_limit):
            model += pl.lpSum([schedule[(i, j, t)] for i in range(num_total_tasks)]) <= 1

    # Constraint: All tasks are assigned to exactly one CPU slot and complete
    for i in range(num_total_tasks):
        model += pl.lpSum([schedule_flat[(i, t)] for t in range(time_limit)]) == task_time[i]

    # Constraint: All tasks must run contiguously for their entire duration
    for i in range(num_total_tasks):
        for j in range(num_execution_slots):
            # Ensure that the task either starts and runs for its entire duration or doesn't start
            for t in range(time_limit - task_time[i] + 1):
                # Task starts at time 't' and runs for 'task_time[i]' time units
                model += (
                    pl.lpSum([schedule[(i, j, t + k)] for k in range(task_time[i])]) >= task_time[i] * start[i, j, t]
                )

    # Constraint: Buffer size is the total size of producer output not yet consumed
    # TODO: should a consumer release the buffer at start or finish? currently it's at start
    for t in range(time_limit):
        buffer_increase = pl.lpSum(
            [
                producer_output_size[i] * pl.lpSum([finish[(i, j, t)] for j in range(num_execution_slots)])
                for i in range(num_producers)
            ],
        )
        buffer_decrease = pl.lpSum(
            [
                consumer_input_size[i]
                * pl.lpSum([start[(i + num_producers, j, t)] for j in range(num_execution_slots)])
                for i in range(num_consumers)
            ],
        )
        model += buffer[t] >= buffer_decrease
        model += buffer[t + 1] == buffer[t] + buffer_increase - buffer_decrease

    # Constraint: Buffer size is bounded
    for t in range(time_limit):
        model += buffer[t] <= buffer_size_limit

    # Constraint: Buffer must be empty at the beginning and end of the schedule
    model += buffer[0] == 0
    model += buffer[time_limit] == 0

    # Objective function: Minimize the latest finish time
    latest_finish_time = pl.LpVariable("lf", lowBound=0, cat="Integer")
    for i in range(num_total_tasks):
        for j in range(num_execution_slots):
            for t in range(time_limit):
                model += latest_finish_time >= finish[(i, j, t)] * t

    model += latest_finish_time

    # Write down the problem
    model.writeLP(f"{problem_title}.lp")

    # Solve the problem
    # model.solve(solver=pl.PULP_CBC_CMD(threads=os.cpu_count()))
    model.solve(solver=pl.CPLEX_CMD(threads=os.cpu_count()))

    # Print all variables
    # for v in model.variables():
    #     print(v.name, "=", v.varValue)

    # Output results
    print(">>> Status:", pl.LpStatus[model.status])

    separator_line = "++" + "-" * (time_limit * 6 + 7) + "++"
    print(separator_line)
    for j in range(num_execution_slots):
        print(f"|| {j:4} ||", end="")
        for t in range(time_limit):
            idle = True
            for i in range(num_total_tasks):
                if pl.value(schedule[(i, j, t)]) == 1:
                    label = f"P{i}" if i < num_producers else f"C{i - num_producers}"
                    print(f" {label:<3} |", end="")
                    idle = False
            if idle:
                print("     |", end="")
        print("|")
    print("||  buf ||", end="")
    for t in range(time_limit):
        print(f" {int(pl.value(buffer[t])):<3} |", end="")
    print(f"| ({int(pl.value(buffer[time_limit]))})")
    print(separator_line)
    print("|| time ||", end="")
    for t in range(time_limit):
        print(f" {t:<3} |", end="")
    print("|")
    print(separator_line)
    print("Latest Finish Time =", pl.value(model.objective))
    print("Total Run Time =", pl.value(model.objective) + 1)

    return pl.value(model.objective) + 1


def main():
    # solve(
    #     # num_producers=2,
    #     # num_consumers=2,
    #     # producer_time=3,
    #     # consumer_time=3,
    #     # producer_output_size=2,
    #     # consumer_input_size=2,
    #     # time_limit=5,
    #     # num_execution_slots=1,
    #     # buffer_size_limit=2,
    # )
    solve(
        num_producers=5,
        num_consumers=5,
        producer_time=1,
        consumer_time=2,
        time_limit=10,
        num_execution_slots=4,
        buffer_size_limit=4,
    )


if __name__ == "__main__":
    main()

# TODO: schedule looks wrong for p=1, c=2
