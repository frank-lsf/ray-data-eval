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
    time_limit: int = 2,
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
        "s",
        [(i, j, t) for i in range(num_total_tasks) for j in range(num_execution_slots) for t in range(time_limit)],
        cat="Binary",
    )
    buffer = pl.LpVariable.dicts("b", range(time_limit + 1), lowBound=0, cat="Integer")

    # Constraint: One CPU slot can run only one task at a time
    for j in range(num_execution_slots):
        for t in range(time_limit):
            model += pl.lpSum([schedule[(i, j, t)] for i in range(num_total_tasks)]) <= 1

    # Constraint: All tasks are assigned to exactly one CPU slot and complete
    for i in range(num_total_tasks):
        model += (
            pl.lpSum([schedule[(i, j, t)] for j in range(num_execution_slots) for t in range(time_limit)])
            == task_time[i]
        )

    # Constraint: All tasks must run contiguously
    for i in range(num_total_tasks):
        for j in range(num_execution_slots):
            for t in range(time_limit - task_time[i] + 1):
                model += (
                    pl.lpSum([schedule[(i, j, t + k)] for k in range(task_time[i])])
                    <= task_time[i] * schedule[(i, j, t)]
                )

    # Constraint: Buffer size is the total size of producer output not yet consumed
    for t in range(time_limit):
        buffer_increase = pl.lpSum(
            [
                producer_output_size[i]
                * pl.lpSum([schedule[(i, j, t - k)] for k in range(producer_time[i])])
                / producer_time[i]
                for i in range(num_producers)
                for j in range(num_execution_slots)
                if t >= producer_time[i]
            ]
        )
        buffer_decrease = pl.lpSum(
            [
                consumer_input_size[i]
                * pl.lpSum([schedule[(i + num_producers, j, t - k)] for k in range(consumer_time[i])])
                / consumer_time[i]
                for i in range(num_consumers)
                for j in range(num_execution_slots)
                if t >= consumer_time[i]
            ]
        )
        model += buffer[t + 1] == buffer[t] + buffer_increase - buffer_decrease

    # Constraint: Buffer size is bounded
    for t in range(time_limit):
        model += buffer[t] <= buffer_size_limit

    # Constraint: Buffer must be empty at the end of the time limit
    model += buffer[time_limit] == 0

    # Objective function: Minimize the latest finish time
    latest_finish_time = pl.LpVariable("latest_finish_time", lowBound=0, cat="Integer")
    for i in range(num_total_tasks):
        for j in range(num_execution_slots):
            for t in range(time_limit):
                model += latest_finish_time >= t * schedule[(i, j, t)] + 1

    model += latest_finish_time

    # Solve the problem
    model.solve()

    # Output results
    print("Status:", pl.LpStatus[model.status])
    print("Latest Finish Time =", pl.value(model.objective))
    print("Schedule:")
    for i in range(num_producers):
        for j in range(num_execution_slots):
            for t in range(time_limit):
                if pl.value(schedule[(i, j, t)]) == 1:
                    print(f"Producer {i}: {t} - {t + producer_time[i]}")
    for i in range(num_consumers):
        for j in range(num_execution_slots):
            for t in range(time_limit):
                if pl.value(schedule[(i + num_producers, j, t)]) == 1:
                    print(f"Consumer {i}: {t} - {t + consumer_time[i]}")

    # Draw ASCII art of the schedule where each row is a CPU slot and each column is a time step
    # Draw the table borders and timesteps
    print("+" + "-" * (time_limit * 5 + 5) + "+")
    for j in range(num_execution_slots):
        print(f"|| {j} || ", end="")
        for t in range(time_limit):
            idle = True
            for i in range(num_total_tasks):
                if pl.value(schedule[(i, j, t)]) == 1:
                    label = f"P{i}" if i < num_producers else f"C{i - num_producers}"
                    print(f"{label} | ", end="")
                    idle = False
            if idle:
                print("   | ", end="")
        print()
    print("+" + "-" * (time_limit * 5 + 5) + "+")
    print("|| t || ", end="")
    for t in range(time_limit):
        print(f"{t: 2} | ", end="")
    print()
    print("+" + "-" * (time_limit * 5 + 5) + "+")

    return pl.value(model.objective)


def main():
    solve()


if __name__ == "__main__":
    main()
