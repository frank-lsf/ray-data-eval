import pytest  # noqa: F401

from ray_data_eval.solver.solver import solve
from ray_data_eval.solver.config import SchedulingProblem


def _solve(**kwargs):
    return solve(SchedulingProblem(**kwargs))


def test_default():
    result = _solve()
    assert result == 2


def test_default_with_long_time_budget():
    result = _solve(time_limit=10)
    assert result == 2


def test_big_buffer():
    result = _solve(buffer_size_limit=10)
    assert result == 2


def test_2_cpu():
    result = _solve(num_execution_slots=2)
    assert result == 2


def test_long_schedule():
    result = _solve(num_producers=5, num_consumers=5, time_limit=10)
    assert result == 10


def test_long_producers():
    result = _solve(producer_time=2, time_limit=4)
    assert result == 3

    result = _solve(producer_time=3, time_limit=4)
    assert result == 4


def test_long_consumers():
    result = _solve(consumer_time=2, time_limit=4)
    assert result == 3

    result = _solve(consumer_time=3, time_limit=4)
    assert result == 4


def test_long_tasks():
    result = _solve(producer_time=2, consumer_time=2, time_limit=4)
    assert result == 4


def test_simple_1_cpu():
    result = _solve(num_producers=4, num_consumers=4, time_limit=10)
    assert result == 8


def test_simple_2_cpu():
    result = _solve(num_producers=4, num_consumers=4, time_limit=10, num_execution_slots=2)
    assert result == 5


def test_simple_4_cpu():
    result = _solve(num_producers=4, num_consumers=4, time_limit=10, num_execution_slots=3)
    assert result == 5


def test_producer_output_size():
    result = _solve(
        num_producers=1,
        num_consumers=2,
        producer_output_size=2,
        buffer_size_limit=2,
    )
    assert result == 3


def test_consumer_input_size():
    result = _solve(
        num_producers=2,
        num_consumers=1,
        consumer_input_size=2,
        buffer_size_limit=2,
    )
    assert result == 3


def test_long_case_1_cpu():
    result = _solve(
        num_producers=5,
        num_consumers=5,
        producer_time=1,
        consumer_time=2,
        time_limit=15,
        num_execution_slots=1,
    )
    assert result == 15


def test_long_case_2_cpu():
    result = _solve(
        num_producers=5,
        num_consumers=5,
        producer_time=1,
        consumer_time=2,
        time_limit=15,
        num_execution_slots=2,
    )
    assert result == 9


def test_long_case_3_cpu():
    result = _solve(
        num_producers=5,
        num_consumers=5,
        producer_time=1,
        consumer_time=2,
        time_limit=15,
        num_execution_slots=3,
    )
    assert result == 7


def test_long_case_4_cpu():
    result = _solve(
        num_producers=5,
        num_consumers=5,
        producer_time=1,
        consumer_time=2,
        time_limit=15,
        num_execution_slots=4,
    )
    assert result == 7


def test_long_case_4_cpu_4_mem():
    result = _solve(
        num_producers=5,
        num_consumers=5,
        producer_time=1,
        consumer_time=2,
        time_limit=15,
        num_execution_slots=4,
        buffer_size_limit=4,
    )
    assert result == 5
