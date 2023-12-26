import pytest  # noqa: F401

from solver import solve


def test_default():
    result = solve()
    assert result == 2


def test_default_with_long_time_budget():
    result = solve(time_limit=10)
    assert result == 2


def test_big_buffer():
    result = solve(buffer_size_limit=10)
    assert result == 2


def test_2_cpu():
    result = solve(num_execution_slots=2)
    assert result == 1


def test_long_schedule():
    result = solve(num_producers=5, num_consumers=5, time_limit=10)
    assert result == 10


def test_long_producers():
    result = solve(producer_time=2, time_limit=4)
    assert result == 3


def test_long_consumers():
    result = solve(consumer_time=2, time_limit=4)
    assert result == 3


def test_long_tasks():
    result = solve(producer_time=2, consumer_time=2, time_limit=4)
    assert result == 4


def test_simple_1_cpu():
    result = solve(num_producers=4, num_consumers=4, time_limit=10)
    assert result == 8


def test_simple_2_cpu():
    result = solve(num_producers=4, num_consumers=4, time_limit=10, num_execution_slots=2)
    assert result == 4


def test_simple_3_cpu():
    result = solve(num_producers=4, num_consumers=4, time_limit=10, num_execution_slots=3)
    assert result == 3


def test_long_case_1_cpu():
    result = solve(
        num_producers=4,
        num_consumers=4,
        producer_time=1,
        consumer_time=2,
        time_limit=12,
        num_execution_slots=1,
    )
    assert result == 12
