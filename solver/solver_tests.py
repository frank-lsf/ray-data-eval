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
