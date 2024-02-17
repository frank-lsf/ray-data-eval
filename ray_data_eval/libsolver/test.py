import libsolver

from ray_data_eval.common.pipeline import test_problem, training_problem, long_training_problem
import time

start_time = time.time()
# libsolver.solve(test_problem)
libsolver.solve(long_training_problem)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
