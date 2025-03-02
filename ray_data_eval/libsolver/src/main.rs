use std::time::Instant;
mod solver;
mod types;

use types::*;

fn init_logging() {
    env_logger::Builder::from_default_env()
        .format_timestamp(Some(env_logger::fmt::TimestampPrecision::Millis))
        .init();
}

fn main() {
    init_logging();
    let test_problem = SchedulingProblem::new(
        "test_problem".to_string(),
        ResourcesSpec {
            cpu: 3,
            gpu: 0,
            num_executors: 3,
        },
        15,
        4,
        vec![
            OperatorSpec::new(
                "P".to_string(),
                0,
                10,
                1,
                0,
                1,
                ResourcesSpec {
                    cpu: 1,
                    gpu: 0,
                    num_executors: 1,
                },
            ),
            OperatorSpec::new(
                "C".to_string(),
                1,
                10,
                2,
                1,
                0,
                ResourcesSpec {
                    cpu: 1,
                    gpu: 0,
                    num_executors: 1,
                },
            ),
        ],
    );
    let num_producers = 3;
    let training_problem = SchedulingProblem::new(
        "training_problem".to_string(),
        ResourcesSpec {
            cpu: 8,
            gpu: 4,
            num_executors: 12,
        },
        300,
        30,
        vec![
            OperatorSpec::new(
                "P".to_string(),
                0,
                num_producers,
                8,
                0,
                8,
                ResourcesSpec {
                    cpu: 1,
                    gpu: 0,
                    num_executors: 1,
                },
            ),
            OperatorSpec::new(
                "I".to_string(),
                1,
                num_producers * 8,
                1,
                1,
                1,
                ResourcesSpec {
                    cpu: 0,
                    gpu: 1,
                    num_executors: 1,
                },
            ),
            OperatorSpec::new(
                "C".to_string(),
                2,
                num_producers * 4,
                2,
                2,
                0,
                ResourcesSpec {
                    cpu: 1,
                    gpu: 0,
                    num_executors: 1,
                },
            ),
        ],
    );
    let start = Instant::now();
    // solver::solve(&test_problem);
    solver::solve(&training_problem);
    let duration = start.elapsed();

    println!("Time elapsed: {:?}", duration);
}
