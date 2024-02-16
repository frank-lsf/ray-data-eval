mod solver;
mod types;

use pyo3::prelude::*;

fn init_logging() {
    env_logger::Builder::from_default_env()
        .format_timestamp(Some(env_logger::fmt::TimestampPrecision::Millis))
        .init();
}

#[pyfunction]
fn solve(problem: types::SchedulingProblem) -> PyResult<()> {
    init_logging();
    solver::solve(&problem);
    Ok(())
}

#[pymodule]
fn libsolver(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
