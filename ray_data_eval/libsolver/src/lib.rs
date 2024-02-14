mod solver;
mod types;

use pyo3::prelude::*;

#[pyfunction]
fn solve(problem: types::SchedulingProblem) -> PyResult<()> {
    println!("Solving problem: {}", problem.name);
    solver::solve();
    Ok(())
}

#[pymodule]
fn libsolver(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
