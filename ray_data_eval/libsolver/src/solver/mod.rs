mod environment;

use crate::solver::environment::*;
use crate::types::*;
use log::info;

pub fn solve(problem: &SchedulingProblem) {
    info!("Solving problem: {}", problem.name);
    let mut heap = std::collections::BinaryHeap::new();
    let mut best_solution: Option<Solution> = None;

    heap.push(Environment::new(
        &problem.resources,
        &problem.operators,
        &problem.tasks,
        problem.time_limit,
        problem.buffer_size_limit,
    ));
    let mut visited = std::collections::HashSet::new();
    while let Some(state) = heap.pop() {
        let fingerprint = state.get_fingerprint();
        if visited.contains(&fingerprint) {
            continue;
        }
        visited.insert(fingerprint);
        let solution_lower_bound = state.get_solution_lower_bound();
        if let Some(best) = &best_solution {
            if best.total_time < solution_lower_bound {
                continue;
            }
        }
        if let Some(total_time) = state.get_solution_tick() {
            if best_solution
                .as_ref()
                .map_or(true, |best| total_time < best.total_time)
            {
                info!("New best solution: {}", total_time);
                state.print();
                best_solution = Some(Solution { total_time, state });
            }
        } else {
            let next_states = state.get_next_states(&visited);
            heap.extend(next_states);
        }
    }
    info!("Total unique states visited: {}", visited.len());
    if let Some(solution) = best_solution {
        info!("Best solution: {:?}", solution.total_time);
        solution.state.print();
    } else {
        info!("No solution found");
    }
}
