mod environment;

use crate::solver::environment::*;
use crate::types::*;
use log::info;

pub fn solve(problem: &SchedulingProblem) {
    info!("Solving problem: {}", problem.name);
    let mut heap = std::collections::BinaryHeap::new();
    let mut best_solution: Option<Solution> = None;
    let mut best_solution_set = Vec::new();

    fn update_best_solution(
        best_solution: &mut Option<Solution>,
        best_solution_set: &mut Vec<Solution>,
        solution: Solution,
    ) {
        info!("New best solution: {}", solution.total_time);
        solution.state.print();
        best_solution_set.clear();
        best_solution_set.push(solution.clone());
        *best_solution = Some(solution);
    }

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
        // state.print();
        let solution_lower_bound = state.get_solution_lower_bound();
        if let Some(best) = &best_solution {
            if best.total_time < solution_lower_bound {
                continue;
            }
        }
        if let Some(solution) = state.get_solution() {
            if let Some(best) = &best_solution {
                if solution.total_time < best.total_time {
                    update_best_solution(&mut best_solution, &mut best_solution_set, solution);
                } else if solution.total_time == best.total_time {
                    best_solution_set.push(solution);
                }
            } else {
                update_best_solution(&mut best_solution, &mut best_solution_set, solution);
            }
        } else {
            let next_states = state.get_next_states();
            heap.extend(next_states);
        }
    }
    info!("Total unique states visited: {}", visited.len());
    if let Some(solution) = best_solution {
        info!("Best solution: {:?}", solution.total_time);
        solution.state.print();
        info!(
            "Number of equivalent solutions: {}",
            best_solution_set.len()
        );
        // for solution in best_solution_set {
        //     solution.state.print();
        // }
    } else {
        info!("No solution found");
    }
}
