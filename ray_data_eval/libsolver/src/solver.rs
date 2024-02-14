pub fn solve() {
    let mut stack = Vec::new();
    let mut best_solution: Option<Solution> = None;
    stack.push(Environment::new());
    while let Some(state) = stack.pop() {
        if let Some(solution) = state.get_solution() {
            if let Some(best) = &best_solution {
                if solution.total_time < best.total_time {
                    best_solution = Some(solution);
                }
            } else {
                best_solution = Some(solution);
            }
        } else {
            let mut next_states = state.get_next_states();
            println!("Next states: {:?}", next_states);
            stack.append(&mut next_states);
        }
        println!("Stack size: {}", stack.len());
    }
    println!("Best solution: {:?}", best_solution);
}

#[derive(Debug)]
struct Solution {
    total_time: u32,
}

#[derive(Debug, Clone)]
struct Environment {
    tick: u32,
}

impl Environment {
    fn new() -> Self {
        Self { tick: 0 }
    }

    fn is_finished(&self) -> bool {
        self.tick >= 5
    }

    fn get_solution(&self) -> Option<Solution> {
        if !self.is_finished() {
            None
        } else {
            Some(Solution {
                total_time: self.tick,
            })
        }
    }

    fn get_next_states(&self) -> Vec<Self> {
        if self.is_finished() {
            vec![]
        } else {
            let mut next_states = Vec::new();
            for i in 1..=3 {
                let mut next = self.clone();
                next.tick += i;
                next_states.push(next);
            }
            next_states
        }
    }
}
