# Scheduling Problem Solver

## Usage

1. Install Rust (nightly build): https://rustup.rs/
2. `cargo install`
3. To install the Python package, use `maturin develop`
4. To turn on all optimizations, use `maturin develop --release`
5. To benchmark Rust code, use `cargo flamegraph` (will run `main.rs`)

## Algorithm

This solver implements a single-threaded search algorithm. It models the execution environment in discrete time steps (ticks), and models the states of each CPU/GPU executor, and a memory buffer between each pair of operators. Any environment state can produce a set of descendent states by enumerating all possible actions that a scheduling policy can take. The search algorithm explores all possible states, with the objective to find an execution trace that minimizes the total completion time of all operators.

Search optimizations:
- Best-first search: we maintain a priority queue of states, and always explore the state at the top of the queue. The sorting order is defined by the total number of tasks completed, i.e. we prefer states in which more tasks are completed (rather than those with cores idling, making no progress). This is such that we can reach a solution first, then aggressively prune out the "hopeless" states.
- Solution lower bounds. When exploring a state, we will compute the lower bound of the solution, given the current state. This is a strict lower bound, implemented in `environment.rs::Environment::get_solution_lower_bound()`. If the solution lower bound is greater than the currently known best answer, then this state is "hopeless", and we prune it from the search tree.
- Canonical orders. We define canonical orders for executors and tasks. i.e. The first task always starts on CPU 1, and an operator always starts its task #1. In other words, all executors and tasks are symmetric.
- Caching visited states. We maintain a hash set of visited states so that we do not explore the same state twice.
- State equivalence. We define two states to be equivalent as long as their _current_ states are the same, regardless of how they arrived at this state. This is a Markovian definition that helps us bring down the complexity of the problem from exponential to polynomial (need proof). Specifically, two states are equivalent if:
  - All executors are in the same state. This is defined as if two executors are executing the same task, started at the same tick.
  - All buffers are in the same state, defined as if they have the same amount of data items in use.
  - All operators are in the same state, defined as if they have the equal number of tasks completed.
