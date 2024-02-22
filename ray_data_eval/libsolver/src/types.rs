use pyo3::prelude::*;

// --- Problem definitions ---

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Resource {
    CPU,
    GPU,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, FromPyObject)]
pub struct ResourcesSpec {
    pub cpu: i32,
    pub gpu: i32,
    pub num_executors: i32,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, FromPyObject)]
pub struct TaskSpec {
    pub id: String,
    pub operator_idx: usize,
    pub duration: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub resources: ResourcesSpec,
}

#[derive(Debug, Clone, FromPyObject)]
pub struct OperatorSpec {
    pub name: String,
    pub operator_idx: usize,
    pub num_tasks: usize,
    pub duration: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub resources: ResourcesSpec,
    pub tasks: Vec<TaskSpec>,
}

impl OperatorSpec {
    pub fn new(
        name: String,
        operator_idx: usize,
        num_tasks: usize,
        duration: usize,
        input_size: usize,
        output_size: usize,
        resources: ResourcesSpec,
    ) -> Self {
        let tasks = (0..num_tasks)
            .map(|i| TaskSpec {
                id: name.clone(),
                operator_idx,
                duration,
                input_size,
                output_size,
                resources: resources.clone(),
            })
            .collect();
        OperatorSpec {
            name,
            operator_idx,
            num_tasks,
            duration,
            input_size,
            output_size,
            resources,
            tasks,
        }
    }
}

#[derive(Debug, Clone, FromPyObject)]
pub struct SchedulingProblem {
    pub operators: Vec<OperatorSpec>,
    pub name: String,
    pub resources: ResourcesSpec,
    pub time_limit: u32,
    pub buffer_size_limit: usize,
    pub num_operators: i32,
    pub tasks: Vec<TaskSpec>,
    pub num_total_tasks: i32,
}

impl SchedulingProblem {
    pub fn new(
        name: String,
        resources: ResourcesSpec,
        time_limit: u32,
        buffer_size_limit: usize,
        operators: Vec<OperatorSpec>,
    ) -> Self {
        let num_operators = operators.len() as i32;
        let num_total_tasks = operators.iter().map(|o| o.num_tasks as i32).sum();
        let tasks = operators.iter().flat_map(|o| o.tasks.clone()).collect();
        SchedulingProblem {
            name,
            resources,
            time_limit,
            buffer_size_limit,
            operators,
            num_operators,
            tasks,
            num_total_tasks,
        }
    }
}
