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
    pub duration: i32,
    pub input_size: i32,
    pub output_size: i32,
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

#[derive(Debug, Clone, FromPyObject)]
pub struct SchedulingProblem {
    pub operators: Vec<OperatorSpec>,
    pub name: String,
    pub resources: ResourcesSpec,
    pub time_limit: u32,
    pub buffer_size_limit: u32,
    pub num_operators: i32,
    pub tasks: Vec<TaskSpec>,
    pub num_total_tasks: i32,
}

// --- Solution definitions ---
