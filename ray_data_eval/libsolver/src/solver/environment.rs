use std::{collections::HashMap, hash::Hash, sync::Arc};

use crate::types::*;
use itertools::Itertools;
use log::{debug, info};
use std::hash::Hasher;

type OperatorIndex = usize;
type TaskID = String;
type Tick = u32;

#[derive(Debug)]
pub struct Solution {
    pub total_time: u32,
    pub state: Environment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TaskStateType {
    Pending,
    Running,
    PendingOutput,
    Finished,
}

#[derive(Debug, Clone)]
struct TaskState {
    pub state: TaskStateType,
    pub started_at: Option<Tick>,
    pub execution_started_at: Option<Tick>,
    pub execution_finished_at: Option<Tick>,
    pub finished_at: Option<Tick>,
}

impl TaskState {
    pub fn new() -> Self {
        Self {
            state: TaskStateType::Pending,
            started_at: None,
            execution_started_at: None,
            execution_finished_at: None,
            finished_at: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OperatorState {
    num_tasks: usize,
    pub next_task_idx: usize,
}

impl OperatorState {
    pub fn new(num_tasks: usize) -> Self {
        Self {
            num_tasks,
            next_task_idx: 0,
        }
    }

    pub fn num_tasks_remaining(&self) -> usize {
        self.num_tasks - self.next_task_idx
    }

    pub fn is_finished(&self) -> bool {
        self.num_tasks_remaining() == 0
    }
}

#[derive(Clone)]
enum Action {
    Noop,
    StartTask { operator_idx: OperatorIndex },
}

impl std::fmt::Debug for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Noop => write!(f, "Noop"),
            Action::StartTask { operator_idx } => write!(f, "StartTask({})", operator_idx),
        }
    }
}

type ActionSet = Vec<Action>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RunningTask {
    spec: TaskSpec,
    started_at: Tick,
    remaining_ticks: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Executor {
    pub name: String,
    pub resource: Resource,
    running_task: Option<RunningTask>,
    timeline: Vec<String>,
}

impl Hash for Executor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.running_task.hash(state);
        self.timeline.hash(state);
    }
}

impl Executor {
    pub fn new(name: String, resource: Resource) -> Self {
        Self {
            name,
            resource,
            running_task: None,
            timeline: Vec::new(),
        }
    }

    pub fn tick(&mut self, tick: Tick, task_states: &mut HashMap<String, TaskState>) {
        self.timeline.push(self.get_timeline_item());
        if let Some(task) = &mut self.running_task {
            task.remaining_ticks -= 1;
            if task.remaining_ticks <= 0 {
                // TODO: check for buffer
                let can_finish = true; // self.try_finishing_running_task();
                if can_finish {
                    update_task_state(tick, task_states, &task.spec.id, TaskStateType::Finished);
                    self.finish_running_task();
                } else {
                    update_task_state(
                        tick,
                        task_states,
                        &task.spec.id,
                        TaskStateType::PendingOutput,
                    );
                }
            }
        }
    }

    pub fn can_start_task(&self, task: &TaskSpec) -> bool {
        if task.resources.cpu > 0 && self.resource != Resource::CPU {
            false
        } else if task.resources.gpu > 0 && self.resource != Resource::GPU {
            false
        } else if let Some(_) = &self.running_task {
            false
        } else {
            true
        }
    }

    pub fn start_task(&mut self, task: &TaskSpec, tick: Tick) -> bool {
        if !self.can_start_task(task) {
            false
        } else {
            self.running_task = Some(RunningTask {
                spec: task.clone(),
                started_at: tick,
                remaining_ticks: task.duration,
            });
            true
        }
    }

    fn get_timeline_item(&self) -> String {
        if let Some(task) = &self.running_task {
            if task.remaining_ticks > 0 {
                task.spec.id.clone()
            } else {
                "! ".to_string()
            }
        } else {
            "  ".to_string()
        }
    }

    fn finish_running_task(&mut self) -> RunningTask {
        self.running_task.take().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct Environment {
    executors: Vec<Executor>,
    operator_specs: Arc<Vec<OperatorSpec>>,
    operator_states: Vec<OperatorState>,
    task_states: HashMap<String, TaskState>,
    tick: Tick,
    time_limit: Tick,
}

impl Ord for Environment {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.tick == other.tick {
            self.num_tasks_finished().cmp(&other.num_tasks_finished())
        } else {
            self.tick.cmp(&other.tick)
        }
    }
}

impl PartialOrd for Environment {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Environment {
    fn eq(&self, other: &Self) -> bool {
        self.executors == other.executors
            && self.operator_states == other.operator_states
            && self.tick == other.tick
    }
}

impl Eq for Environment {}

impl Hash for Environment {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.executors.hash(state);
        self.operator_states.hash(state);
    }
}

fn update_task_state(
    tick: Tick,
    task_states: &mut HashMap<String, TaskState>,
    tid: &TaskID,
    state: TaskStateType,
) {
    let task_state = task_states.get_mut(tid).unwrap();
    task_state.state = state;
    match task_state.state {
        TaskStateType::Running => {
            task_state.started_at = Some(tick);
        }
        TaskStateType::PendingOutput => {
            task_state.execution_finished_at = Some(tick);
        }
        TaskStateType::Finished => {
            task_state.finished_at = Some(tick);
        }
        _ => {}
    }
}

fn get_action_set_combinations(
    num_slots: usize,
    operator_indexes: Vec<OperatorIndex>,
) -> Vec<ActionSet> {
    let mut choices = operator_indexes
        .iter()
        .map(|idx| Action::StartTask { operator_idx: *idx })
        .collect_vec();
    choices.push(Action::Noop);
    choices
        .into_iter()
        .combinations_with_replacement(num_slots)
        .collect_vec()
}

fn get_action_sets_by_resource(
    resource: Resource,
    executors: &Vec<Executor>,
    operator_specs: &Vec<OperatorSpec>,
    operator_states: &Vec<OperatorState>,
) -> Vec<ActionSet> {
    let filtered_operators = operator_specs
        .iter()
        .zip(operator_states.iter())
        .filter(|(op, state)| {
            (op.resources.cpu > 0 && resource == Resource::CPU
                || op.resources.gpu > 0 && resource == Resource::GPU)
                && !state.is_finished()
        })
        .map(|(spec, _)| spec.operator_idx)
        .collect::<Vec<_>>();
    let filtered_executors = executors
        .iter()
        .filter(|executor| executor.resource == resource && executor.running_task.is_none())
        .collect_vec();
    get_action_set_combinations(filtered_executors.len(), filtered_operators)
}

impl Environment {
    pub fn new(
        resources: &ResourcesSpec,
        operators: &Vec<OperatorSpec>,
        tasks: &Vec<TaskSpec>,
        time_limit: Tick,
    ) -> Self {
        Self {
            executors: (0..resources.cpu)
                .map(|i| Executor::new(format!("CPU{}", i), Resource::CPU))
                .chain(
                    (0..resources.gpu).map(|i| Executor::new(format!("GPU{}", i), Resource::GPU)),
                )
                .collect(),
            operator_specs: Arc::new(operators.clone()),
            operator_states: operators
                .iter()
                .map(|op| OperatorState::new(op.tasks.len()))
                .collect(),
            task_states: tasks
                .iter()
                .map(|t| (t.id.clone(), TaskState::new()))
                .collect(),
            tick: 0,
            time_limit,
        }
    }

    fn num_tasks_finished(&self) -> usize {
        self.task_states
            .values()
            .filter(|state| state.state == TaskStateType::Finished)
            .count()
    }

    fn is_finished(&self) -> bool {
        self.task_states
            .values()
            .all(|state| state.state == TaskStateType::Finished)
    }

    pub fn get_solution(&self) -> Option<Solution> {
        if !self.is_finished() {
            None
        } else {
            Some(Solution {
                total_time: self.tick,
                state: self.clone(),
            })
        }
    }

    pub fn get_solution_lower_bound(&self) -> Tick {
        // TODO(@lsf): group by resources; find max duration for each resource
        // will be an even better lower bound.
        let total_duration = self
            .operator_specs
            .iter()
            .zip(self.operator_states.iter())
            .map(|(spec, state)| state.num_tasks_remaining() * spec.duration)
            .sum::<usize>();
        let total_ticks_remaining =
            (total_duration as f32 / self.executors.len() as f32).ceil() as u32;
        self.tick + total_ticks_remaining
    }

    fn get_action_sets(&self) -> Vec<ActionSet> {
        let action_sets_list = [Resource::CPU, Resource::GPU].map(|res| {
            get_action_sets_by_resource(
                res,
                &self.executors,
                &self.operator_specs,
                &self.operator_states,
            )
        });
        let mut combinations = vec![vec![]]; // Initialize with an empty action set
        for action_set_list in action_sets_list.iter() {
            combinations = combinations
                .iter()
                .cartesian_product(action_set_list.iter())
                .map(|(a, b)| a.iter().chain(b.iter()).cloned().collect_vec())
                .collect_vec();
        }
        combinations
    }

    fn perform_action(&mut self, action: &Action) -> bool {
        match action {
            Action::Noop => true,
            Action::StartTask { operator_idx } => {
                let task = {
                    if let Some(operator_state) = self.operator_states.get_mut(*operator_idx) {
                        let task_idx = operator_state.next_task_idx;
                        operator_state.next_task_idx += 1;
                        if let Some(task) = self.operator_specs[*operator_idx].tasks.get(task_idx) {
                            Some(task.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                if let Some(task) = task {
                    self.start_task_on_any_executor(&task)
                } else {
                    false
                }
            }
        }
    }

    fn tick_with_actions(&mut self, action_set: &ActionSet) -> bool {
        for action in action_set {
            if !self.perform_action(&action) {
                return false;
            }
        }
        self.tick += 1;
        for executor in self.executors.iter_mut() {
            executor.tick(self.tick, &mut self.task_states);
        }
        true
    }

    pub fn get_next_states(&self) -> Vec<Self> {
        if self.is_finished() || self.tick >= self.time_limit {
            vec![]
        } else {
            let mut ret = Vec::new();
            let action_sets = self.get_action_sets();
            for action_set in action_sets {
                let mut state_ = self.clone();
                if state_.tick_with_actions(&action_set) {
                    debug!("Tick: {}, actions: {:?}", self.tick, action_set);
                    ret.push(state_);
                }
            }
            ret
        }
    }

    fn start_task_on_executor(&mut self, task: &TaskSpec, executor_idx: usize) -> bool {
        let executor = &mut self.executors[executor_idx];
        let started = executor.start_task(task, self.tick);
        if started {
            update_task_state(
                self.tick,
                &mut self.task_states,
                &task.id,
                TaskStateType::Running,
            );
            true
        } else {
            false
        }
    }

    fn start_task_on_any_executor(&mut self, task: &TaskSpec) -> bool {
        for i in 0..self.executors.len() {
            if self.start_task_on_executor(task, i) {
                return true;
            }
        }
        false
    }

    pub fn print(&self) {
        for executor in self.executors.iter() {
            info!("{:?}", executor.timeline);
        }
    }

    pub fn get_fingerprint(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}
