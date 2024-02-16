use std::{hash::Hash, sync::Arc};

use crate::types::*;
use itertools::Itertools;
use log::{debug, info};
use std::hash::Hasher;

type OperatorIndex = usize;
type Tick = u32;

#[derive(Debug)]
pub struct Solution {
    pub total_time: u32,
    pub state: Environment,
}

#[derive(Debug, Clone, Hash)]
pub struct Buffer {
    size: usize,
    consumable_size: usize,
    timeline: Vec<usize>,
    consumable_timeline: Vec<usize>,
}

impl Buffer {
    pub fn new() -> Self {
        Self {
            size: 0,
            consumable_size: 0,
            timeline: Vec::new(),
            consumable_timeline: Vec::new(),
        }
    }

    pub fn tick(&mut self, _tick: Tick) {
        self.timeline.push(self.size);
    }

    pub fn push(&mut self, size: usize) -> bool {
        self.size += size;
        self.consumable_size += size;
        true
    }

    pub fn consume(&mut self, size: usize) -> bool {
        if self.consumable_size >= size {
            self.consumable_size -= size;
            true
        } else {
            false
        }
    }

    pub fn pop(&mut self, size: usize) -> bool {
        if self.size >= size {
            self.size -= size;
            true
        } else {
            false
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

fn try_finishing_running_task(
    task: &RunningTask,
    input_buffer: Option<&mut Buffer>,
    output_buffer: Option<&mut Buffer>,
) -> bool {
    if let Some(input_buffer) = input_buffer {
        if !input_buffer.pop(task.spec.input_size) {
            return false;
        }
    }
    if let Some(output_buffer) = output_buffer {
        if !output_buffer.push(task.spec.output_size) {
            return false;
        }
    }
    true
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

    pub fn tick(&mut self, _tick: Tick, buffers: &mut Vec<Buffer>) -> i32 {
        self.timeline.push(self.get_timeline_item());
        if let Some(task) = &mut self.running_task {
            task.remaining_ticks -= 1;
            if task.remaining_ticks <= 0 {
                let operator_idx = task.spec.operator_idx;
                let (input_buffer, output_buffer) = {
                    let (left, right) = buffers.split_at_mut(operator_idx);
                    (left.last_mut(), right.first_mut())
                };
                let can_finish = try_finishing_running_task(task, input_buffer, output_buffer);
                if can_finish {
                    self.running_task = None;
                    1
                } else {
                    -1
                }
            } else {
                0
            }
        } else {
            0
        }
    }

    pub fn can_start_task(&self, task: &TaskSpec, input_buffer: &Option<&mut Buffer>) -> bool {
        if task.resources.cpu > 0 && self.resource != Resource::CPU {
            false
        } else if task.resources.gpu > 0 && self.resource != Resource::GPU {
            false
        } else if let Some(_) = &self.running_task {
            false
        } else if let Some(input_buffer) = input_buffer {
            input_buffer.consumable_size >= task.input_size
        } else {
            task.input_size == 0
        }
    }

    pub fn start_task(
        &mut self,
        task: &TaskSpec,
        tick: Tick,
        input_buffer: Option<&mut Buffer>,
    ) -> bool {
        if !self.can_start_task(task, &input_buffer) {
            false
        } else {
            self.running_task = Some(RunningTask {
                spec: task.clone(),
                started_at: tick,
                remaining_ticks: task.duration as i32,
            });
            if let Some(input_buffer) = input_buffer {
                if !input_buffer.consume(task.input_size) {
                    return false;
                }
            }
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
}

#[derive(Debug, Clone)]
pub struct Environment {
    executors: Vec<Executor>,
    buffers: Vec<Buffer>,
    operator_specs: Arc<Vec<OperatorSpec>>,
    operator_states: Vec<OperatorState>,
    tick: Tick,
    time_limit: Tick,
    buffer_size_limit: usize,
    num_tasks: usize,
    num_tasks_finished: usize,
}

impl Ord for Environment {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.num_tasks_finished == other.num_tasks_finished {
            self.tick.cmp(&other.tick)
        } else {
            self.num_tasks_finished.cmp(&other.num_tasks_finished)
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
        self.buffers.hash(state);
        self.operator_states.hash(state);
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
    let num_executors = executors
        .iter()
        .filter(|executor| executor.resource == resource && executor.running_task.is_none())
        .count();
    get_action_set_combinations(num_executors, filtered_operators)
}

impl Environment {
    pub fn new(
        resources: &ResourcesSpec,
        operators: &Vec<OperatorSpec>,
        tasks: &Vec<TaskSpec>,
        time_limit: Tick,
        buffer_size_limit: usize,
    ) -> Self {
        Self {
            executors: (0..resources.cpu)
                .map(|i| Executor::new(format!("CPU{}", i), Resource::CPU))
                .chain(
                    (0..resources.gpu).map(|i| Executor::new(format!("GPU{}", i), Resource::GPU)),
                )
                .collect(),
            buffers: operators.iter().map(|_| Buffer::new()).collect(),
            operator_specs: Arc::new(operators.clone()),
            operator_states: operators
                .iter()
                .map(|op| OperatorState::new(op.tasks.len()))
                .collect(),
            tick: 0,
            time_limit,
            buffer_size_limit,
            num_tasks: tasks.len(),
            num_tasks_finished: 0,
        }
    }

    fn is_finished(&self) -> bool {
        self.num_tasks_finished == self.num_tasks
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

    fn get_total_duration_of_pending_tasks_by_resource(&self, resource: &Resource) -> usize {
        self.operator_specs
            .iter()
            .zip(self.operator_states.iter())
            .filter(|(spec, _)| {
                spec.resources.cpu > 0 && *resource == Resource::CPU
                    || spec.resources.gpu > 0 && *resource == Resource::GPU
            })
            .map(|(spec, state)| state.num_tasks_remaining() * spec.duration)
            .sum()
    }

    fn get_total_ticks_remaining_by_resource(&self, resource: &Resource) -> u32 {
        let total_duration = self.get_total_duration_of_pending_tasks_by_resource(resource);
        let num_executors = self
            .executors
            .iter()
            .filter(|executor| executor.resource == *resource)
            .count();
        (total_duration as f32 / num_executors as f32).ceil() as u32
    }

    pub fn get_solution_lower_bound(&self) -> Tick {
        let ticks_remaining = [Resource::CPU, Resource::GPU]
            .map(|res| self.get_total_ticks_remaining_by_resource(&res))
            .iter()
            .fold(0, |a, &b| a.max(b));
        self.tick + ticks_remaining
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

    fn buffer_size_under_limit(&self) -> bool {
        self.buffers.iter().map(|buffer| buffer.size).sum::<usize>() <= self.buffer_size_limit
    }

    fn tick_with_actions(&mut self, action_set: &ActionSet) -> bool {
        for action in action_set {
            if !self.perform_action(&action) {
                return false;
            }
        }
        self.tick += 1;
        for i in 0..self.executors.len() {
            let result = self.executors[i].tick(self.tick, &mut self.buffers);
            if result > 0 {
                self.num_tasks_finished += 1;
            } else if result < 0 {
                return false;
            }
        }
        if !self.buffer_size_under_limit() {
            return false;
        }
        for buffer in self.buffers.iter_mut() {
            buffer.tick(self.tick);
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
        let input_buffer = if task.operator_idx == 0 {
            None
        } else {
            Some(&mut self.buffers[task.operator_idx - 1])
        };
        executor.start_task(task, self.tick, input_buffer)
    }

    fn check_task_input(&self, task: &TaskSpec) -> bool {
        if task.input_size == 0 || task.operator_idx == 0 {
            true
        } else {
            let buffer = &self.buffers[task.operator_idx - 1];
            buffer.consumable_size >= task.input_size
        }
    }

    fn start_task_on_any_executor(&mut self, task: &TaskSpec) -> bool {
        if !self.check_task_input(task) {
            return false;
        }
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
        for buffer in self.buffers.iter() {
            info!("{:?}", buffer.timeline);
        }
        info!("");
    }

    pub fn get_fingerprint(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}
