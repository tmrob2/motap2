use std::collections::{HashSet, VecDeque, HashMap};
use itertools::Itertools;
use std::io::BufReader;
use std::path::Path;
use std::error::Error;
use std::fs::File;
use serde::Deserialize;
//use petgraph::data::ElementIterator;
use std::slice::Iter;
use std::iter::{FromIterator, Filter};
use regex::{Regex};
use petgraph::{Graph, graph::NodeIndex};
use std::num::ParseIntError;
use petgraph::algo::{kosaraju_scc, has_path_connecting, all_simple_paths};
use std::hash::Hash;
use ndarray::arr1;
use minilp::{Problem, OptimizationDirection, ComparisonOp, Variable, LinearExpr};
use std::env::var;

//use lazy_static::lazy_static;
extern crate serde_json;

#[derive(Debug, Deserialize)]
pub struct Target {
    pub target: Vec<f64>
}

#[derive(Debug, Deserialize)]
pub struct DRA {
    pub states: Vec<u32>,
    pub sigma: Vec<String>,
    pub safety: bool,
    pub initial: u32,
    pub delta: Vec<DRATransitions>,
    pub acc: Vec<AcceptanceSet>,
}

#[derive(Debug, Deserialize)]
pub struct DFA {
    pub states: Vec<u32>,
    pub sigma: Vec<String>,
    pub initial: u32,
    pub delta: Vec<DRATransitions>,
    pub acc: Vec<u32>,
    pub dead: Vec<u32>
}

#[derive(Debug, Clone)]
pub struct DRAMod {
    pub states: Vec<u32>,
    pub sigma: Vec<String>,
    pub safety: bool,
    pub initial: u32,
    pub delta: Vec<DRATransitions>,
    pub acc: Vec<AcceptanceSet>,
    pub dead: Vec<u32>
}

impl DRAMod {
    pub fn generate_graph(&mut self) -> Graph<String, String> {
        let mut graph: Graph<String, String> = Graph::new();

        for state in self.states.iter() {
            graph.add_node(format!("{}", state));
        }
        //println!("{:?}", graph.raw_nodes());
        for transition in self.delta.iter() {
            let origin_index = graph.node_indices().
                find(|x| graph[*x] == format!("{}", transition.q)).unwrap();
            let destination_index = match graph.node_indices().find(|x| graph[*x] == format!("{}", transition.q_prime)){
                None => {panic!("state: {} not found!", transition.q_prime)}
                Some(x) => {x}
            };
            graph.add_edge(origin_index, destination_index, format!("{:?}", transition.w.to_vec()));
        }
        graph
    }

    pub fn accepting_states(&self, g: &Graph<String, String>) -> Vec<NodeIndex> {
        let mut acc_indices: Vec<NodeIndex> = Vec::new();
        for state in self.acc.iter() {
            for k in state.k.iter() {
                let k_index = g.node_indices().
                    find(|x| g[*x] == format!("{}", k)).unwrap();
                acc_indices.push(k_index);
            }
        }
        acc_indices
    }

    pub fn reachable_from_states(&self, g: &Graph<String, String>, acc: &Vec<NodeIndex>) -> Vec<(u32, bool)> {
        let reachable: Vec<bool> = vec![true; self.states.len()];
        let mut reachable_states: Vec<(u32, bool)> = self.states.iter().cloned().zip(reachable.into_iter()).collect();
        for (state, truth) in reachable_states.iter_mut() {
            let from_node_index: NodeIndex = g.node_indices().find(|x| g[*x] == format!("{}", state)).unwrap();
            for k in acc.iter() {
                *truth = has_path_connecting(g, from_node_index, *k, None);
                //println!("Path from {} to {} is {}", g[from_node_index], g[*k], truth);
            }
        }
        reachable_states
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AcceptanceSet {
    pub l: Vec<u32>,
    pub k: Vec<u32>
}

#[derive(Debug, Deserialize, Clone)]
pub struct DRATransitions {
    q: u32,
    w: Vec<String>,
    q_prime: u32,
}

#[derive(Debug, Deserialize)]
pub struct MDP {
    pub states: Vec<u32>,
    pub initial: u32,
    pub transitions: Vec<Transition>,
    pub labelling: Vec<MDPLabellingPair>
}

#[derive(Debug, Deserialize)]
pub struct MDPLabellingPair {
    pub s: u32,
    pub w: String
}

#[derive(Debug, Deserialize)]
pub struct Transition {
    pub s: u32,
    pub a: String,
    pub s_prime: Vec<TransitionPair>,
    pub rewards: f64
}

#[derive(Debug, Deserialize)]
pub struct TransitionPair {
    pub s: u32,
    pub p: f64
}

#[derive(Debug,Clone,Eq, PartialEq, Hash)]
pub struct ModelCheckingPair {
    pub s: u32,
    pub q: Vec<u32>
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Copy)]
pub struct DFAModelCheckingPair {
    pub s: u32,
    pub q: u32,
}

#[derive(Debug)]
pub struct ProductTransition {
    pub sq: ModelCheckingPair,
    pub a: String,
    pub sq_prime: Vec<ModelCheckingPair>
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ProductLabellingPair {
    pub sq: ModelCheckingPair,
    pub w: Vec<String>
}

#[derive(Debug)]
pub struct ProductDRA {
    pub states: Vec<Vec<u32>>,
    pub sigma: Vec<String>,
    pub initial: Vec<u32>,
    pub delta: Vec<ProductDRATransitions>,
    pub acc: Vec<AcceptanceSet>,
}

#[derive(Deserialize,Debug)]
// Because this is a deterministic automata (q,w) -> q will always be true, and not (q,w) -> Vec<q>
pub struct ProductDRATransitions {
    q: Vec<u32>,
    w: Vec<String>,
    q_prime: Vec<u32>,
}

#[derive(Debug)]
pub struct TempProductDRATransitions {
    pub q: Vec<u32>,
    pub w: String,
    pub q_prime: Vec<u32>
}

impl ProductDRA {
    pub fn create_states(&mut self, dra: &DRAMod) {
        if self.states.is_empty() {
            // this is the first product DRA
            for q in dra.states.iter() {
                self.states.push(vec![*q]);
            }
        } else {
            let mut q_bar: Vec<Vec<u32>> = Vec::new();
            let it = self.states.clone().into_iter().cartesian_product(dra.states.clone().into_iter());
            for (vect, val) in it.into_iter(){
                let mut new_q: Vec<u32> = vect;
                new_q.push(val);
                q_bar.push(new_q)
            }
            self.states = q_bar;
        }
    }

    pub fn create_transitions(&mut self, dras: &Vec<DRAMod>) {

        //let re: Regex = Regex::new(r"\([0-9]\)").unwrap();
        let mut temp_transitions: Vec<TempProductDRATransitions> = Vec::new();

        for w in self.sigma.iter() {
            for state in self.states.iter() {
                let mut q_prime: Vec<u32> = vec![0; state.len()];
                for (i, q) in state.iter().enumerate() {
                    // if w contains an (k) for k in N then we need to replace the number by a symbolic k
                    // the last DRA could be a safety property, we can check this with the DRA safety variable

                    // In the product DRA we determine which task is active.
                    let new_word = w.to_string();
                    /*if dras[i].safety {
                        new_word = re.replace(w, "(k)").to_string();
                        //println!("w: {}, new word: {}", w, new_word);
                    }*/
                    for transition in dras[i].delta.iter().filter(|x| x.q == *q && x.w.contains(&new_word)){
                        q_prime[i] = transition.q_prime;
                    }
                }
                temp_transitions.push(TempProductDRATransitions {
                    q: state.clone(),
                    w: w.to_string(),
                    q_prime: q_prime
                });
            }
        }

        for q_bar in self.states.iter() {
            for q_bar_prime in self.states.iter() {
                let mut word_vect: Vec<String> = Vec::new();
                for transition in temp_transitions.iter().filter(|x| x.q == *q_bar && x.q_prime == *q_bar_prime) {
                    //println!("transition compression: {:?}", transition)
                    word_vect.push(transition.w.to_string());
                }
                if !word_vect.is_empty() {
                    self.delta.push(ProductDRATransitions {
                        q: q_bar.clone(),
                        w: word_vect,
                        q_prime: q_bar_prime.clone()
                    })
                }
            }
        }
    }

    pub fn task_count(dras: &Vec<DRA>) -> u32 {
        let mut count: u32 = 0;
        for dra in dras.iter() {
            if !dra.safety {
                count += 1;
            }
        }
        count
    }

    pub fn set_initial(&mut self, dras: &Vec<DRAMod>) {
        let mut init: Vec<u32> = vec![0; dras.len()];
        for (i, dra) in dras.iter().enumerate() {
            init[i] = dra.initial;
        }
        self.initial = init;
    }
    
    pub fn default() -> ProductDRA {
        ProductDRA {
            states: vec![],
            sigma: vec![],
            initial: vec![],
            delta: vec![],
            acc: vec![]
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq)]
struct RemoveSCCState {
    state: ModelCheckingPair,
    scc_ind: usize,
    state_ind: usize,
}

pub struct ProductMDP {
    pub states: Vec<ModelCheckingPair>,
    pub initial: ModelCheckingPair,
    pub transitions: Vec<ProductTransition>,
    pub labelling: Vec<ProductLabellingPair>
}

impl ProductMDP {

    /// Create states should only be used for the initial product creation
    pub fn create_states(&mut self, mdp: &MDP, dra: &ProductDRA) {
        self.states = Vec::new();
        let cp= mdp.states.clone().into_iter().cartesian_product(dra.states.clone().into_iter());
        for (s_m, q_a) in cp.into_iter() {
            self.states.push(ModelCheckingPair{ s: s_m, q: q_a.clone() });
        }
    }

    pub fn create_transitions(&mut self, mdp: &MDP, dra: &ProductDRA, verbose: &u32, dras: &Vec<DRAMod>, safety_present: &bool) {
        for state_from in self.states.iter() {
            if *verbose == 2 {
                println!("state from: {:?}", state_from);
            }
            // For each state we want to filter the mdp transitions to that state
            for t in mdp.transitions.iter().filter(|x| x.s == state_from.s) {
                // We then want to know, given some state in M, what state does it transition to?
                // there are exactly m DRAs currently added to the product (in the initial case this is 1)
                // and we want to know that given DRA (task j), in the loop signature above, we enumerate over q
                // for each task in the vector of DRAs minus the safety task, we want to determine what happens
                // when we transition from s -> a -> s' such that the trace will be L(s')
                // but the MDP wil have a non specific labelling, such as initiate(k)

                // Observe the state and determine which task is active, if the MDP state is 0 and the automata state is final
                // then any of the remaining tasks can be chosen. We must always be thinking that the MDP is projecting its states
                // onto some automata representing a task, but the MDP cannot switch from one task to another, it must continue with
                // some active task to completion

                // determine the active task
                let mut inactive_tasks: Vec<usize> = Vec::new();
                let mut active_task: bool = false;
                let mut active_tasks: Vec<usize> = Vec::new();
                let task_elements: &[u32] = if *safety_present {
                    state_from.q.split_last().unwrap().1
                } else {
                    &state_from.q[..]
                };
                let new_task = task_elements.iter().enumerate().
                    all(|(i, x)| dras[i].initial == *x
                        || dras[i].dead.iter().any(|y| y == x)
                        || dras[i].acc.iter().any(|y| y.k.iter().any(|z|z == x)));
                if new_task {
                    if task_elements.iter().enumerate().all(|(i, x)| dras[i].dead.iter().any(|y| y == x)
                        || dras[i].acc.iter().any(|y| y.k.iter().any(|z|z == x))) {
                        // every task has finished
                        if *verbose == 2 {
                            println!("Every task has finished at state: {:?}", task_elements);
                        }
                        for k in 0..task_elements.len(){
                            inactive_tasks.push(k + 1);
                        }
                    } else {
                        if *verbose == 2 {
                            println!("There are tasks remaining: {:?}", task_elements);
                        }
                        for (i,q) in task_elements.iter().enumerate() {
                            if dras[i].initial == *q {
                                if *verbose == 2 {
                                    println!("initiating task: {}", i + 1);
                                }
                                inactive_tasks.push(i + 1);
                            }
                        }
                    }
                } else {
                    active_task = true;
                    if *verbose == 2{
                        println!("There is an  active task: {:?}", task_elements);
                    }
                    for (i, q) in task_elements.iter().enumerate() {
                        if dras[i].initial != *q
                            && dras[i].dead.iter().all(|y| y != q)
                            && dras[i].acc.iter().all(|y| y.k.iter().all(|z|z != q)) {
                            active_tasks.push(i+1)
                        }
                    }
                }
                let mut task_queue: Vec<usize> = Vec::new();
                if active_task {
                    for task in active_tasks.into_iter() {
                        task_queue.push(task);
                    }
                } else {
                    for task in inactive_tasks.into_iter() {
                        task_queue.push(task);
                    }
                }
                if *verbose == 2 {
                    println!("task queue: {:?}", task_queue);
                }
                for task in task_queue.iter() {
                    let mut p_transition = ProductTransition {
                        sq: ModelCheckingPair { s: state_from.s, q: state_from.q.clone() },
                        a: format!("{}{}", t.a.to_string(), task).to_string(),
                        sq_prime: Vec::new()
                    };
                    for mdp_sprime in t.s_prime.iter() {
                        // label may be empty, if empty, then we should treat it as no label
                        // get the label specific to a task if it meets the appropriate pattern
                        let mut q_prime: Vec<u32> = Vec::new();
                        for lab in mdp.labelling.iter().filter(|l| l.s == mdp_sprime.s) {
                            // task specific labels will be required to have the form ...(k) where the label is specific to
                            // task k. Obviously the DRA will not accept this, and we will require that (k) be replaced by
                            // the enumerated task automaton e.g. (1) for automaton 0.
                            let mut ltemp: String = lab.w.clone();
                            if lab.w.contains("(k)") {
                                let substring = format!("({})", task);
                                ltemp = lab.w.to_string().replace(&*"(k)".to_string(), &*substring);
                                if *verbose == 2 {
                                    println!("replaced {}, with {}", lab.w, ltemp);
                                }
                            }
                            // find if there is a DRA transition. If no transition exists for the labelling is this an error? Let's print some
                            // of the results to find out.

                            for dra_transition in dra.delta.iter().
                                filter(|qq| qq.q == state_from.q && qq.w.contains(&ltemp)) {
                                q_prime = dra_transition.q_prime.clone();
                            }
                            if q_prime.is_empty() {
                                if *verbose == 2 {
                                    println!("No transition was found for (q: {:?}, w: {})", state_from.q, ltemp);
                                    println!("Available transitions are: ");
                                    for dra_transition in dra.delta.iter().
                                        filter(|qq| qq.q == state_from.q) {
                                        println!("{:?}", dra_transition);
                                    }
                                }
                            }
                            self.labelling.push(ProductLabellingPair { sq: ModelCheckingPair { s: mdp_sprime.s, q: q_prime.clone() }, w: vec![ltemp] });
                        }
                        p_transition.sq_prime.push(ModelCheckingPair{ s: mdp_sprime.s, q: q_prime });
                    }
                    self.transitions.push(p_transition);
                }
            }
        }
    }

    /// Adds in the task k labelling update i.e. {init, success, fail}
    ///
    /// ** Note
    /// This function is a new addition and will always be somewhat redundant
    /// because we can do this in the create transitions or create states function can minimise
    /// the required loops over states by one.
    pub fn update_prod_labelling(&mut self, dras: &Vec<DRAMod>, safety_present: &bool) {
        for state in self.states.iter() {
            let mut state_label_found: bool = false;
            let mut state_label: Vec<String> = Vec::new();
            let mut label_index: Option<usize> = None;
            for (i, l) in self.labelling.iter().enumerate().filter(|(_ii,x)| x.sq == *state) {
                state_label_found = true;
                state_label = l.w.to_vec();
                label_index = Some(i);
            }
            let q_bar: &[u32] = if *safety_present {
                state.q.split_last().unwrap().1
            } else {
                &state.q[..]
            };

            for (i, q) in q_bar.iter().enumerate() {
                if dras[i].dead.iter().any(|y| y == q) {
                    // label as fail (i)
                    state_label.push(format!("fail({})", i))

                } else if dras[i].acc.iter().any(|y| y.k.iter().any(|z|z == q)) {
                    // label as success (i)
                    state_label.push(format!("success({})",i))
                }
            }

            if state_label_found {
                self.labelling[label_index.unwrap()].w = state_label;
            } else {
                self.labelling.push(ProductLabellingPair{ sq: ModelCheckingPair { s: state.s, q: state.q.to_vec() }, w: state_label });
            }
        }
    }

    pub fn set_initial(&mut self, mdp: &MDP, dra: &ProductDRA) {
        self.initial = ModelCheckingPair { s: mdp.initial, q: dra.initial.clone() };
    }

    /// Traversing from the initial state, prune unreachable states.
    pub fn prune(&mut self, verbose: &u32) {
        let mut queue: VecDeque<ModelCheckingPair> = VecDeque::new();
        let mut visited: Vec<bool> = vec![false; self.states.len()];
        let mut visited_states: Vec<ModelCheckingPair> = Vec::new();
        let mut visited_transitions: Vec<ProductTransition> = Vec::new();
        let mut visited_labels: Vec<ProductLabellingPair> = Vec::new();
        queue.push_front(ModelCheckingPair{ s: self.initial.s, q: self.initial.q.clone() });
        let index_visited: usize = self.states.iter().
            position(|x| x.s == self.initial.s && x.q == self.initial.q.clone()).unwrap();
        visited[index_visited] = true;
        while !queue.is_empty() {
            if *verbose == 2 {
                println!("queue: {:?}", queue);
            }
            let next_state = queue.pop_front().unwrap();
            visited_states.push(ModelCheckingPair{ s: next_state.s, q: next_state.q.clone() });
            for label in self.labelling.iter().
                filter(|x| x.sq.s == next_state.s && x.sq.q == next_state.q) {
                visited_labels.push(ProductLabellingPair {
                    sq: ModelCheckingPair { s: label.sq.s, q: label.sq.q.clone() },
                    w: label.w.clone()
                });
            }
            for transition in self.transitions.iter().
                filter(|x| x.sq.s == next_state.s && x.sq.q == next_state.q) {
                for transition_to in transition.sq_prime.iter(){
                    let sqprime_index: usize = self.states.iter().
                        position(|xx| xx.s == transition_to.s && xx.q == transition_to.q).unwrap();
                    if !visited[sqprime_index] {
                        queue.push_front(ModelCheckingPair{ s: transition_to.s, q: transition_to.q.clone() });
                        visited[sqprime_index] = true;
                    }
                }
                visited_transitions.push(ProductTransition{
                    sq: ModelCheckingPair { s: transition.sq.s, q: transition.sq.q.clone() },
                    a: transition.a.to_string(),
                    sq_prime: transition.sq_prime.clone()
                });

            }
        }
        self.states = visited_states;
        self.transitions = visited_transitions;
        self.labelling = (HashSet::<_>::from_iter(visited_labels.iter().cloned())).into_iter().collect();

    }

    /// When taking the product of an already generated product MDP with a new DRA, which generates a new Product MDP set of states.
    pub fn add_states(&mut self, dra: &DRA) {
        let mut new_states: Vec<ModelCheckingPair> = Vec::new();
        let cp = self.states.clone().into_iter().cartesian_product(dra.states.clone().into_iter());
        for (old, new) in cp.into_iter() {
            let mut new_q: Vec<u32> = old.q.clone();
            new_q.push(new);
            new_states.push(ModelCheckingPair{ s: old.s, q: new_q })
        }
    }

    pub fn add_initial(&mut self, dra: &DRA) {
        let mut qnew = self.initial.q.clone();
        qnew.push(dra.initial);
        self.initial = ModelCheckingPair{s: self.initial.s, q: qnew};
    }

    pub fn generate_graph(&mut self) -> Graph<String, String> {
        let mut graph: Graph<String, String> = Graph::new();

        for state in self.states.iter() {
            graph.add_node(format!("({},{:?})", state.s, state.q));
        }
        //println!("{:?}", graph.raw_nodes());
        for transition in self.transitions.iter() {
            let origin_index = graph.node_indices().
                find(|x| graph[*x] == format!("({},{:?})", transition.sq.s, transition.sq.q)).unwrap();
            for sqprime in transition.sq_prime.iter() {
                let destination_index = match graph.node_indices().find(|x| graph[*x] == format!("({},{:?})", sqprime.s, sqprime.q)){
                    None => {panic!("state: ({},{:?}) not found!", sqprime.s, sqprime.q)}
                    Some(x) => {x}
                };
                graph.add_edge(origin_index, destination_index, transition.a.clone());
            }
        }
        graph
    }

    /// Function to identify the trivial MEC decomposition of the input graph.
    /// This is essentially any self loop.
    pub fn find_trivial_mecs(&self, mecs: &Vec<Vec<ModelCheckingPair>>) -> HashSet<ModelCheckingPair> {
        // Given a list of MECs we are essentially looking for a state wihtin the MEC which has a
        // self loop and greater than or equal to 2 actions.
        let mut triv_mecs: HashSet<ModelCheckingPair> = HashSet::new();
        for mec in mecs.iter() {
            for state in mec.iter() {
                for transition in self.transitions.iter().
                    filter(|x| x.sq_prime.len() == 1 && x.sq_prime.contains(&x.sq) && x.sq == *state) {
                    //println!("trivial transition: {:?}", transition);
                    triv_mecs.insert(ModelCheckingPair{ s: transition.sq.s, q: transition.sq.q.to_vec() });
                }
            }
        }
        triv_mecs
    }

    /// Function to identify the non-trivial MEC decomposition of the graph
    pub fn find_mecs(&self, g: &Graph<String, String>) -> Vec<Vec<ModelCheckingPair>> {
        let mut g_copy: Graph<String, String> = g.clone();
        //let mut test_counter: u32 = 0;
        // we also need to establish a correspondence between the states of the MDP and the nodes of the graph
        let mut state_actions: HashMap<ModelCheckingPair, HashSet<String>> = HashMap::new();
        for s in self.states.iter() {
            let mut action_vect: HashSet<String> = HashSet::new();
            for transition in self.transitions.iter().filter(|x| x.sq == *s) {
                action_vect.insert(transition.a.to_string());
            }
            state_actions.insert(s.clone(), action_vect);
        }

        let mut mec: Vec<Vec<ModelCheckingPair>> = Vec::new();
        let mut mec_new: Vec<Vec<ModelCheckingPair>> = vec![self.states.clone()];
        let mut removed_hist: Vec<ModelCheckingPair> = Vec::new();

        while mec != mec_new {
            mec = mec_new.drain(0..).collect();
            let mut remove: HashSet<RemoveSCCState> = HashSet::new();
            let scc_ni: Vec<Vec<petgraph::prelude::NodeIndex>> = kosaraju_scc(&g_copy);
            let mut scc_state_ind: Vec<Vec<ModelCheckingPair>> = Vec::new();
            for scc in scc_ni.iter() {
                let mut inner: Vec<ModelCheckingPair> = Vec::new();
                for ni in scc.iter() {
                    let str_state = &g_copy[*ni];
                    match self.convert_string_node_to_state(str_state) {
                        Some(x) => inner.push(x),
                        None => {println!("Error on finding node with state: {}", str_state); return mec_new }
                    }
                }
                scc_state_ind.push(inner);
            }
            for (i, t_k) in scc_state_ind.iter().enumerate() {
                for (j, state) in t_k.iter().enumerate() {
                    let mut action_s: HashSet<String> = state_actions.get(&state).unwrap().clone();
                    //println!("actions: {:?}", action_s);
                    for transition in self.transitions.iter().filter(|x| x.sq == *state) {
                        for sprime in transition.sq_prime.iter() {
                            //println!("s': {:?}", sprime);
                            if !t_k.iter().any(|x| x == sprime) {
                                // is there is a state such that taking action 'a' leaves the SCC with P > 0 then
                                // we remove this action from the SCC and it is not contained in the EC
                                action_s.remove(&*transition.a);
                            }
                        }
                    }
                    if action_s.is_empty() {
                        // remove the state
                        remove.insert(RemoveSCCState {
                            state: ModelCheckingPair { s: state.s, q: state.q.clone() },
                            scc_ind: i,
                            state_ind: j
                        });
                    }
                    state_actions.insert(state.clone(), action_s);
                }
                //println!("T({}), SCC: {:?}, R: {:?}",i, t_k, remove);
            }

            while !remove.is_empty() {
                //println!("remove: {:?} ; remove hist: {:?}", remove, removed_hist);
                let mut remove_more: HashSet<RemoveSCCState> = HashSet::new();
                let mut remove_actions: HashMap<RemoveSCCState, String> = HashMap::new();
                for r in remove.drain() {
                    if !scc_state_ind[r.scc_ind].is_empty() {
                        //println!("looking to remove: {:?}", r);
                        if !removed_hist.iter().any(|x| x == &r.state) {
                            let delete_index = g_copy.node_indices().find(|i| g_copy[*i] == format!("({},{:?})", r.state.s, r.state.q)).unwrap();
                            g_copy.remove_node(delete_index);
                        }

                        // remove the state from the SCCs
                        //println!("Removal History: {:?}", removed_hist);
                        for transition in self.transitions.iter().filter(|x| x.sq_prime.iter().any(|y| *y == r.state) && r.state != x.sq && removed_hist.iter().all(|y| *y != x.sq)){
                            let mut remove_index: Vec<(usize, usize)> = Vec::new();
                            for (k, v) in scc_state_ind.iter().enumerate() {
                                //println!("sccs({}):{:?}, transition state: {:?}", k, v, transition.sq);
                                for (j, _state) in v.iter().filter(|x| **x == transition.sq).enumerate() {
                                    remove_index.push((k,j));
                                }
                            }
                            // there is a problem when the action is a part of a self loop, we are not handling this correctly
                            remove_actions.insert(RemoveSCCState {
                                state: ModelCheckingPair {s: transition.sq.s, q: transition.sq.q.clone()},
                                scc_ind: remove_index[0].0,
                                state_ind: remove_index[0].1,
                            }, transition.a.to_string());
                            //println!("Removal Actions: {:?}", remove_actions);
                        }
                        //println!("scc states: {:?}; total sccs: {:?}", scc_state_ind[r.scc_ind], scc_state_ind);
                        scc_state_ind[r.scc_ind].remove(r.state_ind);
                        removed_hist.push(r.state);
                        //println!("state-actions pairs under review: {:?}", remove_actions);
                    }
                }
                for (state, action) in remove_actions.iter() {
                    let mut actions = state_actions.get(&state.state).unwrap().clone();
                    //println!("actions: {:?}", actions);
                    if actions.is_empty() {
                        //println!("History: {:?}", removed_hist);
                        //println!("Found: {:?} to remove", state.state);
                        remove_more.insert(RemoveSCCState{state: state.state.clone(), scc_ind: state.scc_ind, state_ind: state.state_ind});
                    } else {
                        actions.remove(action);
                        // update the state actions to remove the actions involved
                        state_actions.insert(state.state.clone(), actions);
                    }
                }
                //println!("Remove updated: {:?}", remove_more);
                remove = remove_more.drain().collect();
            }
            // we have to remove the state from the sccs, before we can update what the new MECs are
            //let scc_test: Vec<Vec<petgraph::prelude::NodeIndex>> = kosaraju_scc(&g_copy);
            //let mut count: u32 = 0;
            //for ni_vect in scc_test.iter() {
                //for ni in ni_vect.iter() {
                    //let i = ni;
                    //println!("{}:{:?}", count, g_copy[*i]);
                //}
                //count += 1;
            //}
            /*if test_counter >= 4 {
                return mec_new
            }*/
            //test_counter += 1;
            mec_new = scc_state_ind.clone();
            //println!("Removal history: {:?}", removed_hist);
            //println!("Test counter: {}", test_counter);
        }
        mec_new
    }

    pub fn convert_string_node_to_state(&self, s: &str) -> Option<ModelCheckingPair> {
        //println!("state: {}", s);
        let re: Regex = Regex::new(r"^\((?P<input1>[0-9]+),(?P<input2>\[[0-9]+(?:,\s[0-9]+)*\])\)").unwrap();
        let input1 = re.captures(s).and_then(|x| {
            x.name("input1").map(|y| y.as_str())
        });
        let input2 = re.captures(s).and_then(|x| {
            x.name("input2").map(|y| y.as_str())
        });
        //println!("input1: {:?}, input2: {:?}", input1, input2);
        let return_state: Option<ModelCheckingPair> = match parse_int(input1.unwrap()) {
            Ok(x) =>
                {
                     match parse_str_vect(input2.unwrap()) {
                        Ok(y) => Some(ModelCheckingPair {s: x, q: y }),
                        _ => None
                    }
                },
            _ => {
                println!("while parsing graph node: {}, received error", s, );
                None
            }
        };
        return_state
    }

    pub fn find_state_index(&self, sq: &ModelCheckingPair) -> usize {
        self.states.iter().position(|x| x.s == sq.s && x.q == sq.q).unwrap()
    }

    /// Create a default value
    pub fn default() -> ProductMDP {
        ProductMDP{
            states: vec![],
            initial: ModelCheckingPair { s: 0, q: vec![] },
            transitions: vec![],
            labelling: vec![]
        }
    }
}

#[derive(Debug, Clone)]
pub struct DFAProductTransition {
    pub sq: DFAModelCheckingPair,
    pub a: String,
    pub sq_prime: Vec<DFATransitionPair>,
    pub reward: f64
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct DFAProductLabellingPair {
    pub sq: DFAModelCheckingPair,
    pub w: Vec<String>
}

/// A DFA transition pair is a state and a probability of transitioning to this state
#[derive(Debug, Clone, PartialEq)]
pub struct DFATransitionPair {
    pub state: DFAModelCheckingPair,
    pub p: f64
}

#[derive(Clone, Debug)]
pub struct DFAProductMDP {
    pub states: Vec<DFAModelCheckingPair>,
    pub initial: DFAModelCheckingPair,
    pub transitions: Vec<DFAProductTransition>,
    pub labelling: Vec<DFAProductLabellingPair>,
}

impl DFAProductMDP {
    pub fn create_states(&mut self, mdp: &MDP, dfa: &DFA) {
        self.states = Vec::new();
        let cp= mdp.states.clone().into_iter().cartesian_product(dfa.states.clone().into_iter());
        for (s_m, q_a) in cp.into_iter() {
            self.states.push(DFAModelCheckingPair{ s: s_m, q: q_a});
        }
    }

    pub fn create_labelling(&mut self, mdp: &MDP) {
        for state in self.states.iter() {
            for mdp_label in mdp.labelling.iter().filter(|x| x.s == state.s) {
                self.labelling.push(DFAProductLabellingPair{
                    sq: DFAModelCheckingPair { s: state.s, q: state.q },
                    w: vec![mdp_label.w.to_string()]
                });
            }
        }
        /*for label in self.labelling.iter() {
            println!("debug label: {:?}", label);
        }*/
    }

    pub fn create_transitions(&mut self, mdp: &MDP, dfa: &DFA) {
        for state in self.states.iter() {
            for transition in mdp.transitions.iter()
                .filter(|x| x.s == state.s) {
                let mut t = DFAProductTransition {
                    sq: DFAModelCheckingPair { s: state.s, q: state.q },
                    a: transition.a.to_string(),
                    sq_prime: vec![],
                    reward: 0.0
                };
                t.reward = transition.rewards; // this is the reward inherited from the MDP
                for sprime in transition.s_prime.iter() {
                    for lab in mdp.labelling.iter()
                        .filter(|l| l.s == sprime.s) {
                        for q_prime in dfa.delta.iter()
                            .filter(|q| q.q == state.q && q.w.iter().any(|xx| *xx == lab.w)) {
                            t.sq_prime.push(DFATransitionPair {
                                state: DFAModelCheckingPair {s: sprime.s, q: q_prime.q_prime},
                                p: sprime.p
                            })
                        }
                    }
                }
                self.transitions.push(t);
            }
        }
    }

    /// Prune does graph analysis to check if there is a path from an initial state to any other
    /// state. If the has_path returns ```false``` then the state is removed from the graph and
    /// subsequently the Product MDP (DFA).
    pub fn prune_candidates(&mut self, reachable_state: &Vec<(DFAModelCheckingPair, bool)>) -> (Vec<usize>,Vec<DFAModelCheckingPair>){
        let mut prune_state_indices: Vec<usize> = Vec::new();
        let mut prune_states: Vec<DFAModelCheckingPair> = Vec::new();
        for (state, _truth) in reachable_state.iter().
            filter(|(_x,t)| !*t) {
            let remove_index = self.states.iter().position(|y| y == state).unwrap();
            prune_state_indices.push(remove_index);
            prune_states.push(self.states[remove_index].clone())
        }
        // find the transitions relating to the prune states.
        (prune_state_indices, prune_states)
    }

    pub fn prune_states_transitions(&mut self, prune_state_indices: &Vec<usize>, prune_states: &Vec<DFAModelCheckingPair>) {
        let mut new_transitions: Vec<DFAProductTransition> = Vec::new();
        let prune_state_hash: HashSet<_> = HashSet::from_iter(prune_states.iter().cloned());
        for transition in self.transitions.iter().
            filter(|x| prune_states.iter().all(|y| *y != x.sq) && {
                let mut sq_prime: Vec<DFAModelCheckingPair> = vec![DFAModelCheckingPair{ s: 0, q: 0 }; x.sq_prime.len()];
                for (i, xx) in x.sq_prime.iter().enumerate() {
                    sq_prime[i] = xx.state.clone();
                }
                let sq_prime_hash: HashSet<DFAModelCheckingPair> = HashSet::from_iter(sq_prime.iter().cloned());
                let intersection: HashSet<_> = prune_state_hash.intersection(&sq_prime_hash).collect();
                intersection.is_empty()
            }){
            new_transitions.push(DFAProductTransition {
                sq: DFAModelCheckingPair { s: transition.sq.s, q: transition.sq.q },
                a: transition.a.to_string(),
                sq_prime: transition.sq_prime.to_vec(),
                reward: transition.reward
            });
        }

        self.transitions = new_transitions;
        let mut new_states: Vec<DFAModelCheckingPair> = Vec::new();
        for (i, state) in self.states.iter().enumerate() {
            let delete_truth = prune_state_indices.iter().any(|x| *x == i);
            if !delete_truth {
                new_states.push(DFAModelCheckingPair { s: state.s, q: state.q })
            }
        }
        self.states = new_states;
    }

    pub fn reachable_from_initial(&self, g: &Graph<String, String>) -> Vec<(DFAModelCheckingPair, bool)> {
        let initial: NodeIndex = g.node_indices().
            find(|x| g[*x] == format!("({},{})", self.initial.s, self.initial.q)).unwrap();
        let reachable: Vec<bool> = vec![true; self.states.len()];
        let mut reachable_states: Vec<(DFAModelCheckingPair, bool)> = self.states.iter().cloned().
            zip(reachable.into_iter()).collect();
        for (state, truth) in reachable_states.iter_mut() {
            let to_node_index: NodeIndex = g.node_indices().
                find(|x| g[*x] == format!("({},{})", state.s, state.q)).unwrap();
            *truth = has_path_connecting(g, initial, to_node_index, None);
            //println!("Path from {} to {} is {}", g[initial], g[to_node_index], truth);
        }
        reachable_states
    }

    pub fn prune_graph(&self, g: &mut Graph<String, String>, prune_states: &Vec<usize>) {
        for state in prune_states.iter() {
            let delete = g.node_indices().
                find(|x| g[*x] == format!("({},{})", self.states[*state].s, self.states[*state].q)).unwrap();
            g.remove_node(delete);
        }
    }

    pub fn generate_graph(&self) -> Graph<String, String> {
        let mut graph: Graph<String, String> = Graph::new();

        for state in self.states.iter() {
            graph.add_node(format!("({},{})", state.s, state.q));
        }
        for transition in self.transitions.iter() {
            let origin_index = graph.node_indices().
                find(|x| graph[*x] == format!("({},{})", transition.sq.s, transition.sq.q)).unwrap();
            for sq_prime in transition.sq_prime.iter() {
                let destination_index = match graph.node_indices().
                    find(|x| graph[*x] == format!("({},{})", sq_prime.state.s, sq_prime.state.q)){
                    None => {
                        println!("transition: {:?}", transition);
                        panic!("state: ({},{:?}) not found!", sq_prime.state.s, sq_prime.state.q)
                    }
                    Some(x) => {x}
                };
                graph.add_edge(origin_index, destination_index, transition.a.to_string());
            }
        }
        graph
    }

    pub fn modify_rewards(&mut self, dfa: &DFA) {
        //println!("dfa acc: {:?}", dfa.acc);
        //println!("dfa rej: {:?}", dfa.dead);
        for transition in self.transitions.iter_mut().
            filter(|x| dfa.acc.iter().any(|y| *y == x.sq.q && x.a != "tau")){
            //println!("state: ({},{}) -> 0.0", transition.sq.s, transition.sq.q);
            transition.reward = 0.0;
        }
    }

    pub fn modify_complete(&mut self, dfa: &DFA) {
        let mut transition_modifications: Vec<DFAProductTransition> = Vec::new();
        let mut state_modifications: HashSet<DFAModelCheckingPair> = HashSet::new();
        let mut label_modifications: HashSet<DFAProductLabellingPair> = HashSet::new();
        for state in self.states.iter().
            filter(|x| dfa.acc.iter().all(|y| *y != x.q)) {
            for transition in self.transitions.iter_mut().
                filter(|x| x.sq == *state &&
                    x.sq_prime.iter().
                        any(|xx| dfa.acc.iter().
                            any(|yy| *yy == xx.state.q))) {
                //println!("observed transitions for state: {:?}", transition);
                for sq_prime in transition.sq_prime.iter_mut().
                    filter(|x| dfa.acc.iter().any(|y| *y == x.state.q)){
                    transition_modifications.push(DFAProductTransition {
                        sq: DFAModelCheckingPair { s: 999, q: sq_prime.state.q },
                        a: "tau".to_string(),
                        sq_prime: vec![DFATransitionPair{ state: DFAModelCheckingPair { s: sq_prime.state.s, q: sq_prime.state.q }, p: 1.0 }],
                        reward: transition.reward
                    });
                    sq_prime.state.s = 999; //  change the transition state to s*=999 coded
                    state_modifications.insert(DFAModelCheckingPair {
                        s: 999,
                        q: sq_prime.state.q
                    });
                    // finally we need to modify the labels which go along with the additional
                    // reward state
                    label_modifications.insert(DFAProductLabellingPair {
                        sq: DFAModelCheckingPair { s: 999, q: sq_prime.state.q },
                        w: vec!["com".to_string()]
                    });
                }
            }
        }
        for state_mod in state_modifications.into_iter() {
            self.states.push(state_mod);
        }
        for transition in transition_modifications.into_iter() {
            self.transitions.push(transition);
        }
        for label in label_modifications.into_iter() {
            self.labelling.push(label);
        }
    }

    pub fn edit_labelling(&mut self, dfa: &DFA, mdp: &MDP) {
        // edit labelling
        //println!("dfa accepting: {:?}", dfa.acc);
        //println!("dfa rejecting: {:?}", dfa.dead);
        for label in self.labelling.iter_mut() {
            if label.sq == self.initial {
                // this is an initial state
                label.w.push("ini".to_string());
                //println!("state {:?} has been identified as a initial state", label.sq);
            } else if dfa.dead.iter().any(|x| *x == label.sq.q) && label.sq.s == mdp.initial {
                // this is a rejected state
                label.w.push("fai".to_string());
                //println!("state {:?} has been identified as a rejecting state", label.sq);
            } else if dfa.acc.iter().any(|x| *x == label.sq.q) && label.sq.s == mdp.initial {
                // this is a successful state from which switch transitions are possible to
                // hand over
                //println!("state {:?} has been identified as an accepting state", label.sq);
                label.w.push("suc".to_string());
            }
        }
    }

    /// Creates an empty Product MDP as a skeleton structure for filling in with various
    /// implementation functions
    pub fn default() -> DFAProductMDP {
        DFAProductMDP {
            states: vec![],
            initial: DFAModelCheckingPair { s: 0, q: 0 },
            transitions: vec![],
            labelling: vec![]
        }
    }
}

#[derive(Clone, Debug)]
pub struct TeamInput {
    pub agent: usize,
    pub task: usize,
    pub product: DFAProductMDP,
    pub dead: Vec<u32>,
    pub acc: Vec<u32>
}

impl TeamInput {
    pub fn default() -> TeamInput {
        TeamInput {
            agent: 0,
            task: 0,
            product: DFAProductMDP {
                states: vec![],
                initial: DFAModelCheckingPair { s: 0, q: 0 },
                transitions: vec![],
                labelling: vec![],
            },
            dead: vec![],
            acc: vec![],
        }
    }
}

/// Filters transitions for the current agent and task
fn filter_states<'a>(input: &'a [TeamState], team_transtions: &'a [TeamTransition],
                     i: &'a usize, j: &'a usize) -> impl Iterator<Item = &'a TeamState> {
    let mut switch_states: Vec<TeamState> = Vec::new();
    for transition in team_transtions.iter().
        filter(|x| x.from.agent == *i && x.from.task == *j && x.a == "swi"){
        for state in transition.to.iter() {
            switch_states.push(state.state.clone());
            println!("from ({},{},{},{}), to state: ({},{},{},{})",
                     transition.from.state.s, transition.from.state.q, transition.from.agent, transition.from.task,
                     state.state.state.s, state.state.state.q, state.state.agent, state.state.task
            );
        }
    }
    input.iter().filter(move |x| (x.agent == *i && x.task == *j) || (switch_states.iter().any(|z| *z == **x)))
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct TeamState {
    pub state: DFAModelCheckingPair,
    pub agent: usize,
    pub task: usize
}

impl TeamState {
    fn default() -> TeamState {
        TeamState {
            state: DFAModelCheckingPair { s: 0, q: 0 },
            agent: 0,
            task: 0
        }
    }
}

#[derive(Debug)]
pub struct TeamTransition {
    pub from: TeamState,
    pub a: String,
    pub to: Vec<TeamTransitionPair>,
    pub reward: Vec<f64>
}

#[derive(Debug, Clone)]
pub struct TeamTransitionPair {
    pub state: TeamState,
    pub p: f64
}

#[derive(Debug)]
pub struct TeamLabelling {
    pub state: TeamState,
    pub label: Vec<String>
}

#[derive(Debug, Clone)]
pub struct Mu {
    team_state: TeamState,
    action: Option<String>,
    task_local_index: usize,
    agent_local_index: usize
}

impl Mu {
    fn default() -> Mu {
        Mu {
            team_state: TeamState {
                state: DFAModelCheckingPair { s: 0, q: 0 },
                agent: 0,
                task: 0
            },
            action: None,
            task_local_index: 0,
            agent_local_index: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TeamInitState {
    pub state: TeamState,
    pub index: usize,
    pub obj_index: usize
}

impl TeamInitState {
    fn default() -> TeamInitState {
        TeamInitState {
            state: TeamState {
                state: DFAModelCheckingPair { s: 0, q: 0 },
                agent: 0,
                task: 0
            },
            index: 0,
            obj_index: 0
        }
    }
}

pub struct TeamMDP {
    pub initial: Vec<TeamInitState>,
    pub states: Vec<TeamState>,
    pub transitions: Vec<TeamTransition>,
    pub labelling: Vec<TeamLabelling>,
    pub num_agents: usize,
    pub num_tasks: usize,
}

impl TeamMDP {
    pub fn create_states(&mut self, team_input: &Vec<TeamInput>) {

        let mut initial: Vec<TeamInitState> = vec![TeamInitState::default(); self.num_agents + self.num_tasks];
        for task_agent in team_input.iter() {
            for state in task_agent.product.states.iter() {
                self.states.push(TeamState {
                    state: DFAModelCheckingPair { s: state.s, q: state.q },
                    agent: task_agent.agent,
                    task: task_agent.task
                });
            }
        }

        for k in 0..self.num_agents {

            let init_local_state = team_input.iter().
                find(|x| x.agent == k && x.task == 0).unwrap().product.initial.clone();
            let init_state = TeamState {
                state: DFAModelCheckingPair { s: init_local_state.s, q: init_local_state.q },
                agent: k,
                task: 0
            };
            let init_state_index: usize = self.states.iter().position(|x| *x == init_state).unwrap();
            initial[k] = TeamInitState {
                state: init_state,
                index: init_state_index,
                obj_index: k
            };
        }

        for k in 0..self.num_tasks {
            let init_local_state = team_input.iter().
                find(|x| x.task == k && x.agent == 0).unwrap().product.initial.clone();
            let init_state = TeamState {
                state: DFAModelCheckingPair { s: init_local_state.s, q: init_local_state.q },
                agent: 0,
                task: k
            };
            let init_state_index: usize = self.states.iter().position(|x| *x == init_state).unwrap();
            initial[self.num_agents + k] = TeamInitState {
                state: init_state,
                index: init_state_index,
                obj_index: self.num_agents + k
            }
        }
        self.initial = initial;
    }

    pub fn create_transitions_and_labelling(&mut self, team_input: &Vec<TeamInput>) {
        for state in self.states.iter() {
            for input in team_input.iter().
                filter(|x| x.task == state.task && x.agent == state.agent) {
                // Transitions
                for transition in input.product.transitions.iter().
                    filter(|x| x.sq == state.state) {
                    let mut state_prime: Vec<TeamTransitionPair> = Vec::new();
                    for s_prime in transition.sq_prime.iter() {
                        state_prime.push(TeamTransitionPair {
                            state: TeamState {
                                state: DFAModelCheckingPair { s: s_prime.state.s, q: s_prime.state.q },
                                agent: state.agent,
                                task: state.task
                            },
                            p: s_prime.p
                        })
                    }
                    // this transition is inherited from the input product model, and it will be zero
                    // everywhere else
                    let mut rewards: Vec<f64> = vec![0.0; self.num_tasks + self.num_agents];
                    let state_label = input.product.labelling.iter().
                        find(|x| x.sq == state.state).unwrap().w.to_vec();
                    //println!("state label: {:?}", state_label);
                    if state_label.iter().all(|x| *x != "suc" && *x != "fai") {
                        //println!("state: {:?}, agent: {}, task: {} contains no completion labels", state.state, state.agent, state.task);
                        if transition.reward != 0f64 && transition.a != "tau" {
                            rewards[input.agent] = -1.0 * transition.reward;
                        }
                    }
                    //println!("debug rewards: {:?}", rewards);
                    self.transitions.push(TeamTransition {
                        from: TeamState {
                            state: DFAModelCheckingPair { s: state.state.s, q: state.state.q },
                            agent: state.agent,
                            task: state.task
                        },
                        a: transition.a.to_string(),
                        to: state_prime,
                        // the question is how do we assign a value to the rewards vector
                        reward: rewards
                    });
                }
                // Labelling
                for label_pair in input.product.labelling.iter().
                    filter(|x| x.sq == state.state) {
                    //println!("working on agent: {0} of {2}; task: {1} of {3}", state.agent, state.task, self.num_agents - 1, self.num_tasks - 1);
                    // switching across agents, same task
                    if label_pair.w.iter().any(|x| *x == "ini" || *x == "suc" || *x == "fai") &&
                        state.agent < self.num_agents - 1 {
                        let next_agent_index = team_input.iter().
                            position(|x| x.agent == state.agent + 1).unwrap();
                        let new_switch_transition = TeamTransition {
                            from: TeamState {
                                state: DFAModelCheckingPair { s: state.state.s, q: state.state.q },
                                agent: state.agent,
                                task: state.task
                            },
                            a: "swi".to_string(),
                            to: vec![TeamTransitionPair{
                                state: TeamState {
                                    state: DFAModelCheckingPair {
                                        s: team_input[next_agent_index].product.initial.s,
                                        q: state.state.q
                                    },
                                    agent: input.agent + 1,
                                    task: input.task
                                },
                                p: 1.0
                            }],
                            reward: vec![0.0; self.num_agents + self.num_tasks]
                        };
                        //println!("creating switch transition for agent: {}, task: {}, of: {:?}", state.agent, state.task, new_switch_transition);
                        self.transitions.push(new_switch_transition);

                    } else if label_pair.w.iter().any(|x| *x == "suc" || *x == "fai") &&
                        state.agent == self.num_agents - 1 && state.task < self.num_tasks - 1 {
                        // the switch transition increases the number of tasks and passes it to the
                        // first agent
                        let first_agent_index = team_input.iter().
                            position(|x| x.agent == 0 && x.task == state.task + 1).unwrap();
                        self.transitions.push(TeamTransition {
                            from: TeamState {
                                state: DFAModelCheckingPair { s: state.state.s, q: state.state.q },
                                agent: state.agent,
                                task: state.task
                            },
                            a: "swi".to_string(),
                            to: vec![TeamTransitionPair {
                                state: TeamState {
                                    state: DFAModelCheckingPair {
                                        s: team_input[first_agent_index].product.initial.s,
                                        q: team_input[first_agent_index].product.initial.s
                                    },
                                    agent: 0,
                                    task: state.task + 1
                                },
                                p: 1.0
                            }],
                            reward: vec![0.0; self.num_agents + self.num_tasks]
                        })
                    } else if label_pair.w.iter().any(|x| *x == "suc" || *x == "fai") &&
                        state.agent == self.num_agents - 1 && state.task == self.num_tasks - 1 {
                        self.labelling.push(TeamLabelling {
                            state: TeamState {
                                state: DFAModelCheckingPair { s: state.state.s, q: state.state.q },
                                agent: state.agent,
                                task: state.task
                            },
                            label: vec!["done".to_string()]
                        })
                    } else if label_pair.w.iter().any(|x| *x == "com") {
                        self.labelling.push(TeamLabelling {
                            state: TeamState {
                                state: DFAModelCheckingPair { s: state.state.s, q: state.state.q },
                                agent: state.agent,
                                task: state.task
                            },
                            label: vec![format!("com_{}", state.task)]
                        })
                    }
                }
            }
        }
    }

    pub fn assign_task_rewards(&mut self) {
        for transition in self.transitions.iter_mut() {
            let task_index = self.num_agents + transition.from.task;
            //println!("task index: {}", task_index);
            let transition_copy: TeamTransition = TeamTransition {
                from: TeamState {
                    state: DFAModelCheckingPair { s: transition.from.state.s, q: transition.from.state.q },
                    agent: transition.from.agent,
                    task: transition.from.task
                },
                a: transition.a.to_string(),
                to: transition.to.to_vec(),
                reward: transition.reward.to_vec()
            };
            for label in self.labelling.iter().
                filter(|x| x.state.state == transition_copy.from.state) {
                if label.label.iter().any(|x| *x == format!("com_{}", transition_copy.from.task)) {
                    transition.reward[task_index] = 1.0;
                }
            }
        }
    }

    pub fn generate_graph(&self) -> Graph<String, String> {
        let mut graph: Graph<String, String> = Graph::new();
        //
        for state in self.states.iter() {
            graph.add_node(format!("({},{},{},{})", state.state.s, state.state.q, state.agent, state.task));
        }
        for transition in self.transitions.iter() {
            let origin_index = graph.node_indices().
                find(|x| graph[*x] ==
                    format!(
                        "({},{},{},{})",
                        transition.from.state.s,
                        transition.from.state.q,
                        transition.from.agent,
                        transition.from.task
                    )).unwrap();
            for trans_to in transition.to.iter() {
                let destination_index = match graph.node_indices().
                    find(|x| graph[*x] == format!(
                        "({},{},{},{})",
                        trans_to.state.state.s,
                        trans_to.state.state.q,
                        trans_to.state.agent,
                        trans_to.state.task
                    )) {
                    None => {
                        println!("Transition {:?}", transition);
                        panic!("state: ({},{},{},{}) not found",
                                      trans_to.state.state.s,
                                      trans_to.state.state.q,
                                      trans_to.state.agent,
                                      trans_to.state.task
                    );}
                    Some(x) => {x}
                };
                graph.add_edge(origin_index, destination_index, transition.a.to_string());
            }
        }
        graph
    }

    pub fn min_exp_tot(&self, w: &[f64], eps: &f64) -> Option<(Vec<Mu>,Vec<f64>)> {
        // TODO go through and comment this code in a detailed manner, because I am sure that I will forget it
        let mut mu: Vec<Mu> = self.states.iter().map(|x| Mu{
            team_state: TeamState {
                state: DFAModelCheckingPair { s: x.state.s, q: x.state.q },
                agent: x.agent,
                task: x.task
            },
            action: None,
            task_local_index: 0,
            agent_local_index: 0,
        }).collect();
        let weight = arr1(w);

        let mut task_states: Vec<Vec<(usize, TeamState)>> = Vec::new();
        let mut agent_states: Vec<Vec<(usize, TeamState)>> = Vec::new();

        let mut xtotexpcostbar: Vec<Vec<f64>> = Vec::new();
        let mut ytotexpcostbar: Vec<Vec<f64>> = Vec::new();
        let mut xtaskbar: Vec<Vec<f64>> = Vec::new();
        let mut ytaskbar: Vec<Vec<f64>> = Vec::new();
        let mut xagentbar: Vec<Vec<f64>> = Vec::new();
        let mut yagentbar: Vec<Vec<f64>> = Vec::new();

        for j in (0..self.num_tasks) {
            let state_space: Vec<(usize, TeamState)> = self.states.iter().filter(|x| x.task == j).enumerate().
                map(|(k, x)| (k, *x)).collect();

            let state_space_len: usize = state_space.len();
            task_states.push(state_space);
            let j_task_x: Vec<f64> = vec![0.0; state_space_len];
            let j_task_y: Vec<f64> = vec![0.0; state_space_len];
            xtotexpcostbar.push(j_task_x.clone());
            ytotexpcostbar.push(j_task_y.clone());
            xtaskbar.push(j_task_x);
            ytaskbar.push(j_task_y);
        }

        for i in (0..self.num_agents) {
            let agent_state_space: Vec<(usize, TeamState)> = self.states.iter().
                filter(|x| x.agent == i).enumerate().map(|(k, x)| (k,*x)).collect();
            let i_agent_x: Vec<f64> = vec![0.0; agent_state_space.len()];
            let i_agent_y: Vec<f64> = vec![0.0; agent_state_space.len()];
            agent_states.push(agent_state_space);
            xagentbar.push(i_agent_x);
            yagentbar.push(i_agent_y);
        }

        for j in (0..self.num_tasks).rev() {
            for i in (0..self.num_agents).rev() {
                let mut epsilon: f64 = 1.0;
                while epsilon > *eps {
                    for (k, state) in task_states[j].iter().filter(|(_k, x)| x.agent == i && x.task == j) {
                        // we still need to calculate the overall position index for mu
                        let x_s_index: usize = mu.iter().position(|x| x.team_state == *state).unwrap();
                        let x_agent_index: usize = agent_states[i].iter().position(|(_k, x)| x == state).unwrap();
                        let mut min_action_values: Vec<(String, f64)> = Vec::new();
                        for transition in self.transitions.iter().filter(|x| x.from == *state) {
                            let transition_reward = arr1(&transition.reward);
                            let scalar_weight_rewards = weight.dot(&transition_reward);
                            let mut sum_vect: Vec<f64> = vec![0.0; transition.to.len()];
                            //let mut test_state_reached: bool = false;
                            for (k2, sprime) in transition.to.iter().enumerate() {
                                let x_sprime_index: usize = task_states[sprime.state.task].iter().
                                    position(|(_index, x)| *x == sprime.state).unwrap();
                                sum_vect[k2] = sprime.p * xtotexpcostbar[j][x_sprime_index];
                            }
                            let sum_vect_sum: f64 = sum_vect.iter().sum();
                            let action_reward = scalar_weight_rewards + sum_vect_sum;
                            min_action_values.push((transition.a.to_string(), action_reward));
                        }

                        let mut v: Vec<_> = min_action_values.iter().
                            map(|(z, x)| (z, NonNan::new(*x).unwrap())).collect();
                        v.sort_by_key(|key| key.1);
                        let max_pair = v.last().unwrap();
                        let max_val = max_pair.1.inner();
                        let arg_max = max_pair.0;
                        ytotexpcostbar[j][*k] = max_val;
                        mu[x_s_index].action = Some(arg_max.to_string());
                        mu[x_s_index].team_state = state.clone();
                        mu[x_s_index].task_local_index = *k;
                        mu[x_s_index].agent_local_index = x_agent_index;
                    }
                    let y_bar_diff = absolute_diff_vect(&xtotexpcostbar[j], &ytotexpcostbar[j]);
                    let mut y_bar_diff_max_vect: Vec<NonNan> = y_bar_diff.iter().
                        map(|x| NonNan::new(*x).unwrap()).collect();
                    y_bar_diff_max_vect.sort();
                    epsilon = y_bar_diff_max_vect.last().unwrap().inner();
                    //println!("eps: {}", epsilon);
                    xtotexpcostbar[j] = ytotexpcostbar[j].to_vec();
                }
                epsilon = 1.0;
                while epsilon > *eps {
                    for (k, state) in task_states[j].iter().filter(|(_k,x)| x.agent == i) {
                        let mu_state = mu.iter().find(|x| x.team_state == *state).unwrap();
                        let action = match &mu_state.action {
                            None => {panic!("There was no action recorded for state: ({},{},{},{})", state.state.s, state.state.q, state.agent, state.task)}
                            Some(a) => a
                        };
                        let agent_state_index = mu_state.agent_local_index;
                        for transition in self.transitions.iter().
                            filter(|x| x.from == *state && x.a == *action) {
                            let mut sum_vect_agent: Vec<f64> = vec![0.0; transition.to.len()];
                            let mut sum_vec_task: Vec<f64> = vec![0.0; transition.to.len()];
                            for (l, sprime) in transition.to.iter().enumerate() {
                                if (i < self.num_agents - 1) || (i == self.num_agents - 1 && *action != "swi") {
                                    let x_sprime_agent_index: usize = agent_states[sprime.state.agent].iter().position(|(_ind, x)| *x == sprime.state).unwrap();
                                    let x_sprime_task_index: usize = task_states[sprime.state.task].iter().position(|(_ind, x)| *x == sprime.state).unwrap();
                                    sum_vect_agent[l] = sprime.p * xagentbar[sprime.state.agent][x_sprime_agent_index];
                                    sum_vec_task[l] = sprime.p * xtaskbar[sprime.state.task][x_sprime_task_index];
                                } else {
                                    let x_sprime_agent_index: usize = agent_states[sprime.state.agent].iter().position(|(_ind, x)| *x == sprime.state).unwrap();
                                    sum_vect_agent[l] = sprime.p * xagentbar[sprime.state.agent][x_sprime_agent_index];
                                }
                            }

                            let p_trans_agent: f64 = sum_vect_agent.iter().sum();
                            let p_trans_task: f64 = sum_vec_task.iter().sum();

                            yagentbar[i][agent_state_index] = transition.reward[i] + p_trans_agent;
                            ytaskbar[j][*k] = transition.reward[self.num_agents + j] + p_trans_task;
                        }
                    }
                    let mut eps_inner: f64 = 0.0;
                    let xbar_agent_val: Vec<f64> = xagentbar[i].iter().map(|x| *x).collect();
                    let ybar_agent_val: Vec<f64> = yagentbar[i].iter().map(|x| *x).collect();
                    let diff_agent = absolute_diff_vect(&xbar_agent_val, &ybar_agent_val);
                    let mut diff_agent_nonan: Vec<NonNan> = diff_agent.iter().map(|x| NonNan::new(*x).unwrap()).collect();
                    diff_agent_nonan.sort();
                    let max_val_agent = diff_agent_nonan.last().unwrap().inner();
                    if max_val_agent > eps_inner {
                        eps_inner = max_val_agent;
                    }
                    xagentbar[i] = yagentbar[i].to_vec();
                    let xbar_task_val: Vec<f64> = xtaskbar[j].iter().map(|x| *x).collect();
                    let ybar_task_val: Vec<f64> = ytaskbar[j].iter().map(|x| *x).collect();
                    let diff_task = absolute_diff_vect(&xbar_task_val, &ybar_task_val);
                    let mut diff_task_nonan: Vec<NonNan> = diff_task.iter().map(|x| NonNan::new(*x).unwrap()).collect();
                    diff_task_nonan.sort();
                    let max_val_task = diff_task_nonan.last().unwrap().inner();
                    if max_val_task > eps_inner {
                        eps_inner = max_val_task;
                    }
                    xtaskbar[j] = ytaskbar[j].to_vec();
                    epsilon = eps_inner;
                    //println!("epsilon bar: {}", epsilon);
                }
            }
        }

        let mut r: Vec<f64> = vec![0.0; self.num_agents + self.num_tasks];
        for k in 0..(self.num_tasks + self.num_agents) {
            let init_state = self.initial.iter().find(|x| x.obj_index == k).unwrap().state;
            if k < self.num_agents {
                let index = agent_states[k].iter().position(|(_index, x)| *x == init_state).unwrap();
                r[k] = yagentbar[k][index];

            } else {
                let index: usize = task_states[k - self.num_agents].iter().position(|(_index, x)| *x == init_state).unwrap();
                //println!("init state: ({},{},{},{})", init_state.state.s, init_state.state.q, init_state.agent, init_state.task);
                r[k] = ytaskbar[init_state.task][index];
            }
        }
        //println!("y_agent: {:?}", yagentbar);
        //println!("r: {:?}", r);
        Some((mu, r))
    }

    pub fn team_done_states(&self) -> Vec<TeamState> {
        let mut done_states: Vec<TeamState> = Vec::new();
        for state in self.states.iter().
            filter(|x| self.labelling.iter().any(|x| x.state == x.state &&
                x.label.iter().any(|y| *y == "done"))) {
            done_states.push(state.clone())
        }
        done_states
    }

    pub fn modify_final_rewards(&mut self, team_inputs: &Vec<TeamInput>) {
        let num_tasks: usize = self.num_tasks.clone();
        let num_agents: usize = self.num_agents.clone();
        for input in team_inputs.iter().
            filter(|x| x.task == num_tasks - 1 && x.agent == num_agents - 1) {
            for transition in self.transitions.iter_mut().
                filter(|x| x.from.agent == input.agent && x.from.task == input.task &&
                    (
                        input.dead.iter().any(|y| *y == x.from.state.q) ||
                        input.acc.iter().any(|y| *y == x.from.state.q)
                    ) && x.a != "tau")
            {
                //println!("modifying rewards for state: ({},{},{},{})",
                //         transition.from.state.s, transition.from.state.q, input.agent, input.task);
                transition.reward = vec![0.0; self.num_agents + self.num_tasks];
            }
        }
    }

    pub fn default() -> TeamMDP {
        TeamMDP {
            initial: vec![],
            states: vec![],
            transitions: vec![],
            labelling: vec![],
            num_agents: 0,
            num_tasks: 0
        }
    }

    pub fn multi_obj_sched_synth(&self, target: &Vec<f64>, eps: &f64) -> Option<Alg1Output> {
        let output = Alg1Output { hullset: vec![], mu: vec![] };
        let mut hullset: Vec<Vec<f64>> = Vec::new();
        let mut mu_vect: Vec<Vec<Mu>> = Vec::new();

        println!("num tasks: {}, num agents {}", self.num_tasks, self.num_agents);

        let mut extreme_points: Vec<Vec<f64>> = vec![vec![0.0; self.num_agents + self.num_tasks]; self.num_agents + self.num_tasks];

        for k in 0..(self.num_agents + self.num_tasks) {
            extreme_points[k][k] = 1.0;
            let w_extr: &Vec<f64> = &extreme_points[k];
            //println!("w: {:?}", w_extr);
            let safe_ret = self.min_exp_tot(&w_extr, eps);
            match safe_ret {
                Some((mu_new, r)) => {
                    hullset.push(r);
                },
                None => panic!("No value was returned from the maximisation")
            }
        }
        /*println!("extreme points: ");
        for k in hullset.iter() {
            print!("p: [");
            for (i, j) in k.iter().enumerate() {
                if i == 0 {
                    print!("{0:.1$}", j, 2);
                } else {
                    print!(" ,{0:.1$}", j, 2);
                }
            }
            print!("]");
            println!();
        }

         */
        let mut member_closure: bool = false;
        let dim = self.num_tasks + self.num_agents;
        let t_arr1 = arr1(target);
        while !member_closure {
            let w_new = lp2(&hullset, target, &dim);
            let safe_ret = self.min_exp_tot(&w_new, eps);
            match safe_ret {
                Some((mu_new, r)) => {
                    println!("new r: {:?}", r);
                    let weight_arr1 = arr1(&w_new);
                    let r_arr1 = arr1(&r);
                    let wr_dot = weight_arr1.dot(&r_arr1);
                    let wt_dot = weight_arr1.dot(&t_arr1);
                    if wr_dot < wt_dot {
                        println!("Multi-objective criteria not possible");
                        return None
                    }
                    hullset.push(r);
                    mu_vect.push(mu_new);
                    if member_closure_set(&hullset, target) {
                        println!("member set found");
                        println!("r: {:?}", hullset.last().unwrap());
                        member_closure = true
                    }
                },
                None => panic!("No value was returned from the maximisation")
            }
        }
        Some(output)
    }

}

/// Linear programming test function. Simple implementation of two tasks, to two agents
pub fn pareto_lp_delta() {

    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let a1 = problem.add_var(0.0, (0.0, 1.0));
    let a2 = problem.add_var(0.0, (0.0, 1.0));
    let a3 = problem.add_var(0.0, (0.0, 1.0));
    let a4 = problem.add_var(0.0, (0.0, 1.0));
    let delta = problem.add_var(1.0, (f64::NEG_INFINITY, f64::INFINITY));

    problem.add_constraint(&[(a1, -7.0),(a2, -7.0),(a3, 0.5),(a4, 0.5), (delta, 1.0)], ComparisonOp::Ge, 0.0);
    problem.add_constraint(&[(a1, -8.5), (delta, 1.0)], ComparisonOp::Le, 0.0);
    problem.add_constraint(&[(a2, -13.66),(delta, 1.0)], ComparisonOp::Le, 0.0);
    problem.add_constraint(&[(a1, -6.85),(a2, -6.33),(a3, 0.8),(delta, 1.0)], ComparisonOp::Le, 0.0);
    problem.add_constraint(&[(a1, -13.18),(a3, 0.64),(delta, 1.0)], ComparisonOp::Le, 0.0);
    problem.add_constraint(&[(a1, -1.0),(a2, 1.0),(a3, 1.0),(a4, 1.0)], ComparisonOp::Eq, 1.0);

    let sol = problem.solve().unwrap();
    println!("solution: {:?}", sol);
    println!("solution => a1: {:?}, a2: {:?}, a3: {:?}, a4: {:?}", sol[a1], sol[a2], sol[a3], sol[a4]);
}

pub fn lp2(h: &Vec<Vec<f64>>, t: &Vec<f64>, dim: &usize) -> Vec<f64> {
    let mut problem = Problem::new(OptimizationDirection::Maximize);

    let mut weight_vars: HashMap<String, Variable> = HashMap::new();
    for i in 0..*dim  {
        weight_vars.insert(format!("w{}", i), problem.add_var(0.0, (0.0, 1.0)));
    }
    // Objective variable
    let delta = problem.add_var(1.0, (f64::NEG_INFINITY, f64::INFINITY));

    // Inequalities of the form: Target points or points above the separating hyperplane
    // <a, x> + delta >= 0
    let mut lhs = LinearExpr::empty();
    for j in 0..*dim {
        let var = weight_vars.get(&*format!("w{}", j)).unwrap();
        //println!("added pos constraint {} var with coeff: {}", format!("w{}", j), t[j]);
        lhs.add(*var, t[j]);
    }
    lhs.add(delta, 1.0);
    //println!("added constraint: {:?}", lhs);
    problem.add_constraint(lhs, ComparisonOp::Ge, 0.0);

    // Inequalities of the form: point elements of the Hull set, or points below the separating hyperplane
    // <a, x> + delta <= 0
    for neg in h.iter() {
        lhs = LinearExpr::empty();
        for j in 0..*dim {
            let var = weight_vars.get(&*format!("w{}", j)).unwrap();
            lhs.add(*var, neg[j]);
        }
        lhs.add(delta, 1.0);
        //println!("added constraint: {:?}", lhs);
        problem.add_constraint(lhs, ComparisonOp::Le, 0.0);
    }

    // Weight vector must sum to 1 to meet the conditions of convexity
    lhs = LinearExpr::empty();
    for j in 0..*dim {
        let var = weight_vars.get(&*format!("w{}", j)).unwrap();
        lhs.add(*var, 1.0);
    }
    //println!("added constraint: {:?}", lhs);
    problem.add_constraint(lhs, ComparisonOp::Eq, 1.0);

    let sol = problem.solve().unwrap();
    let mut result: Vec<f64> = vec![0.0; *dim];
    for j in 0..*dim {
        let var = weight_vars.get(&*format!("w{}", j)).unwrap();
        result[j] = sol[*var];
    }
    result
}

/*
pub fn pareto_lp(h: &Vec<Vec<f64>>, k: &Vec<Vec<f64>>, dim: &u32) -> Vec<f64> {
    let mut problem = Problem::new(OptimizationDirection::Maximize);

    let mut vars: HashMap<String, Variable> = HashMap::new();
    for i in 0..*dim {
        vars.insert(format!("w{}", i), problem.add_var(0., (0., 1.)));
    }
    vars.insert(format!("delta"), problem.add_var(1.0, (f64::NEG_INFINITY, f64::INFINITY)));
    let b = problem.add_var(0., (f64::NEG_INFINITY, f64::INFINITY));
    for x in h.iter() {
        let mut lhs = LinearExpr::empty();
        for j in 0..*dim {
            lhs.add(*vars.get(&*format!("w{}", j)).unwrap(), x[j as usize]);
        }
        lhs.add(b, 1.0);
        lhs.add(*vars.get("delta").unwrap(), -1.0);
        problem.add_constraint(lhs, ComparisonOp::Ge, 0.);
    }
    for x in k.iter() {
        let mut lhs = LinearExpr::empty();
        for j in 0..*dim {
            lhs.add(*vars.get(&*format!("w{}", j)).unwrap(), x[j as usize]);
        }
        lhs.add(b, 1.0);
        lhs.add(*vars.get("delta").unwrap(), 1.0);
        problem.add_constraint(lhs, ComparisonOp::Le, 0.);
    };
    let mut lhs = LinearExpr::empty();
    for j in 0..*dim {
        lhs.add(*vars.get(&*format!("w{}", j)).unwrap(), 1.0)
    }
    problem.add_constraint(lhs, ComparisonOp::Eq, 1.);

    let solution = problem.solve().unwrap();
    let mut w: Vec<f64> = vec![0.; *dim as usize];
    for i in 0..*dim {
        let w_val: f64 = ((solution[*vars.get(&*format!("w{}", i)).unwrap()] * 1000.).round() / 1000.) as f64;
        w[i as usize] = w_val;
    }
    w
}

 */

pub fn member_closure_set(hull_set: &Vec<Vec<f64>>, t: &Vec<f64>) -> bool {
    for x in hull_set.iter() {
        let upward_closure = t.iter().zip(x).all(|(t,r)| r >= t);
        if upward_closure {
            return true
        }
    }
    false
}

pub struct Alg1Output {
    pub hullset: Vec<Vec<f64>>,
    pub mu: Vec<Vec<(TeamState, Option<String>)>>
}

pub fn absolute_diff_vect(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let a_b: Vec<_> = a.iter().zip(b.into_iter()).collect();
    let mut c: Vec<f64> = vec![0.0; a.len()];
    for (i, (val1,val2)) in a_b.iter().enumerate() {
        c[i] = (**val1 - **val2).abs();
    }
    c
}

pub fn parse_int(s: &str) -> Result<u32, ParseIntError> {
    s.parse::<u32>()
}

pub fn parse_str_vect(s: &str) -> serde_json::Result<Vec<u32>> {
    let u: Vec<u32> = serde_json::from_str(s)?;
    Ok(u)
}

pub fn read_mdp_json<'a, P: AsRef<Path>>(path:P) -> Result<Vec<MDP>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}

pub fn read_dra_json<'a, P: AsRef<Path>>(path:P) -> Result<Vec<DRA>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}

pub fn read_dfa_json<'a, P: AsRef<Path>>(path: P) -> Result<Vec<DFA>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}

pub fn read_target<'a, P: AsRef<Path>>(path: P) -> Result<Target, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}

#[derive(PartialOrd, PartialEq, Debug, Clone, Copy)]
pub struct NonNan(f64);

impl NonNan {
    pub fn new(val: f64) -> Option<NonNan> {
        if val.is_nan() {
            None
        } else {
            Some(NonNan(val))
        }
    }

    pub fn inner(self) -> f64 {
        self.0
    }
}

impl Eq for NonNan {}

impl Ord for NonNan {
    fn cmp(&self, other: &NonNan) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}