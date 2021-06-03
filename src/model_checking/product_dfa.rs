use petgraph::{Graph, graph::NodeIndex};
use petgraph::algo::{kosaraju_scc, has_path_connecting, all_simple_paths};
use itertools::Itertools;
use std::collections::{HashSet, VecDeque, HashMap};
use std::iter::{FromIterator, Filter};
extern crate serde_json;
use serde::Deserialize;
use super::mdp;
use super::dfa;

use mdp::*;
use dfa::*;
use std::thread::current;
use petgraph::graph::Node;

pub struct ProductDFA {
    pub states: Vec<Vec<u32>>,
    pub sigma: Vec<String>,
    pub initial: Vec<u32>,
    pub delta: Vec<ProductDFATransitions>,
    pub acc: Vec<u32>,
    pub dead: Vec<u32>
}

pub struct ProductDFATransitions {
    pub q: Vec<u32>,
    pub w: Vec<String>,
    pub q_prime: Vec<u32>
}

impl ProductDFA {
    pub fn create_states(&mut self, dfa_new: &DFA) {
        if self.states.is_empty() {
            for q in dfa_new.states.iter() {
                self.states.push(vec![*q]);
            }
        } else {
            let mut states: Vec<Vec<u32>> = Vec::new();
            let s = self.states.clone().into_iter().cartesian_product(dfa_new.states.clone().into_iter());
            for (v, val) in s.into_iter() {
                let mut new_q: Vec<u32> = v;
                new_q.push(val);
                states.push(new_q);
            }
            self.states = states;
        }
    }

    pub fn create_transitions(&mut self, dfa: &DFA) {
        let mut new_transitions: Vec<ProductDFATransitions> = Vec::new();
        for w in self.sigma.iter() {
            for q in self.states.iter() {
                if self.delta.is_empty() {
                    // first task
                    for transition in dfa.delta.iter().
                        filter(|x| vec![x.q] == *q && x.w.iter().any(|x| x == w)) {
                        let mut update_transition: Option<usize> = None;
                        match new_transitions.iter().position(|x| x.q == *q && x.q_prime == vec![transition.q_prime]) {
                            None => {
                                new_transitions.push(ProductDFATransitions {
                                    q: vec![transition.q],
                                    w: transition.w.to_vec(),
                                    q_prime: vec![transition.q_prime]
                                });
                            }
                            Some(x) => update_transition = Some(x)
                        }
                        match update_transition {
                            None => {}
                            Some(x) => new_transitions[x].w.push(w.to_string())
                        }
                    }
                } else {
                    let old_q_len: usize = q.len()-1;
                    //println!("old q len: {:?}", &q[..old_q_len]);
                    let delta = match
                        self.delta.iter().
                        find(|x| &x.q[..] == &q[..old_q_len] && x.w.iter().any(|y| y == w)) {
                        None => { panic!("did not find q: {:?}", q);}
                        Some(x) => {
                            x
                        }
                    };
                    for q2 in dfa.states.iter().filter(|x| *x == q.last().unwrap()) {
                        for transitions in dfa.delta.iter().
                            filter(|x| x.q == *q2 && x.w.iter().any(|y| y == w)) {
                            let mut temp_from_state: Vec<u32> = q.to_vec();
                            //temp_from_state.push(*q2);
                            let mut temp_to_state: Vec<u32> = delta.q_prime.to_vec();
                            temp_to_state.push(transitions.q_prime);
                            let mut add_transition: Option<ProductDFATransitions> = None;
                            let mut update_transition: Option<(usize, String)> = None;
                            match new_transitions.iter_mut().
                                position(|x| x.q == temp_from_state && x.q_prime == temp_to_state) {
                                None => {
                                    // we can add the transition because it doesn't exist
                                    add_transition = Some(ProductDFATransitions {
                                        q: temp_from_state,
                                        w: vec![w.to_string()],
                                        q_prime: temp_to_state
                                    });
                                }
                                Some(mut x) => {
                                    // The transition already exists and we are only interested in accumulating words for which
                                    // the transition in valid
                                    update_transition = Some((x, w.to_string()));
                                }
                            }
                            let mut double_update: u8 = 0;
                            match add_transition {
                                None => { }
                                Some(x) => {new_transitions.push(x); double_update += 1;}
                            }

                            match update_transition {
                                None => {}
                                Some((k,x)) => {
                                    new_transitions[k].w.push(x);
                                    double_update += 1;
                                }
                            }
                            if double_update >= 2 {
                                panic!("Cannot both add a transition and update it");
                            }
                        }
                    }
                }
            }
        }
        self.delta = new_transitions;
    }

    pub fn create_automaton_graph(&self) -> Graph<String, String> {
        let mut graph: Graph<String, String> = Graph::new();
        for state in self.states.iter() {
            graph.add_node(format!("({:?})", state));
        }
        for transition in self.delta.iter() {
            let origin_index: NodeIndex = graph.node_indices().
                find(|x| graph[*x] == format!("({:?})",transition.q)).unwrap();
            let destination_index: NodeIndex = graph.node_indices().
                find(|x| graph[*x] == format!("({:?})", transition.q_prime)).unwrap();
            let label = if transition.w == self.sigma { "true".to_string() } else { format!("{:?}", transition.w) };
            graph.add_edge(origin_index, destination_index, label);
        }
        graph
    }

    pub fn default() -> ProductDFA {
        ProductDFA {
            states: vec![],
            sigma: vec![],
            initial: vec![],
            delta: vec![],
            acc: vec![],
            dead: vec![]
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProductDFAProductMDP {
    pub states: Vec<StatePair>,
    pub initial: StatePair,
    pub transitions: Vec<ProdMDPTransition>,
    pub labelling: Vec<ProdLabellingPair>
}

impl ProductDFAProductMDP {
    pub fn set_initial(&mut self, mdp: &u32, dfa: &Vec<u32>) {
        self.initial.s = *mdp;
        self.initial.q = dfa.to_vec();
    }

    pub fn create_states(&mut self, mdp: &MDP, dfa: &ProductDFA) {
        self.states = Vec::new();
        let cp = mdp.states.clone().into_iter().
            cartesian_product(dfa.states.clone().into_iter());
        for (m, d) in cp.into_iter() {
            self.states.push(StatePair {
                s: m,
                q: d
            });
        }
    }

    pub fn create_transitions(&mut self, mdp: &MDP, dfa: &ProductDFA, tasks: &usize, dfas: &Vec<(usize, DFA)>) {
        // The transitions for the product DFA x MDP need to be created in a different way than usual
        // we will proceed by DFS through the MDP fixing a particular task j and conduct this routine
        // for all j.
        for task in 0..*tasks {
            let (_task_index, base_dfa) = dfas.iter().find(|(i, _dfa)| *i == task).unwrap();
            let (mut next_initial, mut new_transitions, mut additional_states) =
                self.mdp_bfs_task_transition(mdp, dfa, base_dfa, &(task+1), &vec![self.initial.clone()]);
            // add labelling before additional states consumed
            self.labelling.push(ProdLabellingPair{
                s: StatePair { s: self.initial.s, q: self.initial.q.to_vec() },
                w: vec![format!("ini{}", task)]
            });
            for s in next_initial.iter() {
                self.labelling.push(ProdLabellingPair {
                    s: StatePair { s: s.s, q: s.q.to_vec() },
                    w: vec![format!("finish{}", task)]
                });
            }
            for s in additional_states.iter() {
                self.labelling.push(ProdLabellingPair{
                    s: StatePair { s: s.s, q: s.q.to_vec() },
                    w: vec![format!("jfai{}", task)]
                });
                //println!("adding jfai -> {:?}", s);
            }
            self.states.append(&mut additional_states);
            self.transitions.append(&mut new_transitions);
            // for the remaining tasks
            let mut remaining: Vec<usize> = (0..*tasks).collect();
            remaining = remaining.iter().filter(|x| **x != task).map(|x| *x).collect();
            //println!("remaining tasks: {:?}", remaining);
            for j in remaining.iter() {
                let (_task_index, base_dfa) = dfas.iter().find(|(i, _dfa)| *i == *j).unwrap();
                let mut jplus1_queue: Vec<StatePair> = Vec::new();
                let (additional_initial, mut new_transitions, mut additional_states) =
                    self.mdp_bfs_task_transition(mdp, dfa, base_dfa, &(j+1), &next_initial);
                for s in additional_initial.iter() {
                    self.labelling.push(ProdLabellingPair {
                        s: StatePair { s: s.s, q: s.q.to_vec() },
                        w: vec![format!("ini{}", j)]
                    });
                }
                for transition in new_transitions.into_iter() {
                    match self.transitions.iter().
                        any(|x| x.s == transition.s &&
                            x.reward == transition.reward && x.a == transition.a &&
                            x.s_sprime.iter().
                                all(|x| transition.s_sprime.iter().
                                    all(|y| x.s == y.s))
                        ) {
                        true => {}
                        false => self.transitions.push(transition)
                    }
                }
                for state in additional_states.into_iter() {
                    match self.states.iter().any(|x| *x == state) {
                        true => { }
                        false => {
                            self.labelling.push(ProdLabellingPair{
                                s: StatePair { s: state.s, q: state.q.to_vec() },
                                w: vec![format!("jfai{}", j)]
                            });
                            self.states.push(state);
                        }
                    }
                }
                for init_state in additional_initial.iter() {
                    jplus1_queue.push(init_state.clone());
                }
                next_initial = jplus1_queue;
            }
        }
    }

    /// Function to make sure that the bottom MEC is labelled as complete to easily condition that
    /// rewards are not given in these states.
    pub fn create_final_labelling(&mut self, dfas: &Vec<(usize, DFA)>) {
        let mut new_labels: Vec<ProdLabellingPair> = Vec::new();
        for state in self.states.iter().
            filter(|x| ProductDFAProductMDP::find_final(&x.q, dfas)) {
            match self.labelling.iter_mut().find(|x|x.s == *state) {
                None => {new_labels.push(ProdLabellingPair {
                    s: StatePair { s: state.s, q: state.q.to_vec() },
                    w: vec!["complete".to_string()]
                })},
                Some(mut x) => x.w.push("complete".to_string())
            }
        }
        self.labelling.append(&mut new_labels);
    }

    pub fn mdp_bfs_task_transition(&self, mdp: &MDP, dfa: &ProductDFA, base_dfa: &DFA, task: &usize, initial: &Vec<StatePair>) -> (Vec<StatePair>,Vec<ProdMDPTransition>,Vec<StatePair>) {
        let mut product_transitions: Vec<ProdMDPTransition> = Vec::new();
        let mut modification_transitions: Vec<ProdMDPTransition> = Vec::new();
        let mut additional_states: Vec<StatePair> = Vec::new();
        let mut visited: Vec<bool> = vec![false; self.states.len()];
        let mut prod_stack: VecDeque<StatePair> = VecDeque::new();
        let mut next_initial: Vec<StatePair> = Vec::new();
        for init in initial.iter() {
            prod_stack.push_back(StatePair { s: init.s, q: init.q.to_vec() });
            let init_index = self.states.iter().position(|x| x == init).unwrap();
            visited[init_index] = true;
        }

        while !prod_stack.is_empty() {
            // get the adjacent neighbours for the MDP
            let current_state = prod_stack.pop_front().unwrap();
            if current_state.s == mdp.initial && (
                base_dfa.acc.iter().any(|x| *x ==current_state.q[task - 1]) ||
                base_dfa.dead.iter().any(|x| *x == current_state.q[task-1])) &&
                current_state.q.iter().any(|x| *x == 0) {// 0 is not ideal but easy (represents initial)
                next_initial.push(current_state.clone());
            }
            for transition in mdp.transitions.iter().
                filter(|x| x.s == current_state.s) {
                let mut make_modify: bool = false;
                let mut prod_sprime: Vec<ProdTransitionPair> = Vec::new();
                let mut prod_sprime_mod: Vec<ProdTransitionPair> = Vec::new();
                let mut q_mod: Vec<u32> = Vec::new();
                for s_prime in transition.s_prime.iter() {
                    // process the transitions
                    let labelling = mdp.labelling.iter().find(|x| x.s == s_prime.s).unwrap();
                    let ap = if labelling.w == "" { "".to_string() } else {format!("{}{}", labelling.w, task)};
                    /*if *task == 2usize {
                        println!("searching for: {}", ap);
                    }*/
                    for delta in dfa.delta.iter().
                        filter(|x| x.q == current_state.q &&
                            x.w.iter().any(|y| *y == ap)) {

                        let s_prime_index = self.states.iter().
                            position(|x| x.s == s_prime.s && x.q == delta.q_prime).unwrap();

                        if !visited[s_prime_index] {
                            // insert sprime into the queue
                            visited[s_prime_index] = true;
                            /*if *task == 2usize {
                                println!("delta: {:?}", delta);
                            }*/
                            prod_stack.push_back(StatePair { s: s_prime.s, q: delta.q_prime.to_vec() });
                        }
                        if base_dfa.dead.iter().all(|x| *x != delta.q[*task-1]) &&
                            base_dfa.dead.iter().any(|x| *x == delta.q_prime[*task-1])  {
                            prod_sprime.push(ProdTransitionPair{
                                s: StatePair { s: 999, q: delta.q_prime.to_vec() },
                                p: s_prime.p
                            });
                            make_modify = true;
                            //println!("make modification at: s:{:?}", current_state);
                            q_mod = delta.q_prime.to_vec();
                            prod_sprime_mod.push(ProdTransitionPair{
                                s: StatePair { s: s_prime.s, q: delta.q_prime.to_vec() },
                                p: 1.0
                            })
                        } else {
                            prod_sprime.push(ProdTransitionPair{
                                s: StatePair { s: s_prime.s, q: delta.q_prime.to_vec() },
                                p: s_prime.p
                            });
                        }
                    }
                }
                if make_modify {
                    let mod_state_pair = StatePair{s: 999, q: q_mod.to_vec()};
                    match additional_states.iter().find(|x| **x == mod_state_pair) {
                        None => {additional_states.push(mod_state_pair);}
                        Some(_) => {}
                    }

                    product_transitions.push(ProdMDPTransition{
                        s: StatePair{s: current_state.s, q: current_state.q.to_vec() },
                        a: format!("{}{}", transition.a, task),
                        s_sprime: prod_sprime,
                        reward: transition.rewards
                    });
                    let mod_transition = ProdMDPTransition{
                        s: StatePair { s: 999, q: q_mod.to_vec() },
                        a: "tau".to_string(),
                        s_sprime: prod_sprime_mod,
                        reward: 0.0
                    };
                    match modification_transitions.iter().find(|x| x.s == mod_transition.s) {
                        None => {modification_transitions.push(mod_transition)}
                        Some(_) => {}
                    }
                } else {
                    product_transitions.push(ProdMDPTransition {
                        s: StatePair { s: current_state.s, q: current_state.q.to_vec() },
                        a: format!("{}{}", transition.a, task),
                        s_sprime: prod_sprime,
                        reward: transition.rewards
                    });
                }
            }
        }
        product_transitions.append(&mut modification_transitions);
        (next_initial,product_transitions,additional_states)
    }

    pub fn graph_product_dfa_product_mdp(&self) -> Graph<String,String> {
        let mut graph: Graph<String, String> = Graph::new();
        for state in self.states.iter() {
            graph.add_node(format!("({},{:?})", state.s, state.q));
        }
        for transition in self.transitions.iter() {
            let origin_index: NodeIndex = graph.node_indices().
                find(|x| graph[*x] == format!("({},{:?})",transition.s.s, transition.s.q)).unwrap();
            for sprime in transition.s_sprime.iter() {
                let destination_index: NodeIndex = graph.node_indices().
                    find(|x| graph[*x] == format!("({},{:?})", sprime.s.s, sprime.s.q)).unwrap();
                graph.add_edge(origin_index, destination_index, transition.a.to_string());
            }
        }
        graph
    }

    pub fn reachable_from_initial(&self, g: &Graph<String,String>) -> HashMap<StatePair, bool> {
        let initial: NodeIndex = g.node_indices().
            find(|x| g[*x] == format!("({},{:?})", self.initial.s, self.initial.q)).unwrap();
        let mut reachable: HashMap<StatePair, bool> = HashMap::new();
        for s in self.states.iter() {
            let s_node_index: NodeIndex = g.node_indices().
                find(|x| g[*x] == format!("({},{:?})", s.s, s.q)).unwrap();
            let truth = has_path_connecting(g, initial, s_node_index, None);
            reachable.insert(s.clone(), truth);
        }
        reachable
    }

    pub fn prune_states(&mut self, reachable_states: &HashMap<StatePair, bool>) {
        let mut reachable: Vec<StatePair> = Vec::new();
        for state in self.states.iter() {
            let reachable_flag = reachable_states.get(state).unwrap();
            if *reachable_flag {
                reachable.push(state.clone());
            }
        }
        self.states = reachable;
    }

    pub fn prune_graph(graph: &mut Graph<String, String>, reachable_states: &HashMap<StatePair, bool>) {
        for (state, _b) in reachable_states.iter().filter(|(s,x)| !**x) {
            let delete = graph.node_indices().
                find(|x| graph[*x] == format!("({},{:?})", state.s, state.q)).unwrap();
            graph.remove_node(delete);
        }
    }

    fn find_final(q: &Vec<u32>, dfas: &Vec<(usize,DFA)>) -> bool {
        for (j, state) in q.iter().enumerate() {
            if dfas[j as usize].1.dead.iter().all(|x| x != state) &&
                dfas[j as usize].1.acc.iter().all(|x| x != state) {
                return false;
            }
        }
        true
    }

    pub fn default() -> ProductDFAProductMDP {
        ProductDFAProductMDP {
            states: vec![],
            initial: StatePair { s: 0, q: vec![] },
            transitions: vec![],
            labelling: vec![]
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct StatePair {
    pub s: u32,
    pub q: Vec<u32>
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProdMDPTransition {
    pub s: StatePair,
    pub a: String,
    pub s_sprime: Vec<ProdTransitionPair>,
    pub reward: f64
}

pub struct ProdMDPLabelling {
    pub s: StatePair,
    pub w: Vec<String>
}

#[derive(Debug, Clone)]
pub struct ProdLabellingPair {
    pub s: StatePair,
    pub w: Vec<String>
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProdTransitionPair {
    pub s: StatePair,
    pub p: f64
}