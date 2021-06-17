use petgraph::{Graph, graph::NodeIndex};
use petgraph::algo::{kosaraju_scc, has_path_connecting, all_simple_paths};
use itertools::Itertools;
use std::collections::{HashSet, VecDeque, HashMap};
use std::iter::{FromIterator, Filter};
extern crate serde_json;
use serde::Deserialize;
use super::mdp;

use mdp::*;

#[derive(Debug, Deserialize, Clone)]
pub struct DFATransitions {
    pub q: u32,
    pub w: Vec<String>,
    pub q_prime: u32,
}

#[derive(Debug, Deserialize)]
pub struct DFA {
    pub states: Vec<u32>,
    pub sigma: Vec<String>,
    pub initial: u32,
    pub delta: Vec<DFATransitions>,
    pub acc: Vec<u32>,
    pub dead: Vec<u32>
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
        let cp= mdp.states.clone().into_iter().
            cartesian_product(dfa.states.clone().into_iter());
        for (s_m, q_a) in cp.into_iter() {
            self.states.push(DFAModelCheckingPair{ s: s_m, q: q_a});
        }
    }

    pub fn create_labelling(&mut self, mdp: &MDP) {
        for state in self.states.iter() {
            for mdp_label in mdp.labelling.iter().filter(|x| x.s == state.s) {
                self.labelling.push(DFAProductLabellingPair{
                    sq: DFAModelCheckingPair { s: state.s, q: state.q },
                    w: mdp_label.w.to_vec()
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
                //println!("s:{:?}", state);
                for sprime in transition.s_prime.iter() {
                    let label = mdp.labelling.iter().find(|x| x.s == sprime.s).unwrap();
                    //println!("s': {:?}, label: {:?}", sprime.s, label);
                    for q_prime in dfa.delta.iter().
                        filter(|x| x.q == state.q && x.w.iter().any(|y| label.w.iter().any(|z| z == y))) {
                        //println!("q': {:?}", q_prime);
                        t.sq_prime.push(DFATransitionPair {
                            state: DFAModelCheckingPair {s: sprime.s, q: q_prime.q_prime},
                            p: sprime.p
                        });
                    }
                }
                if t.sq_prime.is_empty() {
                    panic!("state: {:?}", state);
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
                    if transition_modifications.iter().all(|x| x.sq != DFAModelCheckingPair {
                        s: 999,
                        q: sq_prime.state.q
                    } && *sq_prime != DFATransitionPair{ state: DFAModelCheckingPair { s: sq_prime.state.s, q: sq_prime.state.q }, p: 1.0 }) {
                        transition_modifications.push(DFAProductTransition {
                            sq: DFAModelCheckingPair { s: 999, q: sq_prime.state.q },
                            a: "tau".to_string(),
                            sq_prime: vec![DFATransitionPair{ state: DFAModelCheckingPair { s: sq_prime.state.s, q: sq_prime.state.q }, p: 1.0 }],
                            reward: transition.reward
                        });
                    }

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

pub fn create_local_product<'a, 'b>(initial_state: &'b DFAModelCheckingPair, mdp: &'a MDP, dfa: &'a DFA) -> DFAProductMDP {
    let mut local_product: DFAProductMDP = DFAProductMDP::default();
    local_product.initial = *initial_state;
    local_product.create_states(mdp, dfa);
    local_product.create_transitions(mdp, dfa);
    /*for t in local_product.transitions.iter() {
        println!("t:{:?}", t);
    }*/
    let mut g = local_product.generate_graph();
    let initially_reachable = local_product.reachable_from_initial(&g);
    let (prune_states_indices, prune_states) : (Vec<usize>, Vec<DFAModelCheckingPair>) = local_product.prune_candidates(&initially_reachable);
    local_product.prune_states_transitions(&prune_states_indices, &prune_states);
    local_product.create_labelling(mdp);
    local_product.modify_complete(dfa);
    //println!("modifying agent: {} task: {}", i, j);
    local_product.edit_labelling(dfa, mdp);
    local_product
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

#[derive(Debug, Clone, Eq, PartialEq, Hash, Copy)]
pub struct DFAModelCheckingPair {
    pub s: u32,
    pub q: u32,
}


