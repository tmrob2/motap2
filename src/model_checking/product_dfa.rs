use petgraph::{Graph, graph::NodeIndex};
use itertools::Itertools;
extern crate serde_json;
use super::mdp;
use super::dfa;

use dfa::*;

pub struct ProductDFA {
    pub states: Vec<Vec<u32>>,
    pub sigma: Vec<String>,
    pub initial: Vec<u32>,
    pub delta: Vec<ProductDFATransition>,
    pub acc: Vec<u32>,
    pub dead: Vec<u32>
}

#[derive(Debug)]
pub struct ProductDFATransition {
    pub q: Vec<u32>,
    pub w: Vec<String>,
    pub q_prime: Vec<u32>
}

impl ProductDFA {
    pub fn create_states<'a, 'b>(dfa_new: &'b DFA, states: &'a mut Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        let mut new_states: Vec<Vec<u32>> = Vec::new();
        if states.is_empty() {
            for q in dfa_new.states.iter() {
                new_states.push(vec![*q]);
            }
        } else {
            let s = states.clone().into_iter().cartesian_product(dfa_new.states.clone().into_iter());
            for (v, val) in s.into_iter() {
                let mut new_q: Vec<u32> = v;
                new_q.push(val);
                new_states.push(new_q);
            }
        }
        new_states
    }

    pub fn create_transitions<'a, 'b>(&'a self, dfa: &'b DFA, transitions: &'b [ProductDFATransition], states: &'b Vec<Vec<u32>>) -> Vec<ProductDFATransition> {
        let mut new_transitions: Vec<ProductDFATransition> = Vec::new();
        for w in self.sigma.iter() {
            for q in states.iter() {
                if transitions.is_empty() {
                    // first task
                    for transition in dfa.delta.iter().
                        filter(|x| vec![x.q] == *q && x.w.iter().any(|x| x == w)) {
                        let mut update_transition: Option<usize> = None;
                        match new_transitions.iter().position(|x| x.q == *q && x.q_prime == vec![transition.q_prime]) {
                            None => {
                                new_transitions.push(ProductDFATransition {
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
                }
                else {
                    let old_q_len: usize = q.len()-1;
                    //println!("old q len: {:?}", &q[..old_q_len]);
                    let delta = match
                        // we are given a word w
                        // find the first transition that involves (q,w), w \in W i.e. the set of all words involved in the transition delta
                        // we should not be doing more work than necessary and iterating over all w, when we know that
                        // all W will be involved just by checking one w.
                        transitions.iter().
                        find(|x| &x.q[..] == &q[..old_q_len] && x.w.iter().any(|y| y == w)) {
                        None => { panic!("did not find q: {:?}", q);}
                        Some(x) => {
                            x
                        }
                    };
                    // A similar case arises for checking the DFA, the largest cardinality of the set of
                    // words allowing delta(q,w) -> q' can be |W|. Therefore we are required to take the
                    // intersection of words in W and and dfa.delta(q) W', all W' will enable a transition
                    // in the product automaton
                    for q2 in dfa.states.iter().filter(|x| *x == q.last().unwrap()) {
                        for t in dfa.delta.iter().
                            filter(|x| x.q == *q2 && x.w.iter().any(|y| y == w)) {
                            let mut temp_from_state: Vec<u32> = q.to_vec();
                            //temp_from_state.push(*q2);
                            let mut temp_to_state: Vec<u32> = delta.q_prime.to_vec();
                            temp_to_state.push(t.q_prime);
                            let mut add_transition: Option<ProductDFATransition> = None;
                            let mut update_transition: Option<(usize, String)> = None;
                            match new_transitions.iter_mut().
                                position(|x| x.q == temp_from_state && x.q_prime == temp_to_state) {
                                None => {
                                    // we can add the transition because it doesn't exist
                                    add_transition = Some(ProductDFATransition {
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
        new_transitions
    }

    pub fn create_automaton_graph<'a>(&'a self) -> Graph<String, String> {
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