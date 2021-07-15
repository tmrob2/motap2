use itertools::Itertools;
use std::collections::{HashSet};
//use std::iter::{FromIterator};
use super::mdp2;
use mdp2::*;

#[derive(Debug, Clone)]
pub struct DFATransitions<'a> {
    pub q: u32,
    pub w: Vec<&'a HashSet<&'a str>>,
    pub q_prime: u32
}

#[derive(Debug)]
pub struct DFA<'a> {
    pub states: Vec<u32>,
    pub initial: u32,
    pub delta: Vec<DFATransitions<'a>>,
    pub acc: Vec<u32>,
    pub dead: Vec<u32>,
    pub jacc: Vec<u32>
}

#[derive(Clone, Debug, Default)]
pub struct DFAProductMDP<'a> {
    pub states: Vec<DFAModelCheckingPair<'a>>,
    pub initial: DFAModelCheckingPair<'a>,
    pub transitions: Vec<DFAProductTransition<'a>>,
    //pub labelling: Vec<DFAProductLabellingPair<'a>>,
}

/*impl <'a>DFAProductMDP<'a> {
    fn default() -> DFAProductMDP<'a> {
        DFAProductMDP {
            states: vec![],
            initial: Default::default(),
            transitions: vec![],
            //labelling: vec![]
        }
    }
}*/

/// To create the Local Product staets M x A, we require the cartesian product of all (s,q) from S x Q
#[allow(dead_code)]
pub fn create_states<'a>(sbar: &'a [u32], qbar: &'a [u32]) -> Vec<DFAModelCheckingPair<'a>> {
    let mut prod_states: Vec<DFAModelCheckingPair> =
        vec![DFAModelCheckingPair{ state: ProdState { s: 0, q: 0 }, w: vec![] }; sbar.len() * qbar.len()];
    let cp = sbar.iter().cartesian_product(qbar.iter());
    for (k,(s,q)) in cp.into_iter().enumerate() {
        prod_states[k] = DFAModelCheckingPair{ state: ProdState {s: *s, q: *q}, w: vec![] }
    }
    prod_states
}

/// Transition labelling in the Local Product is done according to the MDP transition and the function
/// delta(q, w) where w = L(s'), and s in S from the MDP M.
#[allow(dead_code)]
pub fn create_prod_transitions<'a, 'b>(mdp: &'a MDP2, dfa: &'a DFA, states: &'b [DFAModelCheckingPair<'a>])
                                   -> Vec<DFAProductTransition<'a>> {
    let mut transitions: Vec<DFAProductTransition> = Vec::new();
    for state in states.iter() {
        for transition in mdp.transitions.iter()
            .filter(|x| x.s == state.state.s) {
            let mut t = DFAProductTransition {
                sq: DFAModelCheckingPair { state: ProdState { s: state.state.s, q: state.state.q }, w: vec![] },
                a: transition.a.to_string(),
                sq_prime: vec![],
                reward: 0.0
            };
            t.reward = transition.rewards; // this is the reward inherited from the MDP
            //println!("s:{:?}", state);
            for sprime in transition.s_prime.iter() {
                let label = mdp.labelling.iter().find(|x| x.s == sprime.s).unwrap();
                let q_prime = dfa.delta.iter().find(|x| x.q == state.state.q && x.w.iter().
                    any(|y| label.w.iter().any(|z| z == y)));

                match q_prime {
                    None => {panic!("No transition found: {:?}", state);}
                    Some(x) => {
                        //println!("q': {:?}", q_prime);
                        t.sq_prime.push(DFATransitionPair {
                            state: DFAModelCheckingPair { state: ProdState { s: sprime.s, q: x.q_prime }, w: vec![] },
                            p: sprime.p
                        });
                    }
                }
            }
            if t.sq_prime.is_empty() {
                panic!("state: {:?}", state);
            }
            transitions.push(t);
        }
    }
    transitions
}

#[allow(dead_code)]
pub fn create_reachable_states_with_labels<'a, 'b>(init_state: DFAModelCheckingPair, states: &'b [DFAModelCheckingPair],
                                  reachable_state_truth: &'b [bool],
                                  dfa: &'a DFA, hini: &'a HashSet<&'a str>, hcom: &'a HashSet<&'a str>, hsuc: &'a HashSet<&'a str>,
                                  hfai: &'a HashSet<&'a str>, mdp_init_state: &'a u32)
    -> Vec<DFAModelCheckingPair<'a>> {
    let mut reachable_states: Vec<DFAModelCheckingPair> = Vec::with_capacity(reachable_state_truth.iter().filter(|x| **x).count());
    for (i,truth) in reachable_state_truth.iter().enumerate() {
        if *truth {
            let mut state = DFAModelCheckingPair{ state: ProdState { s: states[i].state.s, q: states[i].state.q }, w: vec![] };
            if state == init_state {
                state.w = vec![hini];
            } else if dfa.jacc.iter().any(|x| *x == state.state.q) {
                state.w = vec![hcom];
            } else if state.state.s == *mdp_init_state && dfa.dead.iter().any(|x| *x == state.state.q) {
                state.w = vec![hfai]
            } else if state.state.s == *mdp_init_state && dfa.acc.iter().any(|x| *x == state.state.q) {
                state.w = vec![hsuc];
            }
            reachable_states.push(state);
        }
    }
    reachable_states
}

/// DFS to search all states reachable from intial
#[allow(dead_code)]
pub fn reachable_from_initial<'a>(initial: &'a DFAModelCheckingPair, states: &'a [DFAModelCheckingPair], transitions: &'a [DFAProductTransition]) -> Vec<bool> {
    let mut stack: Vec<&'a DFAModelCheckingPair> = Vec::new();
    let mut visited: Vec<bool> = vec![false; states.len()];
    stack.push(initial);
    let initial_index = states.iter().position(|x| x == initial).unwrap();
    visited[initial_index] = true;
    while !stack.is_empty() {
        let current_state = stack.pop().unwrap();
        // get all of the current states transitions
        for tr in transitions.iter().
            filter(|x| x.sq.state == current_state.state){
            for sprime in tr.sq_prime.iter() {
                let sprime_index = states.iter().
                    position(|x| x.state == sprime.state.state).unwrap();
                if !visited[sprime_index] {
                    visited[sprime_index] = true;
                    stack.push(&sprime.state);
                }
            }
        }
    }
    visited
}

#[derive(Clone, Debug)]
pub struct TeamInput<'a> {
    pub agent: usize,
    pub task: usize,
    pub product: &'a DFAProductMDP<'a>,
    pub dead: &'a Vec<u32>,
    pub acc: &'a Vec<u32>,
    pub jacc: &'a Vec<u32>
}

#[derive(Debug, Clone)]
pub struct DFAProductTransition<'a> {
    pub sq: DFAModelCheckingPair<'a>,
    pub a: String,
    pub sq_prime: Vec<DFATransitionPair<'a>>,
    pub reward: f64
}

/*#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DFAProductLabellingPair<'a> {
    pub sq: DFAModelCheckingPair<'a>,
    pub w: Vec<&'a HashSet<&'a str>>
}*/

/// A DFA transition pair is a state and a probability of transitioning to this state
#[derive(Debug, Clone, PartialEq)]
pub struct DFATransitionPair<'a> {
    pub state: DFAModelCheckingPair<'a>,
    pub p: f64
}

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct DFAModelCheckingPair<'a> {
    pub state: ProdState,
    pub w: Vec<&'a HashSet<&'a str>>
}

#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct ProdState {
    pub s: u32,
    pub q: u32
}


