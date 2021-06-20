use super::mdp2;
use mdp2::*;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use crate::model_checking::mdp2::{MDP2, MDPLabellingPair};
use std::hash::Hash;
use std::iter::FromIterator;

#[derive(Debug, Clone)]
pub struct DFA2Transitions<'a> {
    pub q: u32,
    pub w: Vec<&'a &'a HashSet<&'a str>>,
    pub q_prime: u32
}

pub struct DFA2<'a> {
    pub states: Vec<u32>,
    pub sigma: &'a Vec<&'a str>,
    pub initial: u32,
    pub delta: &'a Vec<DFA2Transitions<'a>>,
    pub acc: Vec<u32>,
    pub dead: Vec<u32>
}

/// Should be exact words or contains words but not the conjunction
/// ```exact words``` e.g. [HashSet["r"], HashSet["s1","r"]]
/// ```contains_word``` e.g. ("h",2) first member of tuple is a letter and the second is a size, we search
/// for all words containing "h" up to size 2
pub fn construct_dfa_transition<'a, 'b>(ps: &'a [HashSet<&'a str>], exact_words: Option<&'b[HashSet<&'b str>]>, q: u32, q_prime: u32,
                                        contains_word: Option<(Option<&'b str>, usize)>, input: &'a mut HashMap<(u32,u32), Vec<&'a HashSet<&'a str>>>,
                                        not_words: Option<&'a[&'a HashSet<&'a str>]>)
                                        -> &'a mut HashMap<(u32,u32), Vec<&'a HashSet<&'a str>>> {
    match exact_words {
        None => {
            match contains_word {
                None => {panic!("Should be exact words or contains word but not both and not neither")}
                Some((x, size)) => {
                    match x {
                        None => {
                            let words: Vec<_> = ps.iter().filter(|y| y.len() <= size).collect();
                            match not_words {
                                None => input.insert( (q,q_prime), words),
                                Some(y) => {
                                    let mut new_words: Vec<&HashSet<&str>> = Vec::new();
                                    for w in words.iter() {
                                        if !y.iter().any(|z|  z == w) {
                                            new_words.push(w);
                                        }
                                    }
                                    input.insert((q,q_prime), new_words)
                                }
                            };
                        }
                        Some(match_word) => {
                            let words: Vec<_> = ps.iter().
                                filter(|y| y.iter().any(|&z| z == match_word) && y.len() <= size).collect();
                            match not_words {
                                None => input.insert( (q,q_prime), words),
                                Some(y) => {
                                    let mut new_words: Vec<&HashSet<&str>> = Vec::new();
                                    for w in words.iter() {
                                        if !y.iter().any(|z|  z == w) {
                                            new_words.push(w);
                                        }
                                    }
                                    input.insert((q,q_prime), new_words)
                                }
                            };
                        }
                    }
                }
            }
        }
        Some(x) => {
            let mut words = Vec::new();
            for w in x.iter() {
                let ps_slice = ps.iter().find(|&y| y == w).unwrap();
                words.push(ps_slice);
            }
            input.insert( (q,q_prime), words);
        }
    }
    input
}
// ----------------------
// MDP x DFA
// ----------------------

pub fn create_states(sbar: &[u32], qbar: &[u32]) -> Vec<DFA2ModelCheckingPair> {
    let mut prod_states: Vec<DFA2ModelCheckingPair> =
        vec![DFA2ModelCheckingPair{ s: 0, q: 0 }; sbar.len() * qbar.len()];
    let cp = sbar.iter().cartesian_product(qbar.iter());
    for (k,(s,q)) in cp.into_iter().enumerate() {
        prod_states[k] = DFA2ModelCheckingPair{ s: *s, q: *q }
    }
    prod_states
}

pub fn create_prod_transitions<'a>(mdp: &'a MDP2, dfa: &'a DFA2, states: &'a[DFA2ModelCheckingPair], transitions: &'a mut Vec<DFA2ProductTransition> )
    -> &'a mut Vec<DFA2ProductTransition> {
    for state in states.iter() {
        for transition in mdp.transitions.iter()
            .filter(|x| x.s == state.s) {
            let mut t = DFA2ProductTransition {
                sq: DFA2ModelCheckingPair { s: state.s, q: state.q },
                a: transition.a.to_string(),
                sq_prime: vec![],
                reward: 0.0
            };
            t.reward = transition.rewards; // this is the reward inherited from the MDP
            //println!("s:{:?}", state);
            for sprime in transition.s_prime.iter() {
                let label = mdp.labelling.iter().find(|x| x.s == sprime.s).unwrap();
                let q_prime = dfa.delta.iter().find(|x| x.q == state.q && x.w.iter().
                    any(|y| label.w.iter().any(|z| z == *y)));

                match q_prime {
                    None => {panic!("No transition found: {:?}", state);}
                    Some(x) => {
                        //println!("q': {:?}", q_prime);
                        t.sq_prime.push(DFA2TransitionPair {
                            state: DFA2ModelCheckingPair {s: sprime.s, q: x.q_prime},
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

pub fn reachable_from_initial<'a>(states: &'a [DFA2ModelCheckingPair], transitions: &'a mut Vec<DFA2ProductTransition>, initial: &'a DFA2ModelCheckingPair,
                                  reachable: &'a mut Vec<&'a DFA2ModelCheckingPair>)
    -> (&'a mut Vec<&'a DFA2ModelCheckingPair>,&'a mut Vec<DFA2ProductTransition>) {
    let mut visited: Vec<bool> = vec![false; states.len()];
    let mut stack: Vec<&DFA2ModelCheckingPair> = Vec::new();
    let initial_ref = states.iter().position(|x| x == initial).unwrap();
    visited[initial_ref] = true;
    stack.push(&states[initial_ref]);
    while !stack.is_empty() {
        let new_state = stack.pop().unwrap();
        for t in transitions.iter().filter(|x| x.sq == *new_state) {
            for sprime in t.sq_prime.iter() {
                let new_state_index = states.iter().position(|x| *x == sprime.state).unwrap();
                if !visited[new_state_index] {
                    stack.push(&states[new_state_index]);
                    visited[new_state_index] = true;
                }
            }
        }
    }
    reachable.truncate(visited.iter().enumerate().filter(|(i,x)| **x).count());
    for (i, truth) in visited.iter().enumerate() {
        if *truth {
            reachable.push(&states[i]);
        }
    }
    (reachable, transitions)
}

pub fn create_labelling<'a>(states: &'a mut Vec<&'a DFA2ModelCheckingPair>, mdp_labels: &'a [MDPLabellingPair], prod_labels: &'a mut Vec<DFA2ProductLabellingPair<'a>>)
    -> (&'a mut Vec<&'a DFA2ModelCheckingPair>,&'a mut Vec<DFA2ProductLabellingPair<'a>>) {
    for state in states.iter() {
        for l in mdp_labels.iter().filter(|x| x.s == state.s) {
            prod_labels.push(DFA2ProductLabellingPair {
                sq: &state,
                w: &l.w
            })
        }
    }
    (states, prod_labels)
}

pub fn modify_complete<'a>(states: &'a mut Vec<&'a DFA2ModelCheckingPair>, transitions: &'a mut Vec<DFA2ProductTransition>,
                           dfa: &'a DFA2, compl_l: &'a HashSet<&'a str>,
                           add_s: &'a mut HashSet<DFA2ModelCheckingPair>, add_t: &'a mut Vec<DFA2ProductTransition>,
                           add_l: &'a mut Vec<NonRefDFA2ProductLabellingPair<'a>>)
    -> (&'a mut HashSet<DFA2ModelCheckingPair>, &'a mut Vec<DFA2ProductTransition>, &'a mut Vec<NonRefDFA2ProductLabellingPair<'a>>,
        &'a mut Vec<&'a DFA2ModelCheckingPair>, &'a mut Vec<DFA2ProductTransition>) {
    for state in states.iter().
        filter(|x| dfa.acc.iter().all(|y| *y != x.q)) {
        for transition in transitions.iter_mut().
            filter(|x| x.sq == **state && x.sq_prime.iter().
                any(|y| dfa.acc.iter().any(|yy| *yy == y.state.q))) {
            for sq_prime in transition.sq_prime.iter_mut().
                filter(|x| dfa.acc.iter().any(|y| *y == x.state.q)) {
                if add_t.iter().all(|x| x.sq != DFA2ModelCheckingPair {
                    s: 999,
                    q: sq_prime.state.q
                } && *sq_prime != DFA2TransitionPair { state: DFA2ModelCheckingPair { s: sq_prime.state.s, q: sq_prime.state.q }, p: 1.0 }) {
                    add_t.push(DFA2ProductTransition {
                        sq: DFA2ModelCheckingPair { s: 999, q: sq_prime.state.q },
                        a: "tau".to_string(),
                        sq_prime: vec![DFA2TransitionPair { state: DFA2ModelCheckingPair { s: sq_prime.state.s, q: sq_prime.state.q }, p: 1.0 }],
                        reward: 0.0
                    });
                }
                sq_prime.state.s = 999;
                let new_state = DFA2ModelCheckingPair { s: 999, q: sq_prime.state.q };
                match add_l.iter().find(|x| x.sq == new_state) {
                    None => {
                        //let comp_state = additional_states.iter().find(|y| y.q == )
                        add_l.push(NonRefDFA2ProductLabellingPair {
                            sq: DFA2ModelCheckingPair {s: 999, q: sq_prime.state.q},
                            w: vec![compl_l]
                        });
                    },
                    Some(x) => {}
                }
                add_s.insert(new_state);
            }
        }
    }
    (add_s, add_t, add_l, states, transitions)
}

pub fn edit_labelling<'a, 'b>(labelling: &'a mut Vec<DFA2ProductLabellingPair<'a>>, mod_labelling: &'a mut Vec<DFA2ModLabelPair<'a>>,
                              dfa: &'a DFA2, mdp: &'a MDP2, initial: &'b DFA2ModelCheckingPair, ini: &'a str, fai: &'a str, suc: &'a str)
    -> &'a mut Vec<DFA2ModLabelPair<'a>> { //&'a mut [DFA2ProductLabellingPair] {
    for l in labelling.iter() {
        if l.sq == initial {
            let mut h_new = l.w.iter().map(|&x| new_hash(x, ini)).collect::<Vec<HashSet<&'a str>>>();
            mod_labelling.push(DFA2ModLabelPair { sq: l.sq, w: h_new })
        } else if dfa.dead.iter().any(|x| *x == l.sq.q) && l.sq.s == mdp.initial {
            let mut h_new = l.w.iter().map(|&x| new_hash(x, fai)).collect::<Vec<HashSet<&'a str>>>();
            mod_labelling.push(DFA2ModLabelPair { sq: l.sq, w: h_new })
        } else if dfa.acc.iter().any(|x| *x == l.sq.q) && l.sq.s == mdp.initial {
            let mut h_new = l.w.iter().map(|&x| new_hash(x, suc)).collect::<Vec<HashSet<&'a str>>>();
            mod_labelling.push(DFA2ModLabelPair { sq: l.sq, w: h_new })
        }
        else {
            let mut h_new = l.w.iter().map(|&x| x.clone()).collect::<Vec<HashSet<&'a str>>>();
            mod_labelling.push(DFA2ModLabelPair { sq: l.sq, w: h_new })
        }
    }
    mod_labelling
}

fn new_hash<'a>(h: &'a HashSet<&'a str>, val: &'a str) -> HashSet<&'a str> {
    let mut new_h = h.clone();
    new_h.insert(val);
    new_h
}

pub fn create_local_product<'a>(mdp: &'a MDP2, dfa: &'a DFA2, ini: &'a str, fai: &'a str, suc: &'a str,
                                compl_hsh: &'a HashSet<&'a str>, b_state_space: &'a [DFA2ModelCheckingPair], init_state: &'a DFA2ModelCheckingPair,
                                reach_v: &'a mut Vec<&'a DFA2ModelCheckingPair>, trans_v: &'a mut Vec<DFA2ProductTransition>,
                                label_v: &'a mut Vec<DFA2ProductLabellingPair<'a>>, mod_l_v: &'a mut Vec<DFA2ModLabelPair<'a>>,
                                add_s: &'a mut HashSet<DFA2ModelCheckingPair>, add_t: &'a mut Vec<DFA2ProductTransition>,
                                add_l: &'a mut Vec<NonRefDFA2ProductLabellingPair<'a>>)
                                -> (&'a mut Vec<&'a DFA2ModelCheckingPair>,&'a mut Vec<DFA2ProductTransition>, &'a mut Vec<DFA2ModLabelPair<'a>>) {
    // transitions
    let mut trans_v = create_prod_transitions(&mdp, &dfa, &b_state_space[..], trans_v);
    let (mut reach_v, mut trans_v)  = reachable_from_initial(&b_state_space[..], trans_v, &init_state, reach_v);
    // product labelling
    let (mut reach_v, mut label_v) = create_labelling(reach_v, &mdp.labelling[..], label_v);
    // modifications
    let (add_s, add_t, add_l, mut reach_v, mut trans_v) =
        modify_complete(reach_v, trans_v, &dfa, compl_hsh, add_s, add_t, add_l);
    // ---------------------------------------------------
    // Update Labels, Transitions, and Modification States
    // ---------------------------------------------------
    for l in add_l.iter() {
        label_v.push(DFA2ProductLabellingPair{ sq: &l.sq, w: &l.w })
    }
    for t in add_t.into_iter() {
        trans_v.push(DFA2ProductTransition{
            sq: t.sq,
            a: t.a.to_string(),
            sq_prime: t.sq_prime.to_vec(),
            reward: t.reward
        });
    }
    for s in add_s.iter() {
        reach_v.push(s)
    }
    let mut mod_l_v = edit_labelling(label_v, mod_l_v, &dfa, &mdp, &init_state, ini, fai, suc);

    (reach_v, trans_v, mod_l_v)
}

#[derive(Debug, Clone)]
pub struct ProductMDP2<'a> {
    pub states: Vec<&'a DFA2ModelCheckingPair>,
    pub initial: DFA2ModelCheckingPair,
    pub transitions: &'a Vec<DFA2ProductTransition>,
    pub labelling: &'a Vec<DFA2ModLabelPair<'a>>,
}

#[derive(Debug, Clone)]
pub struct DFA2ProductTransition {
    pub sq: DFA2ModelCheckingPair,
    pub a: String,
    pub sq_prime: Vec<DFA2TransitionPair>,
    pub reward: f64
}

#[derive(Debug, Clone)]
pub struct DFA2ProductLabellingPair<'a> {
    pub sq: &'a DFA2ModelCheckingPair,
    pub w: &'a Vec<&'a HashSet<&'a str>>
}

#[derive(Debug, Clone)]
pub struct DFA2ModLabelPair<'a> {
    pub sq: &'a DFA2ModelCheckingPair,
    pub w: Vec<HashSet<&'a str>>
}

#[derive(Debug)]
pub struct NonRefDFA2ProductLabellingPair<'a> {
    pub sq: DFA2ModelCheckingPair,
    pub w: Vec<&'a HashSet<&'a str>>
}

/// A DFA transition pair is a state and a probability of transitioning to this state
#[derive(Debug, Clone, PartialEq)]
pub struct DFA2TransitionPair {
    pub state: DFA2ModelCheckingPair,
    pub p: f64
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Copy)]
pub struct DFA2ModelCheckingPair {
    pub s: u32,
    pub q: u32,
}

#[derive(Clone, Debug)]
pub struct TeamInput<'a> {
    pub agent: usize,
    pub task: usize,
    pub product: ProductMDP2<'a>,
    pub dead: &'a Vec<u32>,
    pub acc: &'a Vec<u32>
}