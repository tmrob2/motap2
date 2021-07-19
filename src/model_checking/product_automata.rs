use itertools::Itertools;
use std::collections::{HashSet};
use super::dfa;
use dfa::DFATransitions;
use super::mdp2;
use mdp2::Transition;
use regex::Regex;
use std::time::Instant;
//use std::borrow::Cow;
//use std::hash::Hash;

// -----------------------------------
// Task Automaton A1 x A2 x ... x An
// -----------------------------------
#[allow(dead_code)]
pub fn create_new_states_and_transitions<'a, 'b>(q1: &'b [CrossProdState], q2: &'b [u32], delta1: &'b [CrossProdTransition<'a>],
                                             delta2: &'b [DFATransitions<'a>], acc1: &'b [CrossProdState], dead1: &'b [CrossProdState],
                                             _jacc1: &'b [CrossProdState], init1: CrossProdState,
                                             acc2: &'b [u32], jacc2: &'b [u32], dead2: &'b [u32], init2: u32, task: i32)
    -> (Vec<CrossProdState>, Vec<CrossProdTransition<'a>>, Vec<CrossProdState>, Vec<CrossProdState>, Vec<CrossProdState>, CrossProdState){
    let mut new_states: Vec<CrossProdState> = Vec::with_capacity(q1.len() * q2.len());
    let mut new_transition: Vec<CrossProdTransition> = Vec::new();
    let mut jacc_new: Vec<CrossProdState> = Vec::new();
    let mut acc_new: Vec<CrossProdState> = Vec::new();
    let mut dead_new: Vec<CrossProdState> = Vec::new();
    let mut init_new: CrossProdState = CrossProdState { q: 0, desc: "".to_string(), trans_ix: vec![], active_ix: 0, jacc: false, acc: false, dead: false, init: false, switch_to: false, task_complete: vec![], done: false };
    //let mut switch_to_new: Vec<CrossProdState> = Vec::new();
    if delta1.is_empty() {
        // this is the case where we create the first DFA as a product DFA
        //println!("jacc1: {:?}, acc1: {:?}, dead1: {:?}, jacc2: {:?}, acc2: {:?}, dead2: {:?}", jacc1, acc1, dead1, jacc2, acc2, dead2);
        for q in q2.iter() {
            //println!("q: {}, jacc non-member: {:?}", q, jacc2.iter().all(|x| x != q));
            //println!("q: {}, acc non-member: {:?}", q, acc2.iter().all(|x| x != q));
            //println!("q: {}, dead non-member: {:?}", q, dead2.iter().all(|x| x != q));
            let active = if acc2.iter().all(|x| x != q) &&
                dead2.iter().all(|x| x != q) && *q != init2 {
                0
            } else {
                -1
            };
            let mut new_prod_state = CrossProdState {
                q: *q,
                desc: format!("q{}", q).to_string(),
                trans_ix: vec![],
                active_ix: active,
                jacc: false,
                acc: false,
                dead: false,
                init: false,
                switch_to: false,
                task_complete: vec![],
                done: false
            };
            for delta in delta2.iter().filter(|x| x.q == *q) {
                let delta_ix = delta2.iter().position(|x| x.q == *q && x.w == delta.w).unwrap();
                new_prod_state.trans_ix.push(delta_ix);
                new_transition.push(CrossProdTransition {
                    q: delta.q,
                    w: delta.w.clone(),
                    q_prime: delta.q_prime
                });
            }
            if jacc2.iter().any(|x| x == q) {
                new_prod_state.jacc = true;
                jacc_new.push(new_prod_state.clone());
            }
            if acc2.iter().any(|x| x == q) {
                new_prod_state.acc = true;
                new_prod_state.done = true;
                new_prod_state.task_complete.push(task);
                acc_new.push(new_prod_state.clone());
            }
            if dead2.iter().any(|x| x == q) {
                new_prod_state.dead = true;
                new_prod_state.done = true;
                new_prod_state.task_complete.push(task);
                dead_new.push(new_prod_state.clone());
            }
            if *q == init2 {
                new_prod_state.init = true;
                new_prod_state.switch_to = true;
                init_new = new_prod_state.clone();
            }
            //println!("{:?}", new_prod_state);
            new_states.push(new_prod_state);
        }
    } else {
        let cp = q1.iter().cartesian_product(q2.iter());
        let mut state_counter: u32 = 0;
        let mut trans_counter: usize = 0;
        //println!("jacc1: {:?}, acc1: {:?}, dead1: {:?}, jacc2: {:?}, acc2: {:?}, dead2: {:?}", jacc1, acc1, dead1, jacc2, acc2, dead2);
        for (q1,q2) in cp.clone().into_iter() {
            // if q1 is active && q2 is also active then this is an immediate violation of our one task active condition
            // and we move on
            let q2_currently_active = acc2.iter().all(|x| x != q2) && dead2.iter().all(|x| x != q2) && init2 != *q2;
            let mut new_prod_state = CrossProdState {
                q: state_counter,
                desc: format!("{}, q{}", q1.desc, q2).to_string(),
                trans_ix: vec![],
                active_ix: if q2_currently_active { task } else { q1.active_ix },
                jacc: false,
                acc: false,
                dead: false,
                init: false,
                switch_to: false,
                task_complete: q1.task_complete.to_vec(),
                done: false
            };
            // this is fine
            // a prod transition takes into account the current transitions of previous product DFA
            // and the DFA to form a cross product with
            for d1 in delta1.iter().filter(|x| x.q == q1.q) {
                for d2 in delta2.iter().filter(|x| x.q == *q2) {
                    //let start = Instant::now();
                    // if q1 is active, and q2' becomes active then this means that a transition was taken
                    // which belongs to another task, violating the one task active condition
                    let q2_active = acc2.iter().all(|x| *x != d2.q_prime) && dead2.iter().all(|x| *x != d2.q_prime) && init2 != d2.q_prime;
                    let q1_active = acc1.iter().all(|x| x.q != d1.q_prime) && dead1.iter().all(|x| x.q != d1.q_prime) && init1.q != d1.q_prime;
                    let current_automaton_pos_activity = q1.active_ix == -1 && q1_active || q1.active_ix >=0 && q1_active || q1.active_ix >=0 && !q1_active;
                    let task_2_pos_activity = q2_currently_active && !q2_active || !q2_currently_active && q2_active || q2_currently_active;
                    // this is fine regardless of if q1 is active
                    let intersection = intersect(&d1.w[..], &d2.w[..]);
                    // if either the current state is active or the transition state is active
                    if !intersection.is_empty() {
                        let to_state = cp.clone().position(|(x1,x2)| x1.q == d1.q_prime && *x2 == d2.q_prime).unwrap();
                        //println!("to q_ix: {:?}", to_state);
                        if !(current_automaton_pos_activity && task_2_pos_activity) {
                            /*if task == 2 {
                                println!("t1: {}, t2: {}, ({},{}) -> ({},{}): ({:?})", current_automaton_pos_activity, task_2_pos_activity, q1.q,q2, d1.q_prime, d2.q_prime, intersection);
                            }*/
                            let new_trans = CrossProdTransition {
                                q: state_counter,
                                w: intersection,
                                q_prime: to_state as u32
                            };
                            new_transition.push(new_trans);
                            new_prod_state.trans_ix.push(trans_counter);
                            trans_counter += 1;
                        }
                    }
                    //let duration = start.elapsed();
                    //println!("time to determining if there is an intersectin of words: {:?}", duration);
                }
            }
            if jacc2.iter().any(|x| x == q2) {
                if q1.acc || q1.dead || q1.init {
                    new_prod_state.jacc = true;
                    jacc_new.push(new_prod_state.clone());
                };
            } else if q1.jacc {
                new_prod_state.jacc = true;
                jacc_new.push(new_prod_state.clone());
            }
            if acc2.iter().any(|x| x == q2) {
                if q1.acc || q1.dead || q1.init {
                    new_prod_state.acc = true;
                    new_prod_state.task_complete.push(task);
                    if q1.done {
                        new_prod_state.done = true;
                    }
                    acc_new.push(new_prod_state.clone());
                }
            } else if q1.acc {
                new_prod_state.acc = true;
                acc_new.push(new_prod_state.clone());
            }
            if dead2.iter().any(|x| x == q2) {
                if q1.acc || q1.dead || q1.init {
                    new_prod_state.dead = true;
                    new_prod_state.task_complete.push(task);
                    if q1.done {
                        new_prod_state.done = true;
                    }
                    dead_new.push(new_prod_state.clone())
                }
            } else if q1.dead {
                new_prod_state.dead = true;
                dead_new.push(new_prod_state.clone())
            }
            if *q2 == init2 || q1.switch_to {
                if q1.dead || q1.acc || q1.switch_to {
                    new_prod_state.switch_to = true;
                }
            }
            if q1.init && *q2 == init2 {
                new_prod_state.init = true;
                //new_prod_state.switch_to = true;
                init_new = new_prod_state.clone();
            } else {
                new_prod_state.init = false;
            }
            //println!("{:?}", new_prod_state);
            new_states.push(new_prod_state);
            state_counter += 1;
        }
    }
    (new_states, new_transition, jacc_new, acc_new, dead_new, init_new)
}

#[allow(dead_code)]
fn intersect<'a, 'b>(w1: &'b [&'a HashSet<&'a str>], w2: &'b [&'a HashSet<&'a str>]) -> Vec<&'a HashSet<&'a str>> {
    let intersection: Vec<_> = w1.iter().filter(|x| w2.iter().any(|y| *x == y)).map(|x| *x).collect();
    intersection
}

#[allow(dead_code)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CrossProdState {
    pub q: u32,
    pub desc: String,
    pub trans_ix: Vec<usize>,
    pub active_ix: i32,
    pub jacc: bool,
    pub acc: bool,
    pub dead: bool,
    pub init: bool,
    pub switch_to: bool,
    pub task_complete: Vec<i32>,
    pub done: bool,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct CrossProdTransition<'a> {
    pub q: u32,
    pub w: Vec<&'a HashSet<&'a str>>,
    pub q_prime: u32
}

#[allow(dead_code)]
pub struct CrossProdDFA<'a> {
    pub states: Vec<CrossProdState>,
    pub delta: Vec<CrossProdTransition<'a>>,
    pub acc: Vec<CrossProdState>,
    pub dead: Vec<CrossProdState>,
    pub jacc: Vec<CrossProdState>
}

// -------------------------------------
// Local Product M x A1 x A2 x ... x An
// -------------------------------------

#[allow(dead_code)]
pub fn create_local_prod_states<'a>(mdp_states: &'a [u32], dfa_states: &'a [CrossProdState], mdp_init: u32)
    -> Vec<LocalProdState<'a>> {
    let mut prod_states: Vec<LocalProdState> = Vec::with_capacity(mdp_states.len() * dfa_states.len());
    let cp = mdp_states.iter().cartesian_product(dfa_states.iter());
    for (k, (s,q)) in cp.into_iter().enumerate() {
        // determine if there is a task active
        let mut new_state = LocalProdState {
            s: *s,
            q: q.q,
            ix: k,
            desc: q.desc.to_string(),
            active_ix: q.active_ix,
            dfa_trans_ix: &q.trans_ix,
            dfa_jacc: q.jacc,
            dfa_acc: q.acc,
            dfa_dead: q.dead,
            done: q.done,
            switch_from: false,
            switch_to: false,
            tasks_complete: q.task_complete.to_vec()
        };
        if *s == mdp_init && q.switch_to {
            new_state.switch_to = true;
            new_state.switch_from = true;
            //println!("{},{:?}", s, q.desc);
        }
        prod_states.push(new_state);
    }
    prod_states
}

#[allow(dead_code)]
pub fn create_local_prod_transitions<'a>(states: &'a [LocalProdState], mdp_labelling: &[mdp2::MDPLabellingPair], mdp_transitions: &'a [Transition],
                                         cross_prod_dfa_trans: &'a [CrossProdTransition], num_tasks: i32) -> Vec<LocalProdTransitions> {
    let mut local_prod_trans: Vec<LocalProdTransitions> = Vec::new();
    for state in states.iter() {
        //println!("###########\ns: ({},{}), {}\n###########", state.s, state.desc, state.dfa_jacc);
        let tasks: Vec<i32> = (0..num_tasks).collect();
        let tasks_remaining: Vec<i32> = if state.tasks_complete == tasks {
            Vec::new()
        } else {
            tasks.clone().into_iter().filter(|x| state.tasks_complete.iter().all(|y| x != y)).map(|x| x).collect()
        };
        //println!("#############\nstate: ({},{}), task active:{:?}, complete: {:?}, {}, tasks remaining: {:?}\n###########", state.s, state.desc, state.active_ix, state.tasks_complete, state.tasks_complete == tasks, tasks_remaining);
        let task_loop: Vec<i32> = if state.active_ix == -1 {
            if tasks == state.tasks_complete {
                tasks
            } else {
                tasks_remaining
            }
        } else {
            vec![state.active_ix]
        };
        for task in task_loop {
            for transition in mdp_transitions.iter().filter(|x| x.s == state.s) {
                //println!("t: {:?}\n###########", transition);
                let mut loc_prod_trans_to: Vec<LocalProductTransitionPair> = Vec::new();
                let start = Instant::now();
                for sprime in transition.s_prime.iter() {
                    let label = mdp_labelling.iter().find(|x| x.s == sprime.s).unwrap();
                    for h in label.w.iter() {
                        //println!("h:{:?}, searching for: {:?}", h, state.dfa_trans_ix);
                        let new_q: Vec<(i32, u32, usize)> = valid_transition(h, sprime.s, states, cross_prod_dfa_trans, &state.dfa_trans_ix[..], task);
                        //println!("new_q: {:?}", new_q);
                        for (task_rtn, _q, z) in new_q.iter() {
                            if *task_rtn == task || *task_rtn == -1 {
                                loc_prod_trans_to.push(LocalProductTransitionPair {
                                    s: *z,
                                    p: sprime.p
                                });
                            }
                        }
                    }
                }
                if !loc_prod_trans_to.is_empty() {
                    let new_transition = LocalProdTransitions {
                        sq: state.ix,
                        a: format!("{}{}", transition.a, task),
                        sq_prime: loc_prod_trans_to,
                        reward: transition.rewards
                    };
                    //println!("t: {:?}", new_transition);
                    local_prod_trans.push(new_transition);
                }
                let duration = start.elapsed();
                println!("time taken to loop through labels and find a valid transition: {:?}", duration);
            }
        }
    }
    local_prod_trans
}

#[allow(dead_code)]
pub fn reachable_states<'a, 'b>(states: &'b [LocalProdState<'a>], transitions: &'b [LocalProdTransitions], init_state: usize) -> Vec<LocalProdState<'a>> {
    let mut visited: Vec<bool> = vec![false; states.len()];
    visited[init_state] = true;
    let mut stack = Vec::new();
    stack.push(&states[init_state]);
    while !stack.is_empty() {
        let state = stack.pop().unwrap();
        for t in transitions.iter().filter(|x| x.sq == state.ix) {
            //print!("s: ({},{}) -> ", state.s, state.q);
            for sprime in t.sq_prime.iter() {
                let sprime_state = &states[sprime.s];
                //print!("s': ({},{})", sprime_state.s, sprime_state.q);
                //print!(", ");
                if !visited[sprime.s] {
                    visited[sprime.s] = true;
                    stack.push(sprime_state);
                }
            }
            //println!();
        }
    }
    let mut reachable_states: Vec<LocalProdState> = Vec::with_capacity(visited.iter().filter(|x| **x).count());
    let mut state_ix: usize = 0;
    for (k, s_tru) in visited.iter().enumerate() {
        if *s_tru {
            let mut new_state = states[k].clone();
            new_state.ix = state_ix;
            new_state.dfa_trans_ix = states[k].dfa_trans_ix;
            reachable_states.push(new_state);
            state_ix += 1;
        }
    }
    reachable_states
}

#[allow(dead_code)]
fn valid_transition(h: &&HashSet<&str>, mdp_sprime: u32, states: &[LocalProdState],
                    cross_prod_dfa_trans: &[CrossProdTransition], trans_ix: &[usize], task: i32) -> Vec<(i32, u32,usize)> {
    let mut qprime_v: HashSet<(i32, u32, usize)> = HashSet::new();
    let re = Regex::new(r"\(([a-z])\)").unwrap();
    if h.is_empty() {
        for t_ix in trans_ix.iter() {
            let delta = &cross_prod_dfa_trans[*t_ix];
            let sqprime_ix = states.iter().position(|x| x.s == mdp_sprime && x.q == delta.q_prime).unwrap();
            //println!("delta words: {:?}", delta.w);
            for h_w in delta.w.iter() {
                if h_w.is_empty() {
                    //println!("q': {:?}", delta.q_prime);
                    qprime_v.insert((states[sqprime_ix].active_ix,delta.q_prime,sqprime_ix));
                }
            }
        }
    } else {
        for w in h.iter() {
            //println!("tix all: {:?}", trans_ix);
            for t_ix in trans_ix.iter() {
                let delta = &cross_prod_dfa_trans[*t_ix];
                //println!("tix: {}, delta: {:?}", t_ix, delta);
                let sqprime_ix = states.iter().position(|x| x.s == mdp_sprime && x.q == delta.q_prime).unwrap();
                //println!("jacc: {}", states[sqprime_ix].dfa_jacc);
                //println!("s': ({},{}), {}", states[sqprime_ix].s, states[sqprime_ix].desc, states[sqprime_ix].dfa_jacc);
                let hash_word = re.replace(w, format!("({})", task));
                //println!("hash word: {:?}", hash_word);
                //println!("delta words: {:?}", delta.w);
                for h_w in delta.w.iter() {
                    for aut_w in h_w.iter() {
                        if hash_word == *aut_w {
                            //println!("q': {:?}", delta.q_prime);
                            qprime_v.insert((task, delta.q_prime, sqprime_ix));
                        }
                    }
                }
            }
        }
    }
    //println!("q':{:?}", qprime_v);
    let qprime_rtn_v: Vec<(i32, u32, usize)> = qprime_v.into_iter().collect();
    qprime_rtn_v
}

#[allow(dead_code)]
pub struct LocalProduct<'a> {
    pub states: Vec<LocalProdState<'a>>,
    pub init: usize,
    pub transitions: Vec<LocalProdTransitions>
}

#[allow(dead_code)]
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LocalProdState<'a> {
    pub s: u32,
    pub q: u32,
    pub ix: usize,
    pub desc: String,
    pub active_ix: i32,
    pub dfa_trans_ix: &'a Vec<usize>,
    pub dfa_jacc: bool,
    pub dfa_acc: bool,
    pub dfa_dead: bool,
    pub done: bool,
    pub switch_from: bool,
    pub switch_to: bool,
    pub tasks_complete: Vec<i32>
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct LocalProdTransitions {
    pub sq: usize,
    pub a: String,
    pub sq_prime: Vec<LocalProductTransitionPair>,
    pub reward: f64
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct LocalProductTransitionPair {
    pub s: usize,
    pub p: f64
}