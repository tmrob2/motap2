use super::product_dfa;
use super::mdp;
use super::dfa;
use product_dfa::*;
use mdp::*;
use dfa::DFA;
use itertools::Itertools;
use petgraph::Graph;
use petgraph::csr::NodeIndex;
use petgraph::algo::has_path_connecting;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use crate::model_checking::team_mdp::ProductDFATeamState;

#[derive(Debug, Clone)]
pub struct ProductDFAProductMDP {
    pub states: Vec<StatePair>,
    pub initial: Option<StatePair>,
    pub transitions: Vec<ProdMDPTransition>,
    pub labelling: Vec<ProdLabellingPair>
}

impl ProductDFAProductMDP { }

pub fn create_states<'a>(mdp: &'a MDP, dfa: &'a ProductDFA) -> Vec<StatePair> {
    let mut states = Vec::new();
    let cp = mdp.states.clone().into_iter().
        cartesian_product(dfa.states.clone().into_iter());
    for (m, d) in cp.into_iter() {
        states.push(StatePair {
            s: m,
            q: d
        });
    }
    states
}

pub fn create_transitions<'a, 'b>(states: &'b [StatePair], mdp: &'a MDP, dfa: &'a ProductDFA, dfas: &'a [(usize, DFA)], verbose: &'a u32) -> (Vec<ProdMDPTransition>, Vec<ProdLabellingPair>) {
    let mut transitions: Vec<ProdMDPTransition> = Vec::new();
    let mut labelling: Vec<ProdLabellingPair> = Vec::new();
    for state in states.iter() {
        let mut state_label: HashSet<String> = HashSet::new();
        let mut inactive_tasks: Vec<usize> = Vec::new();
        let mut active_task: bool = false;
        let mut active_task_index: Option<usize> = None;
        let task_elements: &[u32] = &state.q[..];
        let new_task = task_elements.iter().enumerate().
            all(|(i,x)| dfas[i].1.initial == *x ||
                dfas[i].1.acc.iter().any(|y| y == x) ||
                dfas[i].1.dead.iter().any(|y| y == x));
        if new_task {
            if task_elements.iter().enumerate().
                all(|(i, x)| dfas[i].1.dead.iter().any(|y| y == x)
                    || dfas[i].1.acc.iter().any(|y| y == x)) {
                state_label.insert("done".to_string());
                if *verbose == 3 {
                    println!("Every task has finished at state: ({},{:?})", state.s, state.q);
                }
                for k in 0..task_elements.len() {
                    inactive_tasks.push(k+1);
                }
            } else {
                if *verbose == 3 {
                    println!("There are tasks remaining: {:?}", task_elements);
                }
                for (i, q) in task_elements.iter().enumerate() {
                    if dfas[i].1.initial == *q {
                        if *verbose == 3 {
                            println!("initiating task: {}", i + 1);
                        }
                        if state.s == mdp.initial {
                            state_label.insert(format!("ini{}",i));
                        }
                        inactive_tasks.push(i + 1);
                    } else if dfas[i].1.acc.iter().any(|x| x == q) {
                        state_label.insert(format!("succ{}", i));
                    } else if dfas[i].1.dead.iter().any(|x| x == q) {
                        state_label.insert(format!("fail{}", i));
                    }
                }
            }
        } else {
            active_task = true;
            if *verbose == 3 {
                println!("There is an active task: {:?}", task_elements);
            }
            for (i, q) in task_elements.iter().enumerate() {
                if dfas[i].1.initial != *q &&
                    dfas[i].1.dead.iter().all(|y| y!= q) &&
                    dfas[i].1.acc.iter().all(|y| y != q){
                    active_task_index = Some(i + 1);
                    if *verbose== 3 {
                        println!("active task for state ({},{:?}) is {}", state.s, state.q, i + 1);
                    }
                    state_label.insert(format!("active{}", i));
                }
            }
        }
        let task_queue_len = inactive_tasks.len();
        let mut task_queue: Vec<&usize> = if active_task { vec![&0; 1]} else {vec![&0; task_queue_len]};
        if active_task {
            task_queue[0] = active_task_index.as_ref().unwrap()
        } else {
            for (i,task) in inactive_tasks.iter().enumerate() {
                task_queue[i] = task;
            }
        }
        if *verbose == 3 {
            println!("task queue: {:?}", task_queue);
        }
        for t in mdp.transitions.iter().filter(|x| x.s == state.s){
            for task in task_queue.iter() {
                let mut p_transition = ProdMDPTransition {
                    s: state.clone(),
                    a: format!("{}{}",t.a.to_string(),task),
                    s_sprime: vec![],
                    reward: t.rewards
                };
                for mdp_sprime in t.s_prime.iter() {
                    let mut q_prime: Vec<u32> = vec![];
                    for label in mdp.labelling.iter().
                        filter(|x| x.s == mdp_sprime.s) {
                        for dfa_t in dfa.delta.iter().
                            filter(|x| x.q == *state.q && x.w.iter().
                                any(|y| if !label.w.is_empty() {
                                    label.w.iter().any(|z|*y == format!("{}{}",z,task))
                                } else {
                                    true
                                })){
                            q_prime = dfa_t.q_prime.to_vec();
                        }
                        if q_prime.is_empty() {
                            if *verbose == 3 {
                                println!("No transition was found for (q: {:?}, w: {:?})", state.q, label.w);
                                println!("Available transitions are: ");
                                for dfa_t in dfa.delta.iter().
                                    filter(|q| q.q == *state.q) {
                                    println!("{:?}", dfa_t);
                                }
                            }
                        }
                    }
                    let prod_transition = ProdTransitionPair {
                        s: StatePair{ s: mdp_sprime.s, q: q_prime.to_vec() },
                        p: mdp_sprime.p
                    };
                    p_transition.s_sprime.push(prod_transition);
                }
                transitions.push(p_transition);
            }
        }
        let vec_state_labels: Vec<String> = state_label.into_iter().collect();
        labelling.push(ProdLabellingPair{ s: StatePair{ s: state.s, q: state.q.to_vec() }, w: vec_state_labels});
    }
    (transitions, labelling)
}

/// Identify those transitions with L(s) does not satisfy "succ" and L(s') does satisfy succ
pub fn identify_mod_transitions<'a>(transitions: &'a [ProdMDPTransition], labels: &'a [ProdLabellingPair]) -> (Vec<&'a ProdMDPTransition>, Vec<&'a ProdMDPTransition>) {
    let mut mod_transitions_incomplete: Vec<&'a ProdMDPTransition> = Vec::new();
    let mut mod_transitions_complete: Vec<&'a ProdMDPTransition> = Vec::new();
    for t in transitions.iter().filter(|x| x.s_sprime.iter().any(|y| labels.iter().any(|z| z.s == y.s))) {
        match labels.iter().find(|x| x.s == t.s) {
            None => {}
            Some(x) => {
                if x.w.iter().all(|y| !y.contains("succ") && !y.contains("done") && !y.contains("fail")) {
                    if x.w.iter().
                        any(|y| !y.contains("succ") &&
                            t.s_sprime.iter().
                                any(|y| labels.iter().
                                    any(|z| z.s == y.s && z.w.iter().
                                        any(|z1| z1.contains("succ"))))) {
                        /*println!("s: {:?}, l:{:?}", t.s, x.w);
                        for sprime in t.s_sprime.iter() {
                            let l_prime = labels.iter().find(|x| x.s == sprime.s).unwrap();
                            println!("s':{:?}, l':{:?}", sprime.s, l_prime);
                        }*/
                        mod_transitions_incomplete.push(t);
                    } else if x.w.iter().any(|y| y.contains("active") &&
                        t.s_sprime.iter().
                            any(|y| labels.iter().
                                any(|z| z.s == y.s && z.w.iter().
                                    any(|z1| *z1 == "done") ))){
                        /*println!("s: {:?}, l:{:?}", t.s, x.w);
                        for sprime in t.s_sprime.iter() {
                            let l_prime = labels.iter().find(|x| x.s == sprime.s).unwrap();
                            println!("s':{:?}, l':{:?}", sprime.s, l_prime);
                        }*/
                        mod_transitions_complete.push(t);
                    }
                }
            }
        }
    }
    (mod_transitions_incomplete, mod_transitions_complete)
}

/// Modifies transitions such that L(s) does not satisfy "succ" and L(s') does satisfy succ
pub fn modify_incomplete_tasks<'a, 'b>(mod_incompl_trans: &'b [&'b ProdMDPTransition], dfas: &'a [(usize, DFA)], labels: &'b [ProdLabellingPair]) -> (Vec<ProdMDPTransition>, Vec<&'b ProdMDPTransition>, Vec<StatePair>, Vec<ProdLabellingPair>) {
    let mut mod_states: HashSet<StatePair> = HashSet::new();
    let mut mod_labels: HashSet<ProdLabellingPair> = HashSet::new();
    let re = regex::Regex::new(r"\d+").unwrap();
    let mut rm_transitions: Vec<&'b ProdMDPTransition> = Vec::new();
    let mut mod_transitions_result: Vec<ProdMDPTransition> = Vec::new();
    for t in mod_incompl_trans.iter() {
        // this is a candidate of the incomplete transitions, it should always be accepted as a modifications
        // Some word processing stuff
        // determine which s' is the completing transition
        let mut sprime_mod: Vec<ProdTransitionPair> = Vec::new();
        rm_transitions.push(t);
        let mut new_state = StatePair { s: 0, q: vec![]};
        let mut mod_sprime = StatePair {s: 0, q: vec![]};
        for sprime in t.s_sprime.iter() {
            // Some word processing stuff
            //let parent_label
            let active_parent = &labels.iter().find(|x| x.s == t.s).unwrap().w.iter().find(|x| x.contains("active")).unwrap();
            let task = re.captures(active_parent).unwrap();
            let mut task_number = task.get(0).unwrap().as_str();
            let task_index: usize = task_number.parse().unwrap();
            let success_word = format!("succ{}",task_index);
            match labels.iter().find(|x| x.s == sprime.s).unwrap().w.iter().find(|x| x.contains(&success_word)) {
                None => {
                    // it is not possible for this to be the correct transition, add the current s' and move on
                    sprime_mod.push(ProdTransitionPair{ s: StatePair { s: sprime.s.s, q: sprime.s.q.to_vec() }, p: sprime.p })
                }
                Some(x) => {
                    let mut task_completed_str = "999".to_string();
                    task_completed_str.push_str(&task_number);
                    let mut task_complete_state_num: u32 = task_completed_str.parse().unwrap();
                    if dfas[task_index].1.acc.iter().any(|x| *x == sprime.s.q[task_index]) {
                        // this is an accepting transition
                        sprime_mod.push(ProdTransitionPair{ s: StatePair { s: task_complete_state_num, q: sprime.s.q.to_vec() }, p: sprime.p });
                        let trap_state = StatePair{s: task_complete_state_num, q: sprime.s.q.to_vec()};
                        /*let test_state = StatePair{s: task_complete_state_num, q: vec![2,3,0]};
                        if trap_state == test_state {
                            let parent_words = &labels.iter().find(|x| x.s == t.s).unwrap().w;
                            let all_words = &labels.iter().find(|x| x.s == sprime.s).unwrap().w;
                            println!("parents: {:?} -> words: {:?} -> w:{}, s: {},{:?},{} -> a:{} -> trap: {},{:?}, -> tau -> s'{},{:?}",
                                     parent_words, all_words, x, t.s.s, t.s.q,task_index, t.a, trap_state.s, trap_state.q, sprime.s.s, sprime.s.q);
                        }*/
                        mod_states.insert(trap_state);

                        let trap_label = ProdLabellingPair{ s: StatePair { s: task_complete_state_num, q: sprime.s.q.to_vec() }, w: vec![format!("jsucc{}", task_index)]};
                        //println!("adding s: {}, q: {:?}, task:{}", task_complete_state_num, sprime.s.q, task_index);
                        mod_labels.insert(trap_label);
                        new_state = StatePair{s: task_complete_state_num, q: sprime.s.q.to_vec()};
                        mod_sprime = StatePair{s: sprime.s.s, q: sprime.s.q.to_vec()};
                    } else {
                        // The only way the succ label is seen is if the task DFA is accepting for this particular q
                        panic!("sprime: {},{:?} is not accepting but is {}", sprime.s.s, sprime.s.q, x);
                    }
                }
            }
        }
        let mod_transition = ProdMDPTransition {
            s: StatePair { s: t.s.s, q: t.s.q.to_vec() },
            a: t.a.to_string(),
            s_sprime: sprime_mod,
            reward: t.reward
        };
        mod_transitions_result.push(mod_transition);
        match mod_transitions_result.iter().any(|x| x.s == new_state) {
            true => { }//println!("state: {:?}", new_state);}
            false => {
                let jsucc_compl_transition = ProdMDPTransition {
                    s: new_state,
                    a: "tau".to_string(),
                    s_sprime: vec![ProdTransitionPair { s: mod_sprime, p: 1.0 }],
                    reward: 0.0
                };
                //println!("adding: {:?}", jsucc_compl_transition);
                mod_transitions_result.push(jsucc_compl_transition);
            }
        }
    }
    let mod_states_result: Vec<StatePair> = mod_states.into_iter().collect();
    let mod_labels_result: Vec<ProdLabellingPair> = mod_labels.into_iter().collect();
    (mod_transitions_result, rm_transitions, mod_states_result, mod_labels_result)
}

pub fn modify_complete_tasks<'a, 'b>(mod_compl_trans: &'b[&ProdMDPTransition], dfas: &'a [(usize, DFA)], labels: &'b [ProdLabellingPair]) -> (Vec<ProdMDPTransition>, Vec<&'b ProdMDPTransition>, Vec<StatePair>, Vec<ProdLabellingPair>) {
    let mut mod_states: HashSet<StatePair> = HashSet::new();
    let mut mod_labels: HashSet<ProdLabellingPair> = HashSet::new();
    let re = regex::Regex::new(r"\d+").unwrap();
    let mut mod_transitions: Vec<ProdMDPTransition> = Vec::new();
    let mut rm_transitions: Vec<&'b ProdMDPTransition> = Vec::new();
    for t in mod_compl_trans.iter() {
        let mut sprime_mod: Vec<ProdTransitionPair> = Vec::new();
        let word = labels.iter().find(|x| x.s == t.s).unwrap().w.iter().
            find(|x| x.contains("active")).unwrap();
        let task = re.captures(word).unwrap();
        let mut task_number = task.get(0).unwrap().as_str();
        let task_index: usize = task_number.parse().unwrap();
        let mut task_completed_str = "999".to_string();
        task_completed_str.push_str(&task_number);
        let mut task_complete_state_num: u32 = task_completed_str.parse().unwrap();
        let mut transition_modified: bool = false;
        let mut mod_state = StatePair { s: 0, q: vec![] };
        let mut mod_sprime_state = StatePair { s: 0, q: vec![] };
        for sprime in t.s_sprime.iter() {
            match labels.iter().find(|x| x.s == sprime.s).unwrap().w.iter().find(|x| x.contains("done")) {
                None => {
                    // it is not possible for this to be the correct transition, add the current s' and move on
                    sprime_mod.push(ProdTransitionPair{ s: StatePair { s: sprime.s.s, q: sprime.s.q.to_vec() }, p: sprime.p });
                }
                Some(x) => {
                    if dfas[task_index].1.acc.iter().any(|x| *x == sprime.s.q[task_index]) {
                        // this is an accepting transition
                        sprime_mod.push(ProdTransitionPair{ s: StatePair { s: task_complete_state_num, q: sprime.s.q.to_vec() }, p: sprime.p });
                        mod_states.insert(StatePair{ s: task_complete_state_num, q: sprime.s.q.to_vec() });
                        mod_labels.insert(ProdLabellingPair{ s: StatePair { s: task_complete_state_num, q: sprime.s.q.to_vec() }, w: vec![format!("jsucc{}",task_index)] });
                        mod_state = StatePair { s: task_complete_state_num, q: sprime.s.q.to_vec() };
                        mod_sprime_state = StatePair { s: sprime.s.s, q: sprime.s.q.to_vec() };
                        transition_modified = true;
                    } else {
                        // The only way the succ label is seen is if the task DFA is accepting for this particular q
                        sprime_mod.push(ProdTransitionPair{ s: StatePair { s: sprime.s.s, q: sprime.s.q.to_vec() }, p: sprime.p });
                    }
                }
            }
        }
        if transition_modified {
            let mod_transition = ProdMDPTransition {
                s: StatePair { s: t.s.s, q: t.s.q.to_vec() },
                a: t.a.to_string(),
                s_sprime: sprime_mod,
                reward: t.reward
            };
            mod_transitions.push(mod_transition);
            match mod_transitions.iter().any(|x| x.s == mod_state) {
                true => {}
                false => {
                    let jsucc_mod_transition = ProdMDPTransition {
                        s: mod_state,
                        a: "tau".to_string(),
                        s_sprime: vec![ProdTransitionPair{ s: mod_sprime_state, p: 1.0 }],
                        reward: 0.0
                    };
                    mod_transitions.push(jsucc_mod_transition);
                }
            }
            rm_transitions.push(t);
        }
    }
    let mod_states_result: Vec<StatePair> = mod_states.into_iter().collect();
    let mod_labels_result: Vec<ProdLabellingPair> = mod_labels.into_iter().collect();
    (mod_transitions, rm_transitions, mod_states_result, mod_labels_result)
}

pub fn create_final_labelling<'a, 'b>(labelling: &'b mut [ProdLabellingPair], states: &'b [StatePair], dfas: &'a [(usize, DFA)]) -> Vec<ProdLabellingPair>  {
    let mut new_labels: Vec<ProdLabellingPair> = Vec::new();
    for state in states.iter().
        filter(|x| find_final(&x.q[..], &dfas[..])) {
        match labelling.iter_mut().find(|x|x.s == *state) {
            None => {
                new_labels.push(ProdLabellingPair {
                    s: StatePair { s: state.s, q: state.q.to_vec() },
                    w: vec!["complete".to_string()]
                });
            },
            Some(mut x) => x.w.push("complete".to_string())
        }
    }
    new_labels
}

pub fn find_final(q: &[u32], dfas: &[(usize,DFA)]) -> bool {
    for (j, state) in q.iter().enumerate() {
        if dfas[j as usize].1.dead.iter().all(|x| x != state) &&
            dfas[j as usize].1.acc.iter().all(|x| x != state) {
            return false;
        }
    }
    true
}

pub fn create_graph<'a>(states: &'a [StatePair], transitions: &'a [ProdMDPTransition]) -> Graph<String, String> {
    let mut g: Graph<String, String> = Graph::new();
    for s in states.iter(){
        g.add_node(format!("({},{:?})", s.s, s.q));
    }
    for t in transitions.iter().
        filter(|x| states.iter().any(|y| *y == x.s)) {
        let origin_index = g.node_indices().
            find(|x| g[*x] == format!("({},{:?})", t.s.s, t.s.q)).unwrap();
        for sprime in t.s_sprime.iter() {
            let destination_index = g.node_indices().
                find(|x| g[*x] == format!("({},{:?})", sprime.s.s, sprime.s.q)).unwrap();
            g.add_edge(origin_index, destination_index,t.a.to_string());
        }
    }
    g
}

pub fn reachable_from_initial<'a, 'b>(states: &'a [StatePair], transitions: &'a [ProdMDPTransition], initial: &'b StatePair) -> Vec<StatePair> {
    let mut visited: Vec<bool> = vec![false; states.len()];
    let mut stack: Vec<&StatePair> = Vec::new();
    let initial_ref = states.iter().position(|x| x == initial).unwrap();
    visited[initial_ref] = true;
    stack.push(&states[initial_ref]);
    while !stack.is_empty() {
        // find the adjacent states of stack state
        let new_state = stack.pop().unwrap();
        for t in transitions.iter().filter(|x| x.s == *new_state) {
            for t_sprime in t.s_sprime.iter() {
                let new_state_index = states.iter().position(|x| *x == t_sprime.s).unwrap();
                if !visited[new_state_index] {
                    stack.push(&states[new_state_index]);
                    visited[new_state_index] = true;
                }
            }
        }
    }
    let mut reachable_states: Vec<StatePair> = vec![StatePair{ s: 0, q: vec![] }; visited.iter().filter(|x| **x).count()];
    let mut counter: usize = 0;
    for i in 0..visited.len() {
        if visited[i] {
            reachable_states[counter] = StatePair{s:states[i].s, q: states[i].q.to_vec()};
            counter+=1;
        }
    }
    reachable_states
}

pub fn prune_transitions<'a>(states: &'a [StatePair], transitions: &'a [ProdMDPTransition]) -> Vec<ProdMDPTransition> {
    transitions.into_iter().filter(|x| states.iter().any(|y| x.s == *y)).map(|x| ProdMDPTransition{
        s: StatePair { s: x.s.s, q: x.s.q.to_vec() },
        a: x.a.to_string(),
        s_sprime: x.s_sprime.to_vec(),
        reward: x.reward
    }).collect()
}

pub fn append_labels<'a>(labels: &'a[ProdLabellingPair], mod_labels: &'a[ProdLabellingPair]) -> Vec<ProdLabellingPair> {
    let mut new_labels: Vec<ProdLabellingPair> = vec![ProdLabellingPair{ s: StatePair { s: 0, q: vec![] }, w: vec![] }; labels.len() + mod_labels.len()];
    let mut count: usize = 0;
    for l in labels.iter() {
        new_labels[count] = ProdLabellingPair {
            s: StatePair { s: l.s.s, q: l.s.q.to_vec() },
            w: l.w.to_vec()
        };
        count += 1;
    }
    for l in mod_labels.iter() {
        new_labels[count] = ProdLabellingPair {
            s: StatePair { s: l.s.s, q: l.s.q.to_vec() },
            w: l.w.to_vec()
        };
        count += 1
    }
    new_labels
}

pub fn append_transitions<'a>(transitions: &'a [ProdMDPTransition], mod_transitions: &'a [ProdMDPTransition]) -> Vec<ProdMDPTransition> {
    let mut new_transitions: Vec<ProdMDPTransition> = vec![ProdMDPTransition{
        s: StatePair { s: 0, q: vec![] },
        a: "".to_string(),
        s_sprime: vec![],
        reward: 0.0
    }; transitions.len() + mod_transitions.len()];
    let mut count: usize = 0;
    for t in transitions.iter() {
        new_transitions[count] = ProdMDPTransition {
            s: StatePair { s: t.s.s, q: t.s.q.to_vec() },
            a: t.a.to_string(),
            s_sprime: t.s_sprime.to_vec(),
            reward: t.reward
        };
        count += 1;
    }
    for t in mod_transitions.iter() {
        new_transitions[count] = ProdMDPTransition {
            s: StatePair { s: t.s.s, q: t.s.q.to_vec() },
            a: t.a.to_string(),
            s_sprime: t.s_sprime.to_vec(),
            reward: t.reward
        };
        count += 1;
    }
    new_transitions
}

pub fn remove_transitions<'a>(transitions: &'a [ProdMDPTransition], remove: &'a [&'a ProdMDPTransition]) -> Vec<ProdMDPTransition> {
    let mut new_transitions: Vec<ProdMDPTransition> = vec![ProdMDPTransition{
        s: StatePair { s: 0, q: vec![] },
        a: "".to_string(),
        s_sprime: vec![],
        reward: 0.0
    }; transitions.len() - remove.len()];
    let mut count: usize = 0;
    for t in transitions.iter() {
        match remove.iter().any(|x| x.s == t.s && x.a == t.a) {
            true => {}
            false => {
                new_transitions[count] = ProdMDPTransition {
                    s: StatePair { s: t.s.s, q: t.s.q.to_vec() },
                    a: t.a.to_string(),
                    s_sprime: t.s_sprime.to_vec(),
                    reward: t.reward
                };
                count+=1;
            }
        }
    }
    new_transitions
}

pub fn append_states<'a>(states: &'a [StatePair], mod_states: &'a [StatePair]) -> Vec<StatePair>    {
    let mut new_states: Vec<StatePair> = vec![StatePair{ s: 0, q: vec![] }; states.len() + mod_states.len()];
    let mut count: usize = 0;
    for s in states.iter() {
        new_states[count] = StatePair {
            s: s.s,
            q: s.q.to_vec()
        };
        count += 1;
    }
    for s in mod_states.iter() {
        new_states[count] = StatePair{
            s: s.s,
            q: s.q.to_vec()
        };
        count += 1;
    }
    new_states
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

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct ProdLabellingPair {
    pub s: StatePair,
    pub w: Vec<String>
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProdTransitionPair {
    pub s: StatePair,
    pub p: f64
}