mod model_checking;
use std::collections::{HashSet, HashMap};
use model_checking::mdp2::*;
use model_checking::dfa::*;
use std::iter::FromIterator;
use model_checking::helper_methods::Rewards;
//use std::time::Instant;
use model_checking::product_automata::*;
use model_checking::stapu_framework::*;
use std::time::Instant;

fn main() {
    let mut words: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut initiate_letters: Vec<String> = Vec::with_capacity(4);
    let mut send_letters: Vec<String> = Vec::with_capacity(4);
    let num_tasks: usize = 4;
    for tasks in 0..num_tasks {
        initiate_letters.push(format!("initiate({})", tasks));
        send_letters.push(format!("send({})", tasks));
    }
    let ref_init_letters: Vec<&str> = initiate_letters.iter().map(|x| x as &str).collect();
    let ref_send_letters: Vec<&str> = send_letters.iter().map(|x| x as &str).collect();
    for tasks in 0..num_tasks {
        words.insert(ref_init_letters[tasks], HashSet::from_iter(vec![ref_init_letters[tasks]]));
        words.insert(ref_send_letters[tasks], HashSet::from_iter(vec![ref_send_letters[tasks]]));
    }
    let hempty: HashSet<&str> = HashSet::from_iter(vec![]);
    let hready:HashSet<&str> = HashSet::from_iter(vec!["ready"]);
    let hexit:HashSet<&str> = HashSet::from_iter(vec!["exit"]);
    let hsend: HashSet<&str> = HashSet::from_iter(vec!["send(k)"]);
    let hinitiate: HashSet<&str>= HashSet::from_iter(vec!["initiate(k)"]);
    let _hini: HashSet<&str> = HashSet::from_iter(vec!["ini"]);
    let _hcom: HashSet<&str> = HashSet::from_iter(vec!["com"]);
    let _hsuc: HashSet<&str> = HashSet::from_iter(vec!["suc"]);
    let _hfai: HashSet<&str> = HashSet::from_iter(vec!["fai"]);
    let _swi: String = String::from("swi");
    let _done: &str = "done";
    let _rewards: Rewards = Rewards::NEGATIVE;

    let send_k_tasks: Vec<DFA> = Vec::with_capacity(num_tasks);

    let send_1_task: DFA = construct_send_task(1, 0, num_tasks, &words, &hempty, &hready, &hexit, &ref_init_letters[..], &ref_send_letters[..]);
    let send_2_task: DFA = construct_send_task(2, 1, num_tasks, &words, &hempty, &hready, &hexit, &ref_init_letters[..], &ref_send_letters[..]);
    let send_3_task: DFA = construct_send_task(3, 2, num_tasks, &words, &hempty, &hready, &hexit, &ref_init_letters[..], &ref_send_letters[..]);
    let send_4_task: DFA = construct_send_task(4, 3, num_tasks, &words, &hempty, &hready, &hexit, &ref_init_letters[..], &ref_send_letters[..]);

    let empty_dfa_states: Vec<CrossProdState> = vec![];
    let empty_dfa_transitions: Vec<CrossProdTransition> = vec![];
    let empty_jacc: Vec<CrossProdState> = Vec::new();
    let empty_acc: Vec<CrossProdState> = Vec::new();
    let empty_dead: Vec<CrossProdState> = Vec::new();
    let blank_init = CrossProdState { q: 0, desc: "".to_string(), trans_ix: vec![], active_ix: 0, jacc: false, acc: false, dead: false, init: false, switch_to: false, task_complete: vec![], done: false };
    let (cross_prod_states, cross_prod_transitions, cross_prod_jacc,
        cross_prod_acc, cross_prod_dead, cross_prod_init) =
        create_new_states_and_transitions(&empty_dfa_states[..], &send_1_task.states[..], &empty_dfa_transitions[..],
                                          &send_1_task.delta[..], &empty_acc[..], &empty_dead[..], &empty_jacc[..], blank_init,
                                          &send_1_task.acc[..], &send_1_task.jacc[..],&send_1_task.dead[..], send_1_task.initial, 0);
    let (cross_prod_states, cross_prod_transitions, cross_prod_jacc,
        cross_prod_acc, cross_prod_dead, cross_prod_init) =
        create_new_states_and_transitions(&cross_prod_states[..], &send_2_task.states[..], &cross_prod_transitions[..],
                                          &send_2_task.delta[..],&cross_prod_acc[..], &cross_prod_dead[..], &cross_prod_jacc[..],cross_prod_init,
                                          &send_2_task.acc[..], &send_2_task.jacc[..],&send_2_task.dead[..], send_2_task.initial, 1);

    let (cross_prod_states, cross_prod_transitions, cross_prod_jacc,
        cross_prod_acc, cross_prod_dead, cross_prod_init) =
        create_new_states_and_transitions(&cross_prod_states[..], &send_3_task.states[..], &cross_prod_transitions[..],
                                          &send_3_task.delta[..],&cross_prod_acc[..], &cross_prod_dead[..], &cross_prod_jacc[..],cross_prod_init,
                                          &send_3_task.acc[..], &send_3_task.jacc[..],&send_3_task.dead[..], send_3_task.initial, 2);

    let (cross_prod_states, cross_prod_transitions, cross_prod_jacc,
        cross_prod_acc, cross_prod_dead, cross_prod_init) =
        create_new_states_and_transitions(&cross_prod_states[..], &send_4_task.states[..], &cross_prod_transitions[..],
                                          &send_4_task.delta[..],&cross_prod_acc[..], &cross_prod_dead[..], &cross_prod_jacc[..],cross_prod_init,
                                          &send_4_task.acc[..], &send_4_task.jacc[..],&send_4_task.dead[..], send_4_task.initial, 3);

    let task_automaton = CrossProdDFA {
        states: cross_prod_states,
        delta: cross_prod_transitions,
        acc: cross_prod_acc,
        dead: cross_prod_dead,
        jacc: cross_prod_jacc
    };
    let mdp = construct_mdp(&hempty, &hinitiate, &hready, &hexit, &hsend);
    let mdps: Vec<MDP2> = vec![mdp.clone(), mdp.clone()];
    let num_agents: usize = 2;
    let mut local_prods: Vec<LocalProduct> = Vec::with_capacity(num_tasks * num_agents);
    for mdp in mdps.iter() {
        let local_prod_states = create_local_prod_states(&mdp.states[..], &task_automaton.states[..], mdp.initial);
        let local_prod_init = local_prod_states.iter().position(|x| x.s == mdp.initial && x.q == cross_prod_init.q).unwrap();
        let local_prod_trans =
            create_local_prod_transitions(&local_prod_states[..], &mdp.labelling[..], &mdp.transitions[..], &task_automaton.delta[..], num_tasks as i32);
        let reachable_s = reachable_states(&local_prod_states[..], &local_prod_trans[..], local_prod_init);
        let mut reachable_trans: Vec<LocalProdTransitions> = Vec::new();
        for (ix, s) in reachable_s.iter().enumerate() {
            let old_index = local_prod_states.iter().position(|x| x.s == s.s && x.q == s.q).unwrap();
            for t in local_prod_trans.iter().filter(|x| x.sq == old_index) {
                //print!("s old: ({},{}), -> {}", local_prod_states[t.sq].s, local_prod_states[t.sq].desc, t.a);
                let mut sq_prime_new: Vec<LocalProductTransitionPair> = Vec::with_capacity(t.sq_prime.len());
                for sprime in t.sq_prime.iter() {
                    //print!("s': ({},{}), {}", local_prod_states[sprime.s].s, local_prod_states[sprime.s].desc, t.reward);
                    //print!(", ");
                    let sq_prime_new_ix = reachable_s.iter().
                        position(|x| x.s == local_prod_states[sprime.s].s && x.q == local_prod_states[sprime.s].q).unwrap();
                    sq_prime_new.push(LocalProductTransitionPair{s: sq_prime_new_ix, p: sprime.p });
                }
                reachable_trans.push(LocalProdTransitions {
                    sq: ix,
                    a: t.a.to_string(),
                    sq_prime: sq_prime_new,
                    reward: t.reward
                });
                //println!();
            }
        }
        local_prods.push(LocalProduct {
            states: reachable_s,
            init: local_prod_init,
            transitions: reachable_trans
        });
    }

    let rewards = Rewards::NEGATIVE;
    let mut state_count_ix = 0;
    let mut trans_count_ix = 0;
    let mut team_states: Vec<TeamState> = Vec::new();
    let mut team_transitions: Vec<TeamTransition> = Vec::new();
    let mut switch_from_input: Vec<usize> = Vec::new();
    let mut prev_init_index: usize = 0;
    for (k, loc_prod) in local_prods.iter().enumerate() {
        let (_team_states, _team_transitions, switch_state_ix_update, trans_count_ix_update, state_count_ix_update, prev_init_index_update) =
            create_state_trans_index_mapping(&loc_prod.states[..],&loc_prod.transitions[..],loc_prod.init, prev_init_index, k,
                                         state_count_ix, trans_count_ix, num_agents, num_tasks, &rewards,
                                         &mut team_states, &mut team_transitions, &switch_from_input[..]);
        prev_init_index = prev_init_index_update;
        //println!("prev init index: {}", prev_init_index);
        switch_from_input = switch_state_ix_update;
        //println!("{:?}", switch_from_input);
        state_count_ix = state_count_ix_update;
        //println!("states:{:?}", state_count_ix);
        trans_count_ix = trans_count_ix_update;
    }
    let team_init_index = local_prods[0].init;
    let team_mdp = TeamMDP {
        states: team_states,
        transitions: team_transitions
    };
    /*for state in team_mdp.states.iter() {
        print!("s:({},{},{}) ->  ",team_mdp.states[state.ix].s, team_mdp.states[state.ix].desc,
               team_mdp.states[state.ix].agent);
        for t in state.trans_ix.iter() {
            let transition = &team_mdp.transitions[*t];

            for sprime in transition.s_prime.iter() {
                 print!("(a:{})-> s':({},{},{})", transition.a, team_mdp.states[sprime.s].s, team_mdp.states[sprime.s].desc,
                       team_mdp.states[sprime.s].agent);
                print!(", ");
            }
        }
        println!();
    }

     */
    /*
    let mut hullset: Vec<Vec<f64>> = Vec::new();
    let mut mu_vect: Vec<Vec<String>> = Vec::new();
    let epsilon: f64 = 0.00001;
    let verbose: bool = true;
    //let mut extreme_points: Vec<Vec<f64>> = vec![vec![0.0; num_agents + num_tasks]; num_agents + num_tasks];

    let w_extr: &Vec<f64> = &vec![0.3,0.4,0.3];
    if verbose {
        println!("w: {:?}", w_extr);
    }

    let safe_ret = opt_exp_tot_cost_non_iter(&w_extr[..], &epsilon, &team_mdp.states[..], &team_mdp.transitions[..],
                                             &rewards, &num_agents, &num_tasks, &team_init_index);
    match safe_ret {
        Some((mu_new, r)) => {
            hullset.push(r);
            mu_vect.push(mu_new);
        },
        None => panic!("No value was returned from the maximisation")
    }
    println!("extreme points: ");

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

    let target: Vec<f64> = vec![-25.0, -25.0, 0.7, 0.7, 0.7, 0.7];
    let epsilon: f64 = 0.00001;
    let start = Instant::now();
    let _output = multi_obj_sched_synth_non_iter(team_init_index,
                                                &team_mdp.states[..], &team_mdp.transitions[..], &target,
                                                &epsilon, &rewards, &true, num_tasks, num_agents);
    let duration = start.elapsed();
    println!("duration: {:?}", duration);

}



fn construct_send_task<'a>(k: u32, task: usize, num_tasks: usize, words: &'a HashMap<&'a str, HashSet<&'a str>>,
                           hempty: &'a HashSet<&'a str>, hready: &'a HashSet<&'a str>, hexit: &'a HashSet<&'a str>,
                           init_words: &'a [&'a str], send_words: &'a [&'a str]) -> DFA<'a> {
    let q_states: Vec<u32> = (0..(1 + k + 3)).collect();
    let fail_state: u32 = *q_states.last().unwrap();
    let mut delta: Vec<DFATransitions> = Vec::with_capacity(4 + k as usize * 3);
    let mut non_init: Vec<&HashSet<&str>> = vec![&hready, &hexit, &hempty];
    for task_ix in 0..num_tasks {
        non_init.push(&words.get(send_words[task_ix]).unwrap());
        if task_ix != task {
            //println!("task_ix: {}, task: {}, {:?}", task_ix, task, init_words[task_ix]);
            non_init.push(&words.get(init_words[task_ix]).unwrap());
        }
    }
    let t1 = DFATransitions {
        q: 0,
        w: non_init,
        q_prime: 0
    };
    delta.push(t1);
    let t2 = DFATransitions {
        q: 0,
        w: vec![&words.get(init_words[task]).unwrap()],
        q_prime: 1
    };
    delta.push(t2);
    for i in 1..k+1 {
        delta.push(DFATransitions {
            q: i,
            w: vec![&words.get(send_words[task]).unwrap()],
            q_prime: i + 1
        });
        delta.push(DFATransitions {
            q: i,
            w: vec![&hexit],
            q_prime: fail_state
        });
        let mut non_exit_sprint: Vec<&HashSet<&str>> = vec![&hready, &hempty];
        for task_ix in 0..num_tasks {
            non_exit_sprint.push(&words.get(init_words[task_ix]).unwrap());
            if task_ix != task {
                non_exit_sprint.push(&words.get(send_words[task_ix]).unwrap());
            }
        }
        delta.push(DFATransitions {
            q: i,
            w: non_exit_sprint,
            q_prime: i
        });
    }
    let mut tru_v: Vec<&HashSet<&str>> = vec![&hempty, &hexit, &hready];
    for task_ix in 0..num_tasks {
        tru_v.push(&words.get(init_words[task_ix]).unwrap());
        tru_v.push(&words.get(send_words[task_ix]).unwrap());
    }
    delta.push(DFATransitions {
        q: fail_state,
        w: tru_v.to_vec(),
        q_prime: fail_state
    });
    delta.push(DFATransitions {
        q: fail_state - 2,
        w: tru_v.to_vec(),
        q_prime: fail_state - 1
    });
    delta.push(DFATransitions {
        q: fail_state - 1,
        w: tru_v.to_vec(),
        q_prime: fail_state - 1
    });
    DFA {
        states: q_states,
        initial: 0,
        delta,
        acc: vec![fail_state - 1],
        dead: vec![fail_state],
        jacc: vec![fail_state - 2]
    }
}

fn construct_mdp<'a>(hempty: &'a HashSet<&'a str>, hinitiate: &'a HashSet<&'a str>,
                     hready: &'a HashSet<&'a str>, hexit: &'a HashSet<&'a str>,
                     hsend: &'a HashSet<&'a str>) -> MDP2<'a> {
    let states: Vec<u32> = (0..5).collect();
    let initial: u32 = 0;
    let mut transitions: Vec<Transition> = Vec::with_capacity(7);
    let mut labelling: Vec<MDPLabellingPair<'a>> = Vec::with_capacity(5);
    transitions.push(Transition {
        s: 0,
        a: "a".to_string(),
        s_prime: vec![TransitionPair { s: 0, p: 0.05 }, TransitionPair { s: 1, p: 0.95 }],
        rewards: 3.0
    });
    transitions.push(Transition{
        s: 1,
        a: "a".to_string(),
        s_prime: vec![TransitionPair { s: 2, p: 1.0 }],
        rewards: 2.0
    });
    transitions.push( Transition {
        s: 2,
        a: "a".to_string(),
        s_prime: vec![TransitionPair { s: 3, p: 0.99 }, TransitionPair { s: 4, p: 0.01}],
        rewards: 3.0
    });
    transitions.push( Transition {
        s: 2,
        a: "c".to_string(),
        s_prime: vec![TransitionPair { s: 3, p: 0.9 }, TransitionPair {s: 4, p: 0.1}],
        rewards: 1.0
    });
    transitions.push(Transition {
        s: 2,
        a: "b".to_string(),
        s_prime: vec![TransitionPair {s: 4, p: 0.9}],
        rewards: 0.5
    });
    transitions.push(Transition{
        s: 3,
        a: "a".to_string(),
        s_prime: vec![TransitionPair { s: 2, p: 1.0 }],
        rewards: 0.5
    });
    transitions.push(Transition {
        s: 4,
        a: "a".to_string(),
        s_prime: vec![TransitionPair { s: 0, p: 1.0 }],
        rewards: 3.0
    });
    labelling.push(MDPLabellingPair {
        s: 0,
        w: vec![hempty]
    });
    labelling.push(MDPLabellingPair {
        s: 1,
        w: vec![hinitiate]
    });
    labelling.push(MDPLabellingPair {
        s: 2,
        w: vec![hready]
    });
    labelling.push(MDPLabellingPair {
        s: 3,
        w: vec![hsend]
    });
    labelling.push(MDPLabellingPair {
        s: 4,
        w: vec![hexit]
    });
    MDP2 {
        states,
        initial,
        transitions,
        labelling,
    }
}


