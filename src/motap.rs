mod model_checking;
use std::collections::{HashSet};
use model_checking::helper_methods::*;
use model_checking::decomp_team_mdp::*;
use model_checking::mdp2::*;
use model_checking::dfa::*;
use std::iter::FromIterator;
use crate::model_checking::decomp_team_mdp::create_state_transition_mapping;
use std::time::Instant;

fn main() {
    // Set the alphabet of the MDP, and automata
    //let alphabet: Vec<&str> = vec!["initiate", "ready", "exit", "send"];
    let hempty: HashSet<&str> = HashSet::from_iter(vec![]);
    let hinitiate: HashSet<&str> = HashSet::from_iter(vec!["initiate"]);
    let hready = HashSet::from_iter(vec!["ready"]);
    let hexit = HashSet::from_iter(vec!["exit"]);
    let hsend = HashSet::from_iter(vec!["send"]);
    let hini: HashSet<&str> = HashSet::from_iter(vec!["ini"]);
    let hcom: HashSet<&str> = HashSet::from_iter(vec!["com"]);
    let hsuc: HashSet<&str> = HashSet::from_iter(vec!["suc"]);
    let hfai: HashSet<&str> = HashSet::from_iter(vec!["fai"]);
    let swi: String = String::from("swi");
    let _done: &str = "done";
    let rewards: Rewards = Rewards::NEGATIVE;

    let num_tasks: usize = 2;
    let mut dfas: Vec<DFA> = Vec::with_capacity(num_tasks);

    for k in 1..num_tasks + 1 {
        let send_k_task: DFA = construct_send_task(k as u32, &hempty, &hinitiate, &hready, &hexit, &hsend);
        dfas.push(send_k_task);
    }
    let mdp = construct_mdp(&hempty, &hinitiate, &hready, &hexit, &hsend);
    let mdps: Vec<MDP2> = vec![mdp.clone(); 2];

    let num_agents: usize = mdps.len();
    let mut team_input: Vec<TeamInput> = Vec::with_capacity(num_tasks * num_agents);
    let mut local_product: Vec<DFAProductMDP> = vec![DFAProductMDP::default(); num_agents * num_tasks];
    let mut team_member_attrs: Vec<TeamAttrs> = Vec::with_capacity(num_tasks * num_agents);
    let mut team_counter: usize = 0;
    for (j, dfa) in dfas.iter().enumerate() {
        for (i, m) in mdps.iter().enumerate() {
            let states = create_states(&m.states[..], &dfa.states[..]);
            local_product[team_counter].initial = DFAModelCheckingPair { state: ProdState { s: m.initial, q: dfa.initial }, w: vec![&hini] };
            let init_state = DFAModelCheckingPair { state: ProdState { s: m.initial, q: dfa.initial }, w: vec![] };
            local_product[team_counter].transitions = create_prod_transitions(m, dfa, &states[..]);
            let reachable_state_truth = reachable_from_initial(&init_state,
                                                               &states[..], &local_product[team_counter].transitions[..]);
            let reachable_states = create_reachable_states_with_labels(init_state, &states[..], &reachable_state_truth[..],
                                            &dfa, &hini, &hcom, &hsuc, &hfai, &m.initial);
            local_product[team_counter].states = reachable_states;
            team_member_attrs.push(TeamAttrs {
                agent: i,
                task: j,
                dead: &dfa.dead,
                acc: &dfa.acc,
                jacc: &dfa.jacc
            });
            team_counter += 1;
        }
    }
    for t in 0..num_agents * num_tasks {
        team_input.push(TeamInput {
            agent: team_member_attrs[t].agent,
            task: team_member_attrs[t].task,
            product: &local_product[t],
            dead: team_member_attrs[t].dead,
            acc: team_member_attrs[t].acc,
            jacc: team_member_attrs[t].jacc
        });
    }
    let mut team_mdp_states: Vec<TeamState> = Vec::new();
    let mut team_mdp_transitions: Vec<TeamTransition> = Vec::new();
    let mut prev_state_ix: usize = 0;
    let mut prev_trans_ix: usize = 0;
    let mut next_init_agent_state: Option<usize> = None;
    let mut next_init_task_state: Option<usize> = None;
    // Initially the next initial point for the agent will be None, but after the last agent has been
    // input into the team structure, the "next init agent" state gets recorded, once the first agent has been
    // input into the team structure the next_init_agent will be set back to None again.
    for j in (0..num_tasks).rev() {
        for i in (0..num_agents).rev() {
            let loc_prod = team_input.iter().find(|x| x.agent == i && x.task == j).unwrap();
            let (_team_mdp_states, _team_mdp_transitions, state_ix_counter, trans_ix_counter, update_agent_init_state, update_task_init_state) =
                create_state_transition_mapping(&loc_prod.product.states[..], &loc_prod.product.transitions[..], i, j, prev_state_ix,
                                                prev_trans_ix, &mut team_mdp_states,
                                                &mut team_mdp_transitions, num_agents, num_tasks,
                                                &loc_prod.jacc[..], &loc_prod.acc[..], &loc_prod.dead[..],
                                                &rewards, &hini, &hsuc, &hfai, &swi, next_init_task_state, next_init_agent_state);
            prev_state_ix = state_ix_counter;
            prev_trans_ix = trans_ix_counter;
            next_init_agent_state = update_agent_init_state;
            next_init_task_state = update_task_init_state;
        }
    }
    let initial_prod = team_input.iter().
        find(|x| x.agent == 0 && x.task == 0).unwrap().product;
    let init_team_state_ix = team_mdp_states.iter().
        position(|x| x.task == 0 && x.agent == 0 && *x.state == initial_prod.initial).unwrap();
    let team_mdp = TeamMDP {
        initial: TeamState {
            state: &initial_prod.initial,
            agent: 0,
            task: 0,
            trans_ix: vec![],
            label: Default::default(),
            ix: init_team_state_ix
        },
        states: team_mdp_states,
        transitions: team_mdp_transitions,
        num_agents,
        num_tasks,
        //task_alloc_states: vec![]
    };
    /*for t in team_mdp.transitions.iter() {
        print!("ix: {}, s:({},{},{},{}) -> ", t.from, team_mdp.states[t.from].state.state.s, team_mdp.states[t.from].state.state.q,
               team_mdp.states[t.from].agent, team_mdp.states[t.from].task);
        print!("a: {}, -> ", t.a);
        for sprime in t.to.iter() {
            print!("ix: {}, s':({},{},{},{})", sprime.state, team_mdp.states[sprime.state].state.state.s, team_mdp.states[sprime.state].state.state.q,
               team_mdp.states[sprime.state].agent, team_mdp.states[sprime.state].task);
            print!(", ");
        }
        print!("reward: {:?}", t.reward);
        println!();
    }*/
    let rewards: Rewards = Rewards::NEGATIVE;
    println!("|S|: {}, |P|: {}", team_mdp.states.len(), team_mdp.transitions.len());
    let target: Vec<f64> = vec![
        -10.0, -10.0,
        500.0, 500.0];
    let epsilon: f64 = 0.00001;
    let num_runs: usize = 1;
    let mut times: Vec<f64> = Vec::with_capacity(num_runs);
    for _ in 0..num_runs {
        let ranges = create_ij_state_mapping(num_tasks, num_agents, &team_mdp.states[..]);
        let start = Instant::now();
        let _output = multi_obj_sched_synth(&target[..], &epsilon, &ranges[..],&team_mdp.states[..],
                                            &team_mdp.transitions[..], &rewards, &true, num_tasks, num_agents, team_mdp.initial.ix);
        let duration = start.elapsed();
        times.push(duration.as_secs_f64());
    }
    let total_time = times.iter().fold(0.0, |sum, &val| sum + val) / num_runs as f64;
    println!("duration: {:?}", total_time);

    let mut times: Vec<f64> = Vec::with_capacity(num_runs);
    for _ in 0..num_runs {
        let start = Instant::now();
        let _output = multi_obj_sched_synth_non_iter(&target[..], &epsilon, &team_mdp.states[..],
                                                     &team_mdp.transitions[..], &rewards, &true, num_tasks, num_agents, team_mdp.initial.ix);
        let duration = start.elapsed();
        times.push(duration.as_secs_f64());
    }

    let total_time = times.iter().fold(0.0, |sum, &val| sum + val) / num_runs as f64;

    println!("duration: {:?}", total_time);
}

fn construct_send_task<'a, 'b>(k: u32, hempty: &'a HashSet<&'a str>, hinitiate: &'a HashSet<&'a str>,
                               hready: &'a HashSet<&'a str>, hexit: &'a HashSet<&'a str>, hsend: &'a HashSet<&'a str>)
    -> DFA<'a> {
    let q_states: Vec<u32> = (0..(1 + k + 3)).collect();
    let fail_state: u32 = *q_states.last().unwrap();
    let mut delta: Vec<DFATransitions> = Vec::with_capacity(4 + k as usize * 3);
    let t1 = DFATransitions {
        q: 0,
        w: vec![&hempty, &hready, &hexit, &hsend],
        q_prime: 0
    };
    delta.push(t1);
    let t2 = DFATransitions {
        q: 0,
        w: vec![&hinitiate],
        q_prime: 1
    };
    delta.push(t2);
    for i in 1..k+1 {
        delta.push(DFATransitions {
            q: i,
            w: vec![&hsend],
            q_prime: i + 1
        });
        delta.push(DFATransitions {
            q: i,
            w: vec![&hexit],
            q_prime: fail_state
        });
        delta.push(DFATransitions {
            q: i,
            w: vec![&hempty, &hinitiate, &hready],
            q_prime: i
        });
    }
    delta.push(DFATransitions {
        q: fail_state,
        w: vec![&hempty, &hsend, &hexit, &hready, &hinitiate],
        q_prime: fail_state
    });
    delta.push(DFATransitions {
        q: fail_state - 2,
        w: vec![&hempty, &hsend, &hexit, &hready, &hinitiate],
        q_prime: fail_state - 1
    });
    delta.push(DFATransitions {
        q: fail_state - 1,
        w: vec![&hempty, &hsend, &hexit, &hready, &hinitiate],
        q_prime: fail_state - 1
    });
    DFA {
        states: q_states,
        initial: 0,
        delta: delta,
        acc: vec![fail_state - 1],
        dead: vec![fail_state],
        jacc: vec![fail_state - 2]
    }
}

#[allow(dead_code)]
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