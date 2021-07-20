use super::helper_methods;
use super::dfa;
//use super::mdp2;
use super::gurobi_lp;
//use petgraph::{Graph};
use std::collections::{HashSet};
use ndarray::prelude::*;
use helper_methods::*;
use dfa::*;
//use mdp2::*;
use gurobi_lp::*;
use petgraph::Graph;
use petgraph::graph::NodeIndex;
use ordered_float::OrderedFloat;

#[allow(dead_code)]
pub struct TeamMDP<'a> {
    pub initial: TeamState<'a>,
    pub states: Vec<TeamState<'a>>,
    pub transitions: Vec<TeamTransition<'a>>,
    pub num_agents: usize,
    pub num_tasks: usize,
    //pub task_alloc_states: Vec<TaskAllocStates<'a>>,
}

#[allow(dead_code)]
pub fn multi_obj_sched_synth_non_iter(target: &[f64], eps: &f64, states: &[TeamState], transitions: &[TeamTransition],
                                      rewards: &Rewards, verbose: &bool, num_tasks: usize, num_agents: usize, init_ix: usize) -> Alg1Output {
    let mut hullset: Vec<Vec<f64>> = Vec::new();
    let mut mu_vect: Vec<Vec<String>> = Vec::new();
    let mut alg1_output: Alg1Output = Alg1Output{
        v: vec![],
        mu: vec![],
        hullset: vec![]
    };
    if *verbose {
        println!("num tasks: {}, num agents {}", num_tasks, num_agents);
    }
    let team_init_index = init_ix;

    //let mut extreme_points: Vec<Vec<f64>> = vec![vec![0.0; num_agents + num_tasks]; num_agents + num_tasks];
    /*
    for k in 0..(num_tasks + num_agents) {
        extreme_points[k][k] = 1.0;
        let w_extr: &Vec<f64> = &extreme_points[k];
        if *verbose {
            println!("w: {:?}", w_extr);
        }
        let safe_ret = opt_exp_tot_cost_non_iter(&w_extr[..], &eps, states, transitions,
                                                 rewards, &num_agents, &num_tasks, &team_init_index);
        match safe_ret {
            Some((mu_new, r)) => {
                hullset.push(r);
                mu_vect.push(mu_new);
            },
            None => panic!("No value was returned from the maximisation")
        }
    }

     */
    let w_extr: Vec<f64> = vec![1.0 / (num_agents + num_tasks) as f64; num_agents + num_tasks];
    if *verbose {
        println!("w: {:?}", w_extr);
    }
    let safe_ret = opt_exp_tot_cost_non_iter(&w_extr[..], &eps, states, transitions,
                                             rewards, &num_agents, &num_tasks, &team_init_index);
    match safe_ret {
        Some((mu_new, r)) => {
            hullset.push(r);
            mu_vect.push(mu_new);
        },
        None => panic!("No value was returned from the maximisation")
    }
    if *verbose {
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
    }
    let dim = num_tasks + num_agents;
    let t_arr1 = arr1(target);
    let mut w_new = lp5(&hullset, target, &dim);
    while w_new != None {
        if *verbose {
            println!("w' :{:?}", w_new);
        }
        let safe_ret = opt_exp_tot_cost_non_iter(&w_new.as_ref().unwrap()[..], &eps, states, transitions,
                                                 rewards, &num_agents, &num_tasks, &team_init_index);
        match safe_ret {
            Some((mu_new, r)) => {
                if *verbose {
                    println!("new r: {:?}", r);
                }
                let weight_arr1 = arr1(&w_new.as_ref().unwrap());
                let r_arr1 = arr1(&r);
                let wr_dot = weight_arr1.dot(&r_arr1);
                let wt_dot = weight_arr1.dot(&t_arr1);
                if *verbose {
                    println!("<w,r>: {}, <w,t>: {}", wr_dot, wt_dot);
                }
                if wr_dot < wt_dot {
                    if *verbose {
                        println!("Multi-objective satisfaction not possible");
                    }
                    return alg1_output;
                }
                hullset.push(r);
                mu_vect.push(mu_new);
            },
            None => panic!("No value was returned from the maximisation")
        }
        w_new = lp5(&hullset, target, &dim);
    }
    if *verbose {
        println!("Constructing witness");
    }
    let v = witness(&hullset, target, &dim);
    if *verbose {
        println!("v: {:?}", v);
    }
    match v {
        None => {}
        Some(x) => {alg1_output.v = x}
    };
    alg1_output.mu = mu_vect;
    alg1_output.hullset =  hullset;
    alg1_output
}

#[allow(dead_code)]
pub fn multi_obj_sched_synth(target: &[f64], eps: &f64, ranges: &[(usize, usize)], states: &[TeamState], transitions: &[TeamTransition],
                             rewards: &Rewards, verbose: &bool, num_tasks: usize, num_agents: usize, init_ix: usize) -> Alg1Output {
    let mut hullset: Vec<Vec<f64>> = Vec::new();
    let mut mu_vect: Vec<Vec<String>> = Vec::new();
    let mut alg1_output: Alg1Output = Alg1Output{
        v: vec![],
        mu: vec![],
        hullset: vec![]
    };
    if *verbose {
        println!("num tasks: {}, num agents {}", num_tasks, num_agents);
    }

    let team_init_index = init_ix;

    //let mut extreme_points: Vec<Vec<f64>> = vec![vec![0.0; num_agents + num_tasks]; num_agents + num_tasks];
    /*
    for k in 0..(num_tasks + num_agents) {
        extreme_points[k][k] = 1.0;
        let w_extr: &Vec<f64> = &extreme_points[k];
        if *verbose {
            println!("w: {:?}", w_extr);
        }
        let safe_ret = opt_exp_tot_cost(&w_extr[..], &eps, ranges, states, transitions,
                                        &rewards, &num_agents, &num_tasks, &team_init_index);
        match safe_ret {
            Some((mu_new, r)) => {
                hullset.push(r);
                mu_vect.push(mu_new);
            },
            None => panic!("No value was returned from the maximisation")
        }
    }*/
    let w_extr: Vec<f64> = vec![1.0 / (num_agents + num_tasks) as f64; num_agents + num_tasks];
    if *verbose {
        println!("w: {:?}", w_extr);
    }
    let safe_ret = opt_exp_tot_cost_non_iter(&w_extr[..], &eps, states, transitions,
                                             rewards, &num_agents, &num_tasks, &team_init_index);
    match safe_ret {
        Some((mu_new, r)) => {
            hullset.push(r);
            mu_vect.push(mu_new);
        },
        None => panic!("No value was returned from the maximisation")
    }
    let dim = num_tasks + num_agents;
    if *verbose {
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
    }

    let t_arr1 = arr1(target);
    let mut w_new = lp5(&hullset, target, &dim);
    while w_new != None {
        /*if *verbose {
            println!("w' :{:?}", w_new);
        }*/
        let safe_ret = opt_exp_tot_cost(&w_new.as_ref().unwrap()[..], &eps, ranges, states, transitions,
                                        &rewards, &num_agents, &num_tasks, &team_init_index);
        match safe_ret {
            Some((mu_new, r)) => {
                /*if *verbose {
                    println!("new r: {:?}", r);
                }*/
                let weight_arr1 = arr1(&w_new.as_ref().unwrap());
                let r_arr1 = arr1(&r);
                let wr_dot = weight_arr1.dot(&r_arr1);
                let wt_dot = weight_arr1.dot(&t_arr1);
                if *verbose {
                    println!("<w,r>: {}, <w,t>: {}", wr_dot, wt_dot);
                }
                if wr_dot < wt_dot {
                    if *verbose {
                        println!("Multi-objective satisfaction not possible");
                    }
                    return alg1_output;
                }
                hullset.push(r);
                mu_vect.push(mu_new);
            },
            None => panic!("No value was returned from the maximisation")
        }
        w_new = lp5(&hullset, target, &dim);
    }
    if *verbose{
        println!("Constructing witness");
    }
    let v = witness(&hullset, &target, &dim);
    if *verbose {
        println!("v: {:?}", v);
    }
    match v {
        None => {}
        Some(x) => {alg1_output.v = x}
    };
    alg1_output.mu = mu_vect;
    alg1_output.hullset =  hullset;
    alg1_output
}

#[allow(dead_code)]
pub fn create_state_transition_mapping<'a, 'b, 'c>(states: &'a [DFAModelCheckingPair], trans: &'a [DFAProductTransition],
                                               agent: usize, task: usize, prev_state_index: usize, prev_transition_index: usize,
                                               team_states: &'b mut Vec<TeamState<'a>>, team_trans: &'b mut Vec<TeamTransition<'a>>,
                                               num_agents: usize, num_tasks: usize, jacc: &'a [u32], acc: &'a [u32], dead: &'a [u32], rewards: &Rewards,
                                               hini: &'a HashSet<&'a str>, hsuc: &'a HashSet<&'a str>, hfai: &'a HashSet<&'a str>, swi: &'a String,
                                               next_init_task_state: Option<usize>, next_init_agent_state: Option<usize>)
                                               -> (&'b mut Vec<TeamState<'a>>, &'b mut Vec<TeamTransition<'a>>, usize, usize, Option<usize>, Option<usize>) {
    let rewards_coeff: f64 = match rewards {
        Rewards::NEGATIVE => -1.0,
        Rewards::POSITIVE => 1.0
    };
    let mut current_init_agent_state: Option<usize> = None;
    let mut current_init_task_state: Option<usize> = None;
    let mut state_index_counter: usize = prev_state_index;
    let mut trans_index_counter: usize = prev_transition_index;
    for s in states.iter() {
        let mut new_team_state = TeamState {
            state: s,
            agent,
            task,
            trans_ix: vec![],
            label: HashSet::new(),
            ix: 0
        };
        for t in trans.iter().filter(|x| x.sq.state == s.state) {
            let mut sprime_team_trans_pairs: Vec<TeamTransitionPair> = Vec::with_capacity(t.sq_prime.len());
            for sprime in t.sq_prime.iter() {
                // we can guarantee that the s' is not in a previous product MDP, it will only be the
                // current state index + the s' index value of the product state
                let sprime_ix = states.iter().position(|x| x.state == sprime.state.state).unwrap();
                sprime_team_trans_pairs.push(TeamTransitionPair{ state: sprime_ix + prev_state_index, p: sprime.p });
            }
            // ----------------------------------------
            // Assign the rewards, the product MDP will still be accumulating rewards
            let mut rewards: Vec<f64> = vec![0.0; num_agents + num_tasks];
            if jacc.iter().any(|x| *x == s.state.q) {
                rewards[num_agents + task] = 1000.0;
            } else {
                if task == num_tasks - 1 && (acc.iter().any(|x| *x == s.state.q) || dead.iter().any(|x| *x == s.state.q)) {
                    // do nothing
                } else {
                    rewards[agent] = rewards_coeff * t.reward
                }
            }
            let new_team_transition = TeamTransition {
                from: state_index_counter,
                a: &t.a,
                to: sprime_team_trans_pairs,
                reward: rewards
            };
            team_trans.push(new_team_transition);
            new_team_state.trans_ix.push(trans_index_counter);
            trans_index_counter += 1;
        }
        // -----------------
        // switch transitions
        // (i) start with the initial switch transition, if the state is initial and the next agent
        //     exists (Option<usize>) then create a switch transition
        // ----- How switch transitions work when we are constructing the team MDP in one loop
        // we have previously recorded the transition to state to, i.e. the state in the next local
        // product which satisfies suc || fai, and now we want to find the initial state of the
        // current local product and we transition from the previous suc || fai to the currrent
        // initial state.
        // construct a switch transition if the next_ini_task is not None
        if s.w.iter().any(|x| x.intersection(&hini).count() > 0) {
            if agent > 0 {
                // return the current state index which will be used as the next initial agent index
                current_init_agent_state = Some(state_index_counter);
            } else {
                current_init_agent_state = None;
                // still record the task init state if the agent is 0
                current_init_task_state = Some(state_index_counter);
            }
            match next_init_agent_state {
                None => {}
                Some(x) => {
                    team_trans.push(
                        TeamTransition {
                            from: state_index_counter,
                            a: swi,
                            to: vec![TeamTransitionPair { state: x, p: 1.0 }],
                            reward: vec![0.0; num_agents + num_tasks]
                        }
                    );
                    new_team_state.trans_ix.push(trans_index_counter);
                    trans_index_counter += 1;
                }
            }
        } else if s.w.iter().any(|x| x.intersection(&hfai).count() > 0) || s.w.iter().any(|x| x.intersection(&hsuc).count() > 0){
            if task < num_tasks - 1 {
                match next_init_task_state {
                    None => {}
                    Some(x) => {
                        team_trans.push(
                            TeamTransition {
                                from: state_index_counter,
                                a: swi,
                                to: vec![TeamTransitionPair { state: x, p: 1.0 }],
                                reward: vec![0.0; num_tasks + num_agents]
                            }
                        );
                        new_team_state.trans_ix.push(trans_index_counter);
                        trans_index_counter += 1;
                    }
                }
            }
        }
        new_team_state.ix = state_index_counter;
        team_states.push(new_team_state);
        state_index_counter += 1;
    }
    (team_states, team_trans, state_index_counter, trans_index_counter, current_init_agent_state, current_init_task_state)
}

#[allow(dead_code)]
pub fn create_ij_state_mapping(num_tasks: usize, num_agents: usize, states: &[TeamState]) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(num_agents * num_tasks);
    for j in (0..num_tasks).rev() {
        for i in (0..num_agents).rev() {
            let range: Vec<_> = states.iter().filter(|x| x.agent == i && x.task == j).map(|x| x.ix).collect();
            ranges.push((range[0], *range.last().unwrap()))
        }
    }
    return ranges
}

#[allow(dead_code)]
pub fn opt_exp_tot_cost<'a>(w: &[f64], eps: &f64, ranges: &'a [(usize, usize)], states: &'a [TeamState<'a>], transitions: &'a [TeamTransition],
                            rewards: &Rewards, num_agents: &usize, num_tasks: &usize, init_index: &'a usize)
                            -> Option<(Vec<String>, Vec<f64>)> {
    let mut mu: Vec<String> = vec!["".to_string(); states.len()];
    let mut r: Vec<f64> = vec![0.0; w.len()];
    let weight = arr1(w);

    let mut x_cost_vectors: Vec<f64> = vec![0f64; states.len()];
    let mut y_cost_vectors: Vec<f64> = vec![0f64; states.len()];
    // There is another optimisation here where we can flatten these 2d ndarray to 1d vector which will be faster
    // but this won't be a huge optimisation because ndarray is meant for these operations
    let mut x_agent_cost_vector: Vec<f64> = vec![0.0; states.len() * *num_agents];
    let mut y_agent_cost_vector: Vec<f64> = vec![0.0; states.len() * *num_agents];

    let mut x_task_cost_vector: Vec<f64> = vec![0.0; states.len() * *num_tasks];
    let mut y_task_cost_vector: Vec<f64> = vec![0.0; states.len() * *num_tasks];
    let mut ij_counter: usize = 0;

    for _j in (0..*num_tasks).rev() {
        for _i in (0..*num_agents).rev() {
            let mut epsilon: f64 = 1.0;
            let (s_start, s_end) = ranges[ij_counter];
            while epsilon > *eps {
                for s_ix in s_start..s_end {
                    let s = &states[s_ix];
                    // absolutely limit the number of vector resizes
                    let mut min_action_values: Vec<(String, f64)> = Vec::with_capacity(s.trans_ix.len());
                    for t in s.trans_ix.iter() {
                        let transition = &transitions[*t];
                        let transition_rewards = arr1(&transition.reward);
                        let scaled_weight_rewards = weight.dot(&transition_rewards);
                        let mut sum_vect: Vec<f64> = vec![0.0; transition.to.len()];
                        for (z, sprime) in transition.to.iter().enumerate(){
                            sum_vect[z] = sprime.p * x_cost_vectors[sprime.state];
                        }
                        let sum_vect_sum: f64 = sum_vect.iter().fold(0f64, |sum, &val| sum + val);
                        let action_reward = scaled_weight_rewards + sum_vect_sum;
                        min_action_values.push((transition.a.to_string(), action_reward));
                    }
                    min_action_values.sort_by(|(_a1, a2), (_b1, b2)| a2.partial_cmp(b2).unwrap());
                    //println!("s: ({},{},{},{}), ix:{}, min a: {:?}", s.state.state.s, s.state.state.q, s.agent, s.task, s.ix, min_action_values);
                    let minmax_pair: &(String, f64) = match rewards {
                        Rewards::NEGATIVE => &min_action_values.last().unwrap(),
                        Rewards::POSITIVE => &min_action_values[0]
                    };
                    let minmax_val: f64 = minmax_pair.1;
                    let arg_minmax = &minmax_pair.0;

                    y_cost_vectors[s.ix] = minmax_val;
                    mu[s.ix] = arg_minmax.to_string();
                }
                let mut y_bar_diff = opt_absolute_diff_vect(&x_cost_vectors, &y_cost_vectors).to_vec();
                y_bar_diff.sort_by(|a, b| a.partial_cmp(b).unwrap());
                //println!("ybar: {:?}", y_bar_diff);
                //y_bar_diff_max_vect.sort();
                epsilon = y_bar_diff[y_bar_diff.len() - 1];
                //println!("eps: {}", epsilon);
                x_cost_vectors = y_cost_vectors.to_owned();
            }
            epsilon = 1.0;
            while epsilon > *eps {
                for s_ix in s_start..s_end {
                    let s = &states[s_ix];
                    let chosen_action = &mu[s.ix];
                    for t in s.trans_ix.iter() {
                        let transition = &transitions[*t];
                        if transition.a == chosen_action {
                            let mut sum_vect_agent: Vec<f64> = vec![0.0; *num_agents * transition.to.len()]; //vec![vec![0.0; transition.to.len()]; *num_agents];
                            let mut sum_vect_task: Vec<f64> = vec![0.0; *num_tasks * transition.to.len()]; //vec![vec![0.0; transition.to.len()]; *num_tasks];
                            for (l, sprime) in transition.to.iter().enumerate(){
                                for agent in 0..*num_agents {
                                    sum_vect_agent[agent * transition.to.len() + l] = sprime.p * x_agent_cost_vector[agent * states.len() + sprime.state];
                                }
                                for task in 0..*num_tasks {
                                    sum_vect_task[task * transition.to.len() + l] = sprime.p * x_task_cost_vector[task * states.len() + sprime.state];
                                }
                            }
                            for agent in 0..*num_agents {
                                let start: usize = agent * transition.to.len();
                                let end: usize = start + transition.to.len();
                                let p_trans_agent: f64 = sum_vect_agent[start..end].iter().sum();
                                y_agent_cost_vector[agent * states.len() + s.ix] = transition.reward[agent] + p_trans_agent;
                            }
                            for task in 0..*num_tasks {
                                let start: usize = task * transition.to.len();
                                let end: usize = start + transition.to.len();
                                let p_trans_agent: f64 = sum_vect_task[start..end].iter().sum();
                                y_task_cost_vector[task * states.len() + s.ix] = transition.reward[*num_agents + task] + p_trans_agent;
                            }
                        }
                    }
                }
                let mut diff_task = opt_absolute_diff_vect(&x_task_cost_vector[..], &y_task_cost_vector[..]).to_vec();
                diff_task.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let max_task_val = diff_task[diff_task.len()-1];
                let mut diff_agent = opt_absolute_diff_vect(&x_agent_cost_vector[..], &y_agent_cost_vector[..]).to_vec();
                diff_agent.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let max_agent_val = diff_agent[diff_agent.len()-1];
                x_agent_cost_vector = y_agent_cost_vector.to_owned();
                x_task_cost_vector = y_task_cost_vector.to_owned();
                if max_task_val > max_agent_val {
                    epsilon = max_task_val;
                } else {
                    epsilon = max_agent_val;
                }
            }
            ij_counter += 1;
        }
    }
    //println!("init ix: {}", init_index);
    //println!("y_agent_cost: {:?}", y_agent_cost_vector);
    //println!("y_task_cost: {:?}", y_task_cost_vector);
    for agent in 0..*num_agents {
        r[agent] = y_agent_cost_vector[agent * states.len() + *init_index];
    }

    for task in 0..*num_tasks {
        //println!("task: {} ix: {}", task, task * states.len() + *init_index);
        r[task + *num_agents] = y_task_cost_vector[task * states.len() + *init_index];
    }
    //println!("new r: {:?}", r);
    Some((mu, r))
}

#[allow(dead_code)]
pub fn opt_exp_tot_cost_non_iter<'a>(w: &[f64], eps: &f64, states: &'a [TeamState<'a>], transitions: &'a [TeamTransition],
                            rewards: &Rewards, num_agents: &usize, num_tasks: &usize, init_index: &'a usize)
                            -> Option<(Vec<String>, Vec<f64>)> {
    let mut mu: Vec<String> = vec!["".to_string(); states.len()];
    // inserting into a hashmap might slow things down a lot, should check this
    let mut r: Vec<f64> = vec![0.0; w.len()];
    let weight = arr1(w);

    let mut x_cost_vectors: Vec<f64> = vec![0f64; states.len()];
    let mut y_cost_vectors: Vec<f64> = vec![0f64; states.len()];
    // There is another optimisation here where we can flatten these 2d ndarray to 1d vector which will be faster
    // but this won't be a huge optimisation because ndarray is meant for these operations
    let mut x_agent_cost_vector: Vec<f64> = vec![0.0; states.len() * *num_agents];
    let mut y_agent_cost_vector: Vec<f64> = vec![0.0; states.len() * *num_agents];

    let mut x_task_cost_vector: Vec<f64> = vec![0.0; states.len() * *num_tasks];
    let mut y_task_cost_vector: Vec<f64> = vec![0.0; states.len() * *num_tasks];

    let mut epsilon: f64 = 1.0;
    //let mut counter: u32 = 0;
    while epsilon > *eps {
        for s in states.iter() {
            let mut min_action_values: Vec<(String, f64)> = Vec::with_capacity(s.trans_ix.len());
            for t in s.trans_ix.iter() {
                let transition = &transitions[*t];
                let transition_rewards = arr1(&transition.reward);
                let scaled_weight_rewards = weight.dot(&transition_rewards);
                let mut sum_vect: Vec<f64> = vec![0.0; transition.to.len()];
                for (z, sprime) in transition.to.iter().enumerate(){
                    sum_vect[z] = sprime.p * x_cost_vectors[sprime.state];
                }
                let sum_vect_sum: f64 = sum_vect.iter().fold(0f64, |sum, &val| sum + val);
                let action_reward = scaled_weight_rewards + sum_vect_sum;
                min_action_values.push((transition.a.to_string(), action_reward));
            }
            min_action_values.sort_by(|(_a1, a2), (_b1, b2)| a2.partial_cmp(b2).unwrap());
            //println!("min a: {:?}", min_action_values);
            let   minmax_pair: &(String, f64) = match rewards {
                Rewards::NEGATIVE => &min_action_values.last().unwrap(),
                Rewards::POSITIVE => &min_action_values[0]
            };
            let minmax_val: f64 = minmax_pair.1;
            let arg_minmax = &minmax_pair.0;

            y_cost_vectors[s.ix] = minmax_val;
            mu[s.ix] = arg_minmax.to_string();
        }
        let mut y_bar_diff = opt_absolute_diff_vect(&x_cost_vectors, &y_cost_vectors).to_vec();
        y_bar_diff.sort_by(|a,b| a.partial_cmp(b).unwrap());
        //y_bar_diff_max_vect.sort();
        epsilon = *y_bar_diff.last().unwrap();
        //println!("non iter loops: {}", counter);
        //counter += 1;
        x_cost_vectors = y_cost_vectors.to_vec();
    }
    epsilon = 1.0;
    while epsilon > *eps {
        for s in states.iter() {
            let chosen_action: &String = &mu[s.ix];
            for t in s.trans_ix.iter() {
                let transition = &transitions[*t];
                if transition.a == chosen_action {
                    let mut sum_vect_agent: Vec<f64> = vec![0.0; *num_agents * transition.to.len()]; //vec![vec![0.0; transition.to.len()]; *num_agents];
                    let mut sum_vect_task: Vec<f64> = vec![0.0; *num_tasks * transition.to.len()]; //vec![vec![0.0; transition.to.len()]; *num_tasks];
                    for (l, sprime) in transition.to.iter().enumerate(){
                        for agent in 0..*num_agents {
                            sum_vect_agent[agent * transition.to.len() + l] = sprime.p * x_agent_cost_vector[agent * states.len() + sprime.state];
                        }
                        for task in 0..*num_tasks {
                            sum_vect_task[task * transition.to.len() + l] = sprime.p * x_task_cost_vector[task * states.len() + sprime.state];
                        }
                    }
                    for agent in 0..*num_agents {
                        let start: usize = agent * transition.to.len();
                        let end: usize = start + transition.to.len();
                        let p_trans_agent: f64 = sum_vect_agent[start..end].iter().sum();
                        y_agent_cost_vector[agent * states.len() + s.ix] = transition.reward[agent] + p_trans_agent;
                    }
                    for task in 0..*num_tasks {
                        let start: usize = task * transition.to.len();
                        let end: usize = start + transition.to.len();
                        let p_trans_agent: f64 = sum_vect_task[start..end].iter().sum();
                        y_task_cost_vector[task * states.len() + s.ix] = transition.reward[*num_agents + task] + p_trans_agent;
                    }
                }
            }
        }
        let mut diff_task = opt_absolute_diff_vect(&x_task_cost_vector[..], &y_task_cost_vector[..]).to_vec();
        diff_task.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let max_task_val = diff_task.last().unwrap();
        let mut diff_agent = opt_absolute_diff_vect(&x_agent_cost_vector[..], &y_agent_cost_vector[..]).to_vec();
        diff_agent.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let max_agent_val = diff_task.last().unwrap();
        x_agent_cost_vector = y_agent_cost_vector.to_vec();
        x_task_cost_vector = y_task_cost_vector.to_vec();
        if max_task_val > max_agent_val {
            epsilon = *max_task_val;
        } else {
            epsilon = *max_agent_val;
        }
    }

    for agent in 0..*num_agents {
        r[agent] = y_agent_cost_vector[agent * states.len() + *init_index];
    }

    for task in 0..*num_tasks {
        r[task + *num_agents] = y_task_cost_vector[task * states.len() + *init_index];
    }
    //println!("new r: {:?}", r);
    Some((mu, r))
}

#[allow(dead_code)]
pub fn merge_schedulers (schedulers: &Vec<Vec<String>>, team_states: &[TeamState], transitions: &[TeamTransition],
                         init_state: usize, tasks: usize, weights: &[f64]) -> Graph<String,String> {
    let mut stack: Vec<DFSStackState> = Vec::new();
    let mut graph: Graph<String, String> = Graph::new();
    let init_state = team_states[init_state].ix;
    let c_0: Vec<usize> = (0..tasks).collect();
    let init_g_ix = graph.add_node(format!("({},{:?})", init_state, c_0));
    let init_stack_state: DFSStackState = DFSStackState {
        state: init_state, indices: c_0, parent_action: None,
        parent_prob: None, parent_node_index: None, parent_indices: None
    };
    stack.push(init_stack_state);
    //println!("{:?}", stack);
    // a sketch of the algo
    // |
    // |_____ find root |
    //                  |__ If root add root transitions and (s',C') manually
    // Start iterative part of algorithm:
    // |______ find parent and transitions for parent |
    //                                                |____ Add transition (s',C') to stack
    //let mut count = 0;
    while !stack.is_empty() {
        let new_stack_state = stack.pop().unwrap();
        // if the parent state is not present then this is the root and we are only interested in adding more stack states
        if new_stack_state.parent_node_index.is_none() {
            for t in team_states[new_stack_state.state].trans_ix.iter() {
                let transition = &transitions[*t];
                let mut c_prime: Vec<usize> = Vec::new();
                for task in 0..tasks {
                    if *transition.a == schedulers[task][team_states[new_stack_state.state].ix] {
                        c_prime.push(task);
                    }
                }
                let numerator = weights.iter().enumerate().
                    filter(|(i,_x)| c_prime.iter().any(|j| i == j)).
                    fold(0.0, |sum, (_j, val)| sum + val);

                let denominator = weights.iter().enumerate().
                    filter(|(i,_x)| new_stack_state.indices.iter().any(|j| i == j)).
                    fold(0.0, |sum, (_j, val)| sum + val);
                let p = numerator / denominator;
                if p > 0.0 {
                    for sprime in transition.to.iter() {
                        let new_stack_state = DFSStackState {
                            state: sprime.state,
                            indices: c_prime.to_vec(),
                            parent_action: Some(transition.a),
                            parent_prob: Some(OrderedFloat(p)),
                            parent_node_index: Some(init_g_ix),
                            parent_indices: Some(new_stack_state.indices.to_vec())
                        };
                        if !stack.iter().any(|x| team_states[x.state].ix == team_states[new_stack_state.state].ix) {
                            stack.push(new_stack_state);
                        }
                    }
                }
            }
        } else {
            // create a vertex of the current stack state
            //   first check that the vertex does not already exist
            let new_node_index = match graph.node_indices().find(|x| graph[*x] == format!("({},{:?})", new_stack_state.state, new_stack_state.indices)) {
                None => {
                    //println!("{} added to graph", format!("({},{:?})", new_stack_state.state, new_stack_state.indices));
                    graph.add_node(format!("({},{:?})", new_stack_state.state, new_stack_state.indices))
                }
                Some(x) => x
            };
            // create an edge between the newly added vertex and the parent node index
            graph.add_edge(
                new_stack_state.parent_node_index.unwrap(),
                new_node_index,
                format!("({},{})", new_stack_state.parent_action.unwrap(), new_stack_state.parent_prob.unwrap())
            );
            for t in team_states[new_stack_state.state].trans_ix.iter() {
                let transition = &transitions[*t];
                let mut c_prime: Vec<usize> = Vec::new();
                for task in 0..tasks {
                    if *transition.a == schedulers[task][new_stack_state.state] {
                        c_prime.push(task);
                    }
                }
                let numerator = weights.iter().enumerate().
                    filter(|(i,_x)| c_prime.iter().any(|j| i == j)).
                    fold(0.0, |sum, (_j, val)| sum + val);

                let denominator = weights.iter().enumerate().
                    filter(|(i,_x)| new_stack_state.indices.iter().any(|j| i == j)).
                    fold(0.0, |sum, (_j, val)| sum + val);
                for sprime in transition.to.iter() {
                    let p = numerator / denominator;
                    if p > 0.0 {
                        let sprime_stack_state = DFSStackState {
                            state: sprime.state,
                            indices: c_prime.to_vec(),
                            parent_action: Some(transition.a),
                            parent_prob: Some(OrderedFloat(p)),
                            parent_node_index: Some(new_node_index),
                            parent_indices: Some(new_stack_state.indices.to_vec())
                        };
                        if !stack.iter().any(|x| team_states[x.state].ix == team_states[sprime_stack_state.state].ix && x.indices == sprime_stack_state.indices) {
                            match graph.node_indices().find(|x| graph[*x] == format!("({},{:?})", sprime_stack_state.state, sprime_stack_state.indices)) {
                                None => stack.push(sprime_stack_state),
                                Some(_) => {}
                            }
                        }
                    }
                }
            }
        }
        /*
        if count < 5 {
            for s in stack.iter() {
                println!("{:?}", s);
            }
            println!();
        } else {
            return graph;
        }
        count += 1;

         */
    }
    graph
}

#[derive(Debug, PartialEq, Clone, Eq)]
pub struct TeamState<'a> {
    pub state: &'a DFAModelCheckingPair<'a>,
    pub agent: usize,
    pub task: usize,
    pub trans_ix: Vec<usize>,
    pub label: HashSet<&'a str>,
    pub ix: usize
}

#[derive(Debug, PartialEq)]
pub struct TeamTransition<'a> {
    pub from: usize,
    pub a: &'a String,
    pub to: Vec<TeamTransitionPair>,
    pub reward: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TeamTransitionPair {
    pub state: usize,
    pub p: f64
}

#[derive(Debug)]
pub struct TeamLabelling<'a> {
    pub state: TeamState<'a>,
    pub label: Vec<String>
}

#[derive(Debug)]
pub struct TaskAllocStates<'a> {
    pub index: usize,
    pub state: TeamState<'a>,
    pub value: Vec<f64>
}

#[allow(dead_code)]
pub struct Alg1Output {
    pub v: Vec<f64>,
    pub mu: Vec<Vec<String>>,
    pub hullset: Vec<Vec<f64>>
}

#[allow(dead_code)]
pub struct TeamAttrs<'a> {
    pub agent: usize,
    pub task: usize,
    pub dead: &'a Vec<u32>,
    pub acc: &'a Vec<u32>,
    pub jacc: &'a Vec<u32>
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct DFSStackState<'a> {
    pub state: usize,
    pub indices: Vec<usize>,
    pub parent_action: Option<&'a String>,
    pub parent_prob: Option<OrderedFloat<f64>>,
    pub parent_node_index: Option<NodeIndex<u32>>,
    pub parent_indices: Option<Vec<usize>>
}

