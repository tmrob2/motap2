use super::helper_methods;
use helper_methods::*;
use super::product_automata;
use product_automata::*;
use ndarray::{arr1};
use super::gurobi_lp;
use gurobi_lp::*;
// ------------------------------------
// STAPU Team MDP Structure
// -------------------------------------
#[allow(dead_code)]
pub fn create_state_trans_index_mapping<'a, 'b>(loc_prod_states: &[LocalProdState], loc_prod_trans: &'a [LocalProdTransitions], loc_prod_init: usize,
                                            prev_init_index: usize, agent: usize, current_state_ix: usize, current_transition_ix: usize,
                                            num_agents: usize, num_tasks: usize, rewards: &'a Rewards,
                                            team_states: &'b mut Vec<TeamState>, team_transitions: &'b mut Vec<TeamTransition>,
                                            switch_from_input: &'a [usize])
                                            -> (&'b mut Vec<TeamState>, &'b mut Vec<TeamTransition>, Vec<usize>, usize, usize, usize) {
    let reward_coeff = match rewards {
        Rewards::NEGATIVE => -1.0,
        Rewards::POSITIVE => 1.0
    };
    // the number of new team states will be exactly the same as the number of local product states
    let mut team_transition_ix = current_transition_ix;
    let mut team_state_ix = current_state_ix;
    let mut switch_from_output: Vec<usize> = Vec::new();
    let init_state: usize = loc_prod_init + current_state_ix;
    //println!("init state: {}, current_state_ix: {}", init_state, current_state_ix);
    for state in loc_prod_states.iter() {
        if state.switch_from {
            switch_from_output.push(team_state_ix);
            //println!("{},{},{:?}", team_state_ix, state.ix, state.desc);
        }
        let mut team_state = TeamState {
            s: state.s,
            q: state.q,
            desc: state.desc.to_string(),
            agent,
            trans_ix: vec![],
            ix: current_state_ix + state.ix
        };
        for transition in loc_prod_trans.iter().filter(|x| state.ix == x.sq) {
            let mut sprime_team_trans_pair: Vec<TeamTransitionPair> = Vec::with_capacity(transition.sq_prime.len());
            for sprime in transition.sq_prime.iter() {
                sprime_team_trans_pair.push(TeamTransitionPair{ s: current_state_ix + sprime.s, p: sprime.p });
            }
            let mut rewards: Vec<f64> = vec![0.0; num_agents + num_tasks];
            if state.dfa_jacc {
                rewards[num_agents + state.active_ix as usize] = 1.0;
            } else {
                if state.done {
                    // do nothing
                } else {
                    rewards[agent] = reward_coeff * transition.reward;
                }
            }
            let new_team_transition = TeamTransition {
                s: state.ix + current_state_ix,
                a: transition.a.to_string(),
                s_prime: sprime_team_trans_pair,
                reward: rewards
            };
            team_transitions.push(new_team_transition);
            team_state.trans_ix.push(team_transition_ix);
            team_transition_ix += 1;
        }
        if agent > 0 {
            if team_state_ix == init_state {
                /*println!("({},{},{}) ->init-> ({},{},{})",
                         team_states[prev_init_index].s, team_states[prev_init_index].q, agent-1,
                         state.s, state.q, agent
                );*/
                team_transitions.push(TeamTransition {
                    s: prev_init_index,
                    a: "swi".to_string(),
                    s_prime: vec![TeamTransitionPair { s: team_state_ix, p: 1.0 }],
                    reward: vec![0.0; num_tasks + num_agents]
                });
                team_states[prev_init_index].trans_ix.push(team_transition_ix);
                team_transition_ix += 1;
            }
        }
        if state.switch_to {
            // find the previous switch from transition
            for switch_prev_ix in 0..switch_from_input.len() {
                if team_states[switch_from_input[switch_prev_ix]].q == state.q && team_states[switch_from_input[switch_prev_ix]].s == state.s {
                    /*println!("ix:{};({},{},{}) ->b-> ix:{};({},{},{})",
                             switch_from_input[switch_prev_ix],team_states[switch_from_input[switch_prev_ix]].s,
                             team_states[switch_from_input[switch_prev_ix]].q, team_states[switch_from_input[switch_prev_ix]].agent,
                             team_state_ix, state.s, state.q, team_state.agent
                    );*/
                    team_transitions.push(TeamTransition {
                        s: switch_from_input[switch_prev_ix],
                        a: "swi".to_string(),
                        s_prime: vec![TeamTransitionPair { s: team_state_ix, p: 1.0 }],
                        reward: vec![0.0; num_tasks + num_agents]
                    });
                    team_states[switch_from_input[switch_prev_ix]].trans_ix.push(team_transition_ix);
                    team_transition_ix += 1;
                }
            }
        }
        team_states.push(team_state);
        team_state_ix += 1;
    }
    //println!("init state: {}", init_state);
    (team_states, team_transitions, switch_from_output, team_transition_ix, team_state_ix, init_state)
}

#[allow(dead_code)]
pub fn multi_obj_sched_synth_non_iter(init_state_ix: usize, states: &[TeamState], transitions: &[TeamTransition],
                                      target: &Vec<f64>, eps: &f64, rewards: &Rewards, verbose: &bool, num_tasks: usize,
                                      num_agents: usize)
                                      -> Alg1Output {
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
    let team_init_index = init_state_ix;

    let mut extreme_points: Vec<Vec<f64>> = vec![vec![0.0; num_agents + num_tasks]; num_agents + num_tasks];

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
pub fn opt_exp_tot_cost_non_iter<'a>(w: &[f64], eps: &f64, states: &'a [TeamState], transitions: &'a [TeamTransition],
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
                let mut sum_vect: Vec<f64> = vec![0.0; transition.s_prime.len()];
                for (z, sprime) in transition.s_prime.iter().enumerate(){
                    sum_vect[z] = sprime.p * x_cost_vectors[sprime.s];
                }
                let sum_vect_sum: f64 = sum_vect.iter().fold(0f64, |sum, &val| sum + val);
                let action_reward = scaled_weight_rewards + sum_vect_sum;
                min_action_values.push((transition.a.to_string(), action_reward));
            }
            min_action_values.sort_by(|(_a1, a2), (_b1, b2)| a2.partial_cmp(b2).unwrap());
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
                if transition.a == *chosen_action {
                    let mut sum_vect_agent: Vec<f64> = vec![0.0; *num_agents * transition.s_prime.len()]; //vec![vec![0.0; transition.to.len()]; *num_agents];
                    let mut sum_vect_task: Vec<f64> = vec![0.0; *num_tasks * transition.s_prime.len()]; //vec![vec![0.0; transition.to.len()]; *num_tasks];
                    for (l, sprime) in transition.s_prime.iter().enumerate(){
                        for agent in 0..*num_agents {
                            sum_vect_agent[agent * transition.s_prime.len() + l] = sprime.p * x_agent_cost_vector[agent * states.len() + sprime.s];
                        }
                        for task in 0..*num_tasks {
                            sum_vect_task[task * transition.s_prime.len() + l] = sprime.p * x_task_cost_vector[task * states.len() + sprime.s];
                        }
                    }
                    for agent in 0..*num_agents {
                        let start: usize = agent * transition.s_prime.len();
                        let end: usize = start + transition.s_prime.len();
                        let p_trans_agent: f64 = sum_vect_agent[start..end].iter().sum();
                        y_agent_cost_vector[agent * states.len() + s.ix] = transition.reward[agent] + p_trans_agent;
                    }
                    for task in 0..*num_tasks {
                        let start: usize = task * transition.s_prime.len();
                        let end: usize = start + transition.s_prime.len();
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
        //println!("task y: {:?}", y_task_cost_vector);
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

pub struct Alg1Output {
    pub v: Vec<f64>,
    pub mu: Vec<Vec<String>>,
    pub hullset: Vec<Vec<f64>>
}

#[allow(dead_code)]
pub struct TeamMDP {
    pub states: Vec<TeamState>,
    pub transitions: Vec<TeamTransition>
}

#[derive(Debug)]
pub struct TeamState {
    pub s: u32,
    pub q: u32,
    pub desc: String,
    pub agent: usize,
    pub trans_ix: Vec<usize>,
    pub ix: usize
}

#[derive(Debug)]
pub struct TeamTransition {
    pub s: usize,
    pub a: String,
    pub s_prime: Vec<TeamTransitionPair>,
    pub reward: Vec<f64>
}

#[derive(Debug)]
pub struct TeamTransitionPair {
    pub s: usize,
    pub p: f64
}