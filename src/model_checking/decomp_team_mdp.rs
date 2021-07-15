use super::helper_methods;
use super::dfa;
//use super::mdp2;
use super::gurobi_lp;
//use petgraph::{Graph};
use std::collections::{HashSet};
use ndarray::{arr1};

use helper_methods::*;
use dfa::*;
//use mdp2::*;
use gurobi_lp::*;
//use std::time::Instant;

pub struct TeamMDP<'a> {
    pub initial: TeamState<'a>,
    pub states: Vec<TeamState<'a>>,
    pub transitions: Vec<TeamTransition<'a>>,
    pub num_agents: usize,
    pub num_tasks: usize,
    //pub task_alloc_states: Vec<TaskAllocStates<'a>>,
}

impl <'a>TeamMDP<'a> {

    /// Iterative version of multi-objective model checking
    #[allow(dead_code)]
    pub fn multi_obj_sched_synth(&self, target: &Vec<f64>, eps: &f64, rewards: &Rewards, verbose: &bool) -> Alg1Output {
        let mut hullset: Vec<Vec<f64>> = Vec::new();
        let mut mu_vect: Vec<Vec<String>> = Vec::new();
        let mut alg1_output: Alg1Output = Alg1Output{
            v: vec![],
            mu: vec![],
            hullset: vec![]
        };
        if *verbose {
            println!("num tasks: {}, num agents {}", self.num_tasks, self.num_agents);
        }

        let team_init_index = self.initial.ix;

        let mut extreme_points: Vec<Vec<f64>> = vec![vec![0.0; self.num_agents + self.num_tasks]; self.num_agents + self.num_tasks];

        for k in 0..(self.num_tasks + self.num_agents) {
            extreme_points[k][k] = 1.0;
            let w_extr: &Vec<f64> = &extreme_points[k];
            if *verbose {
                println!("w: {:?}", w_extr);
            }
            let safe_ret = opt_exp_tot_cost(&w_extr[..], &eps, &self.states[..], &self.transitions[..],
                                            &rewards, &self.num_agents, &self.num_tasks, &team_init_index);
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
        let dim = self.num_tasks + self.num_agents;
        let t_arr1 = arr1(target);
        let mut w_new = lp5(&hullset, target, &dim);
        while w_new != None {
            if *verbose {
                println!("w' :{:?}", w_new);
            }
            let safe_ret = opt_exp_tot_cost(&w_new.as_ref().unwrap()[..], &eps, &self.states[..], &self.transitions[..],
                                            &rewards, &self.num_agents, &self.num_tasks, &team_init_index);
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

    /// The non iterative version of multi-objective value iteration, just loops over the state space to generate results
    #[allow(dead_code)]
    pub fn multi_obj_sched_synth_non_iter(&self, target: &Vec<f64>, eps: &f64, rewards: &Rewards, verbose: &bool) -> Alg1Output {
        let mut hullset: Vec<Vec<f64>> = Vec::new();
        let mut mu_vect: Vec<Vec<String>> = Vec::new();
        let mut alg1_output: Alg1Output = Alg1Output{
            v: vec![],
            mu: vec![],
            hullset: vec![]
        };
        if *verbose {
            println!("num tasks: {}, num agents {}", self.num_tasks, self.num_agents);
        }
        let team_init_index = &self.initial.ix;

        let mut extreme_points: Vec<Vec<f64>> = vec![vec![0.0; self.num_agents + self.num_tasks]; self.num_agents + self.num_tasks];

        for k in 0..(self.num_tasks + self.num_agents) {
            extreme_points[k][k] = 1.0;
            let w_extr: &Vec<f64> = &extreme_points[k];
            if *verbose {
                println!("w: {:?}", w_extr);
            }

            let safe_ret = opt_exp_tot_cost_non_iter(&w_extr[..], &eps, &self.states[..], &self.transitions[..],
                                            rewards, &self.num_agents, &self.num_tasks, &team_init_index);
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
        let dim = self.num_tasks + self.num_agents;
        let t_arr1 = arr1(target);
        let mut w_new = lp5(&hullset, target, &dim);
        while w_new != None {
            if *verbose {
                println!("w' :{:?}", w_new);
            }
            let safe_ret = opt_exp_tot_cost_non_iter(&w_new.as_ref().unwrap()[..], &eps, &self.states[..], &self.transitions[..],
                                            rewards, &self.num_agents, &self.num_tasks, &team_init_index);
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
    pub fn statistics(&self) -> (usize,usize) {
        (self.states.len(), self.transitions.len())
    }
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
                rewards[num_agents + task] = 1.0;
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
pub fn opt_exp_tot_cost<'a>(w: &[f64], eps: &f64, states: &'a [TeamState<'a>], transitions: &'a [TeamTransition],
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

    for j in (0..*num_tasks).rev() {
        for i in (0..*num_agents).rev() {
            let mut epsilon: f64 = 1.0;
            while epsilon > *eps {
                for s in states.iter().filter(|x| x.task == j && x.agent == i) {
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
                y_bar_diff.sort_by(|a,b| a.partial_cmp(b).unwrap());
                //y_bar_diff_max_vect.sort();
                epsilon = *y_bar_diff.last().unwrap();
                //println!("eps: {}", epsilon);
                x_cost_vectors = y_cost_vectors.to_vec();
            }
            //println!("ybar: {:?}", y_cost_vectors);
            //println!("mu: {:?}", mu);
            epsilon = 1.0;
            while epsilon > *eps {
                for s in states.iter().filter(|x| x.agent == i && x.task == j) {
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
                //println!("epsilon: {:?}", epsilon);
            }
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

pub struct Alg1Output {
    pub v: Vec<f64>,
    pub mu: Vec<Vec<String>>,
    pub hullset: Vec<Vec<f64>>
}

