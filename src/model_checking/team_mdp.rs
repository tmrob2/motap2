use super::helper_methods;
use super::dfa;
use super::mdp;
use super::product_dfa;
use super::product_dfa_product_mdp;
use super::gurobi_lp;
use petgraph::{Graph, graph::NodeIndex};
use std::collections::{HashMap};
use ndarray::{arr1};

use helper_methods::*;
use product_dfa_product_mdp::*;
use gurobi_lp::*;
use crate::model_checking::decomp_team_mdp::{TeamStateIndex, Rewards};

pub struct DFAProductTeamMDP<'a> {
    pub initial: ProductDFATeamState<'a>,
    pub states: Vec<ProductDFATeamState<'a>>,
    pub transitions: Vec<ProductDFATeamTransition<'a>>,
    pub num_tasks: usize,
    pub num_agents: usize,
}

impl <'a>DFAProductTeamMDP<'a> {

    pub fn generate_team_graph(&self) -> Graph<String,String> {
        let mut g: Graph<String, String> = Graph::new();
        for state in self.states.iter() {
            g.add_node(format!("({},{:?},{})", state.s, state.q, state.agent));
        }
        for transition in self.transitions.iter() {
            let origin_index: NodeIndex = g.node_indices().
                find(|x| g[*x] ==
                    format!("({},{:?},{})", transition.from.s, transition.from.q, transition.from.agent)).unwrap();
            for sprime in transition.to.iter() {
                let destination_index: NodeIndex = g.node_indices().
                    find(|x| g[*x] == format!("({},{:?},{})", sprime.s.s, sprime.s.q, sprime.s.agent)).unwrap_or_else(|| panic!("s: {},{:?},{} ->s': {},{:?},{}",transition.from.s, transition.from.q, transition.from.agent, sprime.s.s, sprime.s.q, sprime.s.agent));
                g.add_edge(origin_index, destination_index, transition.a.to_string());
            }
        }
        g
    }

    pub fn team_ij_index_mapping(&self) -> HashMap<usize, DFAProductTeamStateIndexHelper> {
        let mut state_index: HashMap<usize, DFAProductTeamStateIndexHelper> = HashMap::new();
        for i in 0..self.num_agents {
            let state_indices: Vec<(&ProductDFATeamState, TeamStateIndex)> = self.states.iter().enumerate().
                filter(|(_k,x)| x.agent == i).enumerate().
                map(|(z,(k, x))| (x,TeamStateIndex{
                    local_index: z,
                    team_index: k
                })).collect();
            let mut state_mapping_hash: HashMap<&ProductDFATeamState, TeamStateIndex> = HashMap::new();
            for (state, k) in state_indices.iter() {
                state_mapping_hash.insert(*state, TeamStateIndex{local_index: k.local_index, team_index: k.team_index});
            }
            state_index.insert(
                i,
                DFAProductTeamStateIndexHelper {
                    state_index_map: state_indices,
                    state_hashmap: state_mapping_hash
                }
            );
        }
        state_index
    }

    pub fn exp_tot_cost<'b>(&'a self, w: &'b [f64], eps: &'a f64, team_index: &'a HashMap<usize, DFAProductTeamStateIndexHelper>, rewards: &'a Rewards) -> Option<(HashMap<&'a ProductDFATeamState, String>, Vec<f64>)>{
        let mut mu: HashMap<&'a ProductDFATeamState, String> = HashMap::new();
        let mut r: Vec<f64> = vec![0.0; w.len()];
        let weight = arr1(w);

        let test_state = ProductDFATeamState {
            s: &0,
            q: &vec![0,0,0],
            agent: 0
        };

        let mut x_cost_vectors: Vec<Vec<f64>> = vec![Vec::new(); self.num_agents];
        let mut y_cost_vectors: Vec<Vec<f64>> = vec![Vec::new(); self.num_agents];

        for agent in 0..self.num_agents {
            let u = team_index.get(&agent).unwrap();
            let carinality_u = u.state_hashmap.len();
            x_cost_vectors[agent] = vec![0.0; carinality_u];
            y_cost_vectors[agent] = vec![0.0; carinality_u];
        }
        // agent cost vectors
        let mut x_agent_cost_vector: Vec<Vec<f64>> = vec![vec![0.0; self.states.len()]; self.num_agents];
        let mut y_agent_cost_vector: Vec<Vec<f64>> = vec![vec![0.0; self.states.len()]; self.num_agents];
        // task cost vectors
        let mut x_task_cost_vector: Vec<Vec<f64>> = vec![vec![0.0; self.states.len()]; self.num_tasks];
        let mut y_task_cost_vector: Vec<Vec<f64>> = vec![vec![0.0; self.states.len()]; self.num_tasks];

        let mut counter = 0;

        for i in (0..self.num_agents).rev() {
            let helper_ij = team_index.get(&i).unwrap();
            let helper_iplus1j: Option<&DFAProductTeamStateIndexHelper> = if i < self.num_agents - 1 {
                 Some(team_index.get(&(i+1)).unwrap())
            } else {
                None
            };
            let mut epsilon: f64 = 1.0;
            while epsilon > *eps {
                for (state, k) in helper_ij.state_index_map.iter() {
                    let mut action_values: Vec<(String, f64)> = Vec::new();
                    for transition in self.transitions.iter().
                        filter(|x| x.from == **state) {
                        let transition_rewards = arr1(&transition.reward);
                        let scaled_weight_rewards = weight.dot(&transition_rewards);
                        let mut sum_vect: Vec<f64> = vec![0.0; transition.to.len()];
                        for (z, sprime) in transition.to.iter().enumerate(){
                            if transition.a == "swi" {
                                let x_sprime_index: &TeamStateIndex = match helper_iplus1j {
                                    None => { panic!("There should be no switch transitions in the last agent")}
                                    Some(x) => {
                                        x.state_hashmap.get(&sprime.s).unwrap()
                                    }
                                };
                                sum_vect[z] = sprime.p * x_cost_vectors[sprime.s.agent][x_sprime_index.local_index];
                            } else {
                                let x_sprime_index: &TeamStateIndex = helper_ij.state_hashmap.get(&sprime.s).unwrap();
                                sum_vect[z]= sprime.p * x_cost_vectors[i][x_sprime_index.local_index];
                            }
                        }
                        let sum_vect_sum: f64 = sum_vect.iter().fold(0f64, |sum, &val| sum + val);
                        let mut action_reward = scaled_weight_rewards + sum_vect_sum;
                        action_values.push((transition.a.to_string(), action_reward));
                        //println!("action: {:?}", action_values);
                    }
                    /*if **state == test_state {
                        println!("max_action_values: {:?}", action_values);
                    }*/
                    let mut v: Vec<_> = action_values.iter().
                        map(|(z,x)| (z, NonNan::new(*x).unwrap())).collect();
                    v.sort_by_key(|key| key.1);
                    let mut minmax_pair: &(&String, NonNan) = match rewards {
                        Rewards::NEGATIVE => v.last().unwrap_or_else(|| panic!("s: {},{:?},{}", state.s, state.q, state.agent)),
                        Rewards::POSITIVE => &v[0]
                    };
                    let mut minmax_val: f64 = minmax_pair.1.inner();
                    let argminmax = minmax_pair.0;
                    /*if **state == test_state {
                        println!("action chosen: {:?}", argminmax);
                    }*/
                    y_cost_vectors[i][k.local_index] = minmax_val;
                    mu.insert(*state, argminmax.to_string());
                }
                let y_bar_diff = absolute_diff_vect(&x_cost_vectors[i], &y_cost_vectors[i]);
                let mut y_bar_diff_max_vect: Vec<NonNan> = y_bar_diff.iter().
                    map(|x| NonNan::new(*x).unwrap()).collect();
                y_bar_diff_max_vect.sort();
                epsilon = y_bar_diff_max_vect.last().unwrap().inner();
                //println!("eps: {}", epsilon);
                x_cost_vectors[i] = y_cost_vectors[i].to_vec();
            }
            //println!("xbar: {:?}", x_cost_vectors[i]);
            /*for (s,a) in mu.iter() {
                println!("s: {},{:?},{}, a: {}", s.s, s.q, s.agent, a);
            }*/
            epsilon = 1.0;
            while epsilon > *eps {
                for (state, k) in helper_ij.state_index_map.iter() {
                    let chosen_action: &String = mu.get(*state).unwrap();
                    /*if **state == test_state {
                        println!("action: {}", chosen_action);
                    }*/
                    for transition in self.transitions.iter().
                        filter(|x| x.from == **state && x.a == *chosen_action) {
                        let mut sum_vect_agent: Vec<Vec<f64>> = vec![vec![0.0; transition.to.len()]; self.num_agents];
                        let mut sum_vect_task: Vec<Vec<f64>> = vec![vec![0.0; transition.to.len()]; self.num_tasks];
                        for (l, sprime) in transition.to.iter().enumerate() {
                            let x_sprime_index = if transition.a == "swi" {
                                match helper_iplus1j {
                                    None => {panic!("Switch to agent {}+1 but there is no next agent", i)}
                                    Some(x) => {
                                        x.state_hashmap.get(&sprime.s).unwrap()
                                    }
                                }
                            } else {
                                helper_ij.state_hashmap.get(&sprime.s).unwrap()
                            };
                            for agent in 0..self.num_agents {
                                sum_vect_agent[agent][l] = sprime.p * x_agent_cost_vector[agent][x_sprime_index.team_index];
                            }
                            for task in 0..self.num_tasks {
                                sum_vect_task[task][l] = sprime.p * x_task_cost_vector[task][x_sprime_index.team_index];
                                //if **state == test_state && task == 1 {
                                //if task == 0 && *state.s >= 999 {
                                    //println!("task {} cost, state: ({},{:?},{}) -> ({},{:?},{}), p:{}, x:{}",
                                    //         task, state.s, state.q, state.agent, sprime.s.s, sprime.s.q, sprime.s.agent, sprime.p,  x_task_cost_vector[task][x_sprime_index.team_index])
                                //}
                                //}
                            }
                        }
                        for agent in 0..self.num_agents {
                            let p_trans_agent: f64 = sum_vect_agent[agent].iter().sum();
                            y_agent_cost_vector[agent][k.team_index] = transition.reward[agent] + p_trans_agent;
                            //println!("agent reward, state: ({},{:?},{}) -> reward:{}", state.s, state.q, state.agent, transition.reward[agent])
                        }
                        for task in 0..self.num_tasks {
                            //println!("x: {:?}", x_task_cost_vector[task]);
                            let p_trans_task: f64 = sum_vect_task[task].iter().sum();
                            y_task_cost_vector[task][k.team_index] = transition.reward[self.num_agents + task] + p_trans_task;
                            /*if **state == test_state {
                                println!("task {} reward, state: ({},{:?},{}) -> reward:{:?}, reward given: {}, p trans.: {}",
                                         task, state.s, state.q, state.agent, transition.reward, transition.reward[self.num_agents + task], p_trans_task);
                            }*/
                        }
                    }
                }
                let mut eps_inner: f64 = 0.0;
                for agent in 0..self.num_agents {
                    let diff_agent = absolute_diff_vect(&x_agent_cost_vector[agent], &y_agent_cost_vector[agent]);
                    let mut diff_agent_nonan: Vec<NonNan> = diff_agent.iter().
                        map(|x| NonNan::new(*x).unwrap()).collect();
                    diff_agent_nonan.sort();
                    let min_val_agent = diff_agent_nonan[0].inner();
                    if min_val_agent > eps_inner {
                        eps_inner = min_val_agent;
                    }
                    x_agent_cost_vector[agent] = y_agent_cost_vector[agent].to_vec();
                }

                for task in 0..self.num_tasks {
                    let diff_task = absolute_diff_vect(&x_task_cost_vector[task], &y_task_cost_vector[task]);
                    let mut diff_task_nonan: Vec<NonNan> = diff_task.iter().map(|x| NonNan::new(*x).unwrap()).collect();
                    diff_task_nonan.sort();
                    let max_val_task = diff_task_nonan.last().unwrap().inner();
                    if max_val_task > eps_inner {
                        eps_inner = max_val_task;
                    }
                    x_task_cost_vector[task] = y_task_cost_vector[task].to_vec();
                    //println!("x task:{:?}", x_task_cost_vector);
                }
                epsilon = eps_inner;
                //println!("epsilon: {}", epsilon);
                /*if counter >= 2 {
                    return None;
                }
                counter += 1;*/
            }
        }

        let helper_init = team_index.get(&self.initial.agent).unwrap();
        let init_team_index = helper_init.state_hashmap.get(&self.initial).unwrap();
        for agent in 0..self.num_agents {
            r[agent] = y_agent_cost_vector[agent][init_team_index.team_index];
        }

        for task in 0..self.num_tasks {
            r[task + self.num_agents] = y_task_cost_vector[task][init_team_index.team_index];
        }

        //println!("new r: {:?}", r);
        /*for (state, action)  in mu.iter() {
            println!("state: ({},{:?},{}), action: {}", state.s, state.q, state.agent, action);
        }*/

        Some((mu, r))
    }

    pub fn multi_obj_sched_synth(&'a self, target: &'a Vec<f64>, eps: &'a f64, team_index_mapping: &'a HashMap<usize, DFAProductTeamStateIndexHelper>, rewards: &'a Rewards) -> Alg1Output<'a> {
        let mut hullset: Vec<Vec<f64>> = Vec::new();
        let mut mu_vect: Vec<HashMap<&'a ProductDFATeamState, String>> = Vec::new();
        let mut alg1_output: Alg1Output<'a> = Alg1Output{
            v: vec![],
            mu: vec![],
            hullset: vec![]
        };

        println!("num tasks: {}, num agents {}", self.num_tasks, self.num_agents);

        /*
        let mut extreme_points: Vec<Vec<f64>> = vec![vec![0.0; self.num_agents + self.num_tasks]; self.num_agents + self.num_tasks];

        for k in 0..(self.num_agents + self.num_tasks) {
            /*
            if k < self.num_agents {
                extreme_points[k][k] = 0.7;
                for l in 0..k{
                    extreme_points[k][l] = 0.3 / (self.num_agents - 1) as f64
                }
                if k + 1 < self.num_agents  {
                    for l in k + 1..self.num_agents {
                        extreme_points[k][l] = 0.3 / (self.num_agents - 1) as f64
                    }
                }
            } else {
                for l in 0..self.num_agents{
                    extreme_points[k][l] = 0.3 / self.num_agents as f64;
                }
                extreme_points[k][k] = 0.7;
            }*/
            extreme_points[k][k] = 1.0;
        }*/

        let w_extr: Vec<f64> = vec![1.0 / (self.num_agents + self.num_tasks) as f64; self.num_agents + self.num_tasks];
        println!("w: {:?}", w_extr);
        let safe_ret = self.exp_tot_cost(&w_extr[..], &eps, team_index_mapping, rewards);
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
        //return None;
        let dim = self.num_tasks + self.num_agents;
        let t_arr1 = arr1(target);
        let mut w_new = lp5(&hullset, target, &dim);
        while w_new != None {
            println!("w' :{:?}", w_new);
            let safe_ret = self.exp_tot_cost(w_new.as_ref().unwrap(), &eps, &team_index_mapping, rewards);
            match safe_ret {
                Some((mu_new, r)) => {
                    println!("new r: {:?}", r);
                    let weight_arr1 = arr1(&w_new.as_ref().unwrap());
                    let r_arr1 = arr1(&r);
                    let wr_dot = weight_arr1.dot(&r_arr1);
                    let wt_dot = weight_arr1.dot(&t_arr1);
                    println!("<w,r>: {}, <w,t>: {}", wr_dot, wt_dot);
                    if wr_dot < wt_dot {
                        println!("Multi-objective satisfaction not possible");
                        return alg1_output;
                    }
                    hullset.push(r);
                    mu_vect.push(mu_new);
                },
                None => panic!("No value was returned from the maximisation")
            }
            w_new = lp5(&hullset, target, &dim);
        }
        let v = witness(&hullset, &target, &dim, &self.num_agents);
        println!("v: {:?}", v);
        alg1_output.v = v;
        alg1_output.mu = mu_vect;
        alg1_output.hullset =  hullset;
        alg1_output
    }

    pub fn statistics(&self) {
        let state_space = self.states.len();
        let transitions = self.transitions.len();
        println!("state space: {}, transitions: {}", state_space, transitions);
    }

    pub fn default(initial: &'a StatePair) -> DFAProductTeamMDP<'a> {
        DFAProductTeamMDP {
            initial: ProductDFATeamState {
                s: &initial.s,
                q: &initial.q,
                agent: 0
            },
            states: vec![],
            transitions: vec![],
            num_tasks: 0,
            num_agents: 0
        }
    }
}

pub fn create_states<'a>(team_input: &'a [TeamInputs]) -> (Vec<ProductDFATeamState<'a>>,ProductDFATeamState<'a>){
    let mut states: Vec<ProductDFATeamState<'a>> = Vec::new();
    for local in team_input.iter() {
        for state in local.states.iter() {
            states.push(ProductDFATeamState {
                s: &state.s,
                q: &state.q,
                agent: local.agent
            });
        }
    }
    (
        states,
        ProductDFATeamState {
            s: &team_input[0].initial.s,
            q: &team_input[0].initial.q,
            agent: 0 }
    )
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub struct ProductDFATeamState<'a> {
    pub s: &'a u32,
    pub q: &'a Vec<u32>,
    pub agent: usize
}

pub fn create_transitions<'a>(states: &'a [ProductDFATeamState<'a>], team_input: &'a [TeamInputs], rewards_type: &'a Rewards, num_tasks: usize, num_agents: usize) -> Vec<ProductDFATeamTransition<'a>>{
    // Copy over all transitions from the local products converting them to the team mdp struct
    // Then for states which are initial for task j add a switch transition connecting task j to
    // the next task
    let mut transitions: Vec<ProductDFATeamTransition<'a>> = Vec::new();
    let rewards_coeff = match rewards_type {
        Rewards::NEGATIVE => -1.0,
        Rewards::POSITIVE => 1.0
    };
    let mut switch_transitions: Vec<ProductDFATeamTransition> = Vec::new();
    let re = regex::Regex::new(r"\d+").unwrap();
    let test_state = ProductDFATeamState {
        s: &1,
        q: &vec![0,1],
        agent: 0
    };

    for state in states.iter() {
        let base_local_product = &team_input[state.agent];
        for transition in base_local_product.transitions.iter().
            filter(|x| x.s.s == *state.s && x.s.q == *state.q) {
            let mut team_s_prime: Vec<ProductDFATeamTransitionPair<'a>> = Vec::new();
            for sprime in transition.s_sprime.iter() {
                team_s_prime.push(ProductDFATeamTransitionPair {
                    s: ProductDFATeamState {
                        s: &sprime.s.s,
                        q: &sprime.s.q,
                        agent: state.agent
                    },
                    p: sprime.p
                });
            }
            let mut rewards: Vec<f64> = vec![0.0; num_tasks + num_agents];
            match base_local_product.labelling.iter().
                find(|x| x.s.s == *state.s && x.s.q == *state.q) {
                None => {
                    if transition.reward != 0f64 && transition.a != "tau" {
                        rewards[state.agent] = rewards_coeff * transition.reward;
                    }
                }
                Some(x) => {
                    //println!("state: {:?}, label: {:?}", x.s, x.w);
                    if x.w.iter().any(|x| x.contains("jsucc")) {
                        for w in x.w.iter().filter(|x| x.contains("jsucc")) {
                            let task = re.captures(w).unwrap();
                            let task_number = task.get(0).unwrap().as_str().parse::<u32>().unwrap();
                            //println!("w: {:?}, task number: {:?}", x.w, task_number);
                            rewards[task_number as usize + num_agents] = 1000.0;
                        }
                    } else if x.w.iter().all(|x| !x.contains("jsucc") && x != "done") {
                        if transition.reward != 0f64 && transition.a != "tau" {
                            rewards[state.agent] = rewards_coeff * transition.reward;
                        }
                    }
                }
            }
            let new_transition = ProductDFATeamTransition {
                from: ProductDFATeamState {
                    s: &state.s,
                    q: &state.q,
                    agent: state.agent
                },
                a: transition.a.to_string(),
                to: team_s_prime,
                reward: rewards
            };
            transitions.push(new_transition);
            // end transition
        }
        // switch transitions
        if state.agent < num_agents - 1 {
            match team_input[state.agent].labelling.iter().
                find(|x| x.s.s == *state.s && x.s.q == *state.q) {
                None => {}
                Some(x) => {
                    if x.w.iter().any(|y| y.contains("ini")) {
                        // add a switch transition from the current state to the next agent's equivalent state
                        switch_transitions.push( ProductDFATeamTransition {
                            from: ProductDFATeamState {
                                s: &state.s,
                                q: &state.q,
                                agent: state.agent
                            },
                            a: "swi".to_string(),
                            to: vec![ProductDFATeamTransitionPair {
                                s: ProductDFATeamState {
                                    s: &team_input[state.agent + 1].initial.s,
                                    q: &state.q,
                                    agent: state.agent + 1
                                },
                                p: 1.0
                            }],
                            reward: vec![0.0; num_agents + num_tasks]
                        })
                    }
                }
            }
        }
    }
    transitions.append(&mut switch_transitions);
    transitions
}

pub fn dfs_sched_debugger<'a>(mu: &'a HashMap<&'a ProductDFATeamState, String>, states: &'a [ProductDFATeamState], transitions: &'a [ProductDFATeamTransition], initial: &'a ProductDFATeamState) -> Vec<(&'a ProductDFATeamState<'a>, &'a String)> {
    let mut output: Vec<(&'a ProductDFATeamState, &'a String)> = Vec::new();
    let mut stack: Vec<(&'a ProductDFATeamState, &'a String)> = Vec::new();
    let mut visited: Vec<bool> = vec![false; states.len()];
    let initial_state = states.iter().position(|x| x.s == initial.s && x.q == initial.q && x.agent == initial.agent).unwrap();
    visited[initial_state] = true;
    stack.push((&states[initial_state], mu.get(&states[initial_state]).unwrap()));
    output.push((&states[initial_state], mu.get(&states[initial_state]).unwrap()));
    while !stack.is_empty() {
        let observed_state = stack.pop().unwrap();
        for t in transitions.iter().
            filter(|x| x.from == *observed_state.0 && x.a == *observed_state.1) {
            for sprime in t.to.iter() {
                let sprime_position = states.iter().position(|x| *x == sprime.s).unwrap();
                if !visited[sprime_position] {
                    visited[sprime_position] = true;
                    stack.push((&sprime.s, mu.get(&sprime.s).unwrap()));
                    output.push((&sprime.s, mu.get(&sprime.s).unwrap()));
                }
            }
        }
    }
    output
}

#[derive(Debug)]
pub struct ProductDFATeamTransition<'a> {
    pub from: ProductDFATeamState<'a>,
    pub a: String,
    pub to: Vec<ProductDFATeamTransitionPair<'a>>,
    pub reward: Vec<f64>
}

#[derive(Debug)]
pub struct ProductDFATeamTransitionPair<'a> {
    pub s: ProductDFATeamState<'a>,
    pub p: f64
}

pub struct TeamTransitionLabellingPair<'a> {
    pub s: ProductDFATeamState<'a>,
    pub labelling: Vec<String>
}

pub struct DFAProductTeamStateIndexHelper<'a> {
    pub state_index_map: Vec<(&'a ProductDFATeamState<'a>, TeamStateIndex)>,
    pub state_hashmap: HashMap<&'a ProductDFATeamState<'a>, TeamStateIndex>
}

pub struct Alg1Output<'a> {
    pub v: Vec<f64>,
    pub mu: Vec<HashMap<&'a ProductDFATeamState<'a>, String>>,
    pub hullset: Vec<Vec<f64>>
}


#[derive(Debug, Clone)]
pub struct TeamInputs {
    pub states: Vec<StatePair>,
    pub initial: StatePair,
    pub transitions: Vec<ProdMDPTransition>,
    pub labelling: Vec<ProdLabellingPair>,
    pub agent: usize
}



