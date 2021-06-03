use super::helper_methods;
use super::dfa;
use super::mdp;
use super::product_dfa;
use super::gurobi_lp;
use petgraph::{Graph, graph::NodeIndex};
use std::collections::{HashSet, VecDeque, HashMap};
use ndarray::{arr1, NdIndex};
use rand::seq::SliceRandom;
use regex::Regex;

use helper_methods::*;
use dfa::*;
use mdp::*;
use product_dfa::*;
use gurobi_lp::*;
use petgraph::graph::Node;
use crate::model_checking::decomp_team_mdp::{TeamStateIndex, TeamStateIndexHelper};
use itertools::Itertools;

pub struct DFAProductTeamMDP {
    pub initial: ProductDFATeamState,
    pub states: Vec<ProductDFATeamState>,
    pub transitions: Vec<ProductDFATeamTransition>,
    pub num_tasks: usize,
    pub num_agents: usize,
}

impl DFAProductTeamMDP {
    pub fn create_states<'a>(&mut self, team_input: &'a Vec<LocalProductInput>){
        for local in team_input.iter() {
            for state in local.local_product.states.iter() {
                self.states.push(ProductDFATeamState {
                    s: state.s,
                    q: state.q.to_vec(),
                    agent: local.agent
                });
            }
        }
        self.initial = ProductDFATeamState {
            s: team_input[0].local_product.initial.s,
            q: team_input[0].local_product.initial.q.to_vec(),
            agent: 0
        };
    }

    pub fn create_transitions<'a>(&mut self, team_input: &'a Vec<LocalProductInput>) {
        // Copy over all transitions from the local products converting them to the team mdp struct
        // Then for states which are initial for task j add a switch transition connecting task j to
        // the next task
        let mut switch_transitions: Vec<ProductDFATeamTransition> = Vec::new();
        let re = regex::Regex::new(r"\d+").unwrap();

        for state in self.states.iter() {
            let base_local_product = &team_input[state.agent].local_product;
            for transition in base_local_product.transitions.iter().
                filter(|x| x.s.s == state.s && x.s.q == state.q) {
                let mut team_s_prime: Vec<ProductDFATeamTransitionPair> = Vec::new();
                for sprime in transition.s_sprime.iter() {
                    team_s_prime.push(ProductDFATeamTransitionPair {
                        s: ProductDFATeamState {
                            s: sprime.s.s,
                            q: sprime.s.q.to_vec(),
                            agent: state.agent
                        },
                        p: sprime.p
                    });
                }
                let mut rewards: Vec<f64> = vec![0.0; self.num_tasks + self.num_agents];
                match base_local_product.labelling.iter().
                    find(|x| x.s.s == state.s && x.s.q == state.q) {
                    None => {
                        if transition.reward != 0f64 && transition.a != "tau" {
                            rewards[state.agent] = transition.reward;
                        }
                    }
                    Some(x) => {
                        //println!("state: {:?}, label: {:?}", x.s, x.w);
                        if x.w.iter().any(|x| x.contains("jfai")) {
                            let task = re.captures(&x.w[0]).unwrap();
                            let task_number = task.get(0).unwrap().as_str().parse::<u32>().unwrap();
                            //println!("w: {:?}, task number: {:?}", x.w, task_number);
                            rewards[task_number as usize + self.num_agents] = 1000.0;
                        } else if x.w.iter().all(|x| !x.contains("jfai") && x != "complete") {
                            if transition.reward != 0f64 && transition.a != "tau" {
                                rewards[state.agent] = transition.reward;
                            }
                        }
                    }
                }

                self.transitions.push(ProductDFATeamTransition {
                    from: ProductDFATeamState {
                        s: state.s,
                        q: state.q.to_vec(),
                        agent: state.agent
                    },
                    a: transition.a.to_string(),
                    to: team_s_prime,
                    reward: rewards
                });
                // end transition
            }
            // switch transitions
            if state.agent < self.num_agents - 1 {
                match team_input[state.agent].local_product.labelling.iter().
                    find(|x| x.s.s == state.s && x.s.q == state.q) {
                    None => {}
                    Some(x) => {
                        if x.w.iter().any(|y| y.contains("ini")) {
                            // add a switch transition from the current state to the next agent's equivalent state
                            switch_transitions.push( ProductDFATeamTransition {
                                from: ProductDFATeamState {
                                    s: state.s,
                                    q: state.q.to_vec(),
                                    agent: state.agent
                                },
                                a: "swi".to_string(),
                                to: vec![ProductDFATeamTransitionPair {
                                    s: ProductDFATeamState {
                                        s: team_input[state.agent + 1].local_product.initial.s,
                                        q: state.q.to_vec(),
                                        agent: state.agent + 1
                                    },
                                    p: 1.0
                                }],
                                reward: vec![0.0; self.num_agents + self.num_tasks]
                            })
                        }
                    }
                }
            }
        }
        self.transitions.append(&mut switch_transitions);
    }

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
                    find(|x| g[*x] == format!("({},{:?},{})", sprime.s.s, sprime.s.q, sprime.s.agent)).unwrap();
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

    pub fn min_tot_exp_tot<'a>(&self, w: &[f64], eps: &f64, team_index: &'a HashMap<usize, DFAProductTeamStateIndexHelper>) -> Option<(HashMap<&'a ProductDFATeamState, String>, Vec<f64>)>{
        let mut mu: HashMap<&'a ProductDFATeamState, String> = HashMap::new();
        let mut r: Vec<f64> = vec![0.0; w.len()];
        let weight = arr1(w);

        let test_state = ProductDFATeamState {
            s: 2,
            q: vec![3,2],
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
                    let mut min_action_values: Vec<(String, f64)> = Vec::new();
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
                                //println!("switch: p: {}, x: {}, p.w: {}", sprime.p, x_cost_vectors[sprime.s.agent][x_sprime_index.local_index], scaled_weight_rewards);
                            } else {
                                let x_sprime_index: &TeamStateIndex = helper_ij.state_hashmap.get(&sprime.s).unwrap();
                                sum_vect[z]= sprime.p * x_cost_vectors[i][x_sprime_index.local_index];
                                /*if **state == test_state {
                                    println!("a:{}, p: {}, x: {}, p.w: {}", transition.a, sprime.p, x_cost_vectors[i][x_sprime_index.local_index], scaled_weight_rewards);
                                }*/
                            }
                        }
                        let sum_vect_sum: f64 = sum_vect.iter().fold(0f64, |sum, &val| sum + val);
                        let mut action_reward = scaled_weight_rewards + sum_vect_sum;
                        min_action_values.push((transition.a.to_string(), action_reward));
                    }
                    /*if **state == test_state {
                        println!("min_action_vals: {:?}", min_action_values);
                    }*/
                    let mut v: Vec<_> = min_action_values.iter().
                        map(|(z,x)| (z, NonNan::new(*x).unwrap())).collect();
                    v.sort_by_key(|key| key.1);
                    let mut min_pair: &(&String, NonNan) = &v[0];
                    let mut min_val: f64 = min_pair.1.inner();
                    let argmin = min_pair.0;
                    y_cost_vectors[i][k.local_index] = min_val;
                    mu.insert(*state, argmin.to_string());
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
            epsilon = 1.0;
            while epsilon > *eps {
                for (state, k) in helper_ij.state_index_map.iter() {
                    let chosen_action: &String = mu.get(*state).unwrap();
                    //println!("action: {}", chosen_action);
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
                                //println!("task {} cost, state: ({},{:?},{}) -> ({},{:?},{}), p:{}, x:{}", task, state.s, state.q, state.agent, sprime.s.s, sprime.s.q, sprime.s.agent, sprime.p,  x_task_cost_vector[task][x_sprime_index.team_index])
                            }
                        }
                        for agent in 0..self.num_agents {
                            let p_trans_agent: f64 = sum_vect_agent[agent].iter().sum();
                            y_agent_cost_vector[agent][k.team_index] = transition.reward[agent] + p_trans_agent;
                            //println!("agent reward, state: ({},{:?},{}) -> reward:{}", state.s, state.q, state.agent, transition.reward[agent])
                        }
                        for task in 0..self.num_tasks {
                            let p_trans_task: f64 = sum_vect_task[task].iter().sum();
                            y_task_cost_vector[task][k.team_index] = transition.reward[self.num_agents + task] + p_trans_task;
                            //println!("task {} reward, state: ({},{:?},{}) -> reward:{:?}, reward given: {}", task, state.s, state.q, state.agent, transition.reward, transition.reward[self.num_agents + task])
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
                }
                epsilon = eps_inner;
                //println!("epsilon: {}", epsilon);
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

        println!("new r: {:?}", r);
        /*for (state, action)  in mu.iter() {
            println!("state: ({},{:?},{}), action: {}", state.s, state.q, state.agent, action);
        }*/

        return None
    }

    pub fn default() -> DFAProductTeamMDP {
        DFAProductTeamMDP {
            initial: ProductDFATeamState {
                s: 0,
                q: vec![],
                agent: 0
            },
            states: vec![],
            transitions: vec![],
            num_tasks: 0,
            num_agents: 0
        }
    }
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub struct ProductDFATeamState {
    pub s: u32,
    pub q: Vec<u32>,
    pub agent: usize
}

#[derive(Debug)]
pub struct ProductDFATeamTransition {
    pub from: ProductDFATeamState,
    pub a: String,
    pub to: Vec<ProductDFATeamTransitionPair>,
    pub reward: Vec<f64>
}

#[derive(Debug)]
pub struct ProductDFATeamTransitionPair {
    pub s: ProductDFATeamState,
    pub p: f64
}

pub struct TeamTransitionLabellingPair {
    pub s: ProductDFATeamState,
    pub labelling: Vec<String>
}

#[derive(Debug, Clone)]
pub struct LocalProductInput{
    pub local_product: ProductDFAProductMDP,
    pub agent: usize
}

impl LocalProductInput {
    pub fn default() -> LocalProductInput {
        LocalProductInput {
            local_product: ProductDFAProductMDP {
                states: vec![],
                initial: StatePair { s: 0, q: vec![] },
                transitions: vec![],
                labelling: vec![]
            },
            agent: 0
        }
    }
}

pub struct DFAProductTeamStateIndexHelper<'a> {
    pub state_index_map: Vec<(&'a ProductDFATeamState, TeamStateIndex)>,
    pub state_hashmap: HashMap<&'a ProductDFATeamState, TeamStateIndex>
}

