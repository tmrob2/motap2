use super::helper_methods;
use super::dfa;
use super::mdp;
use super::gurobi_lp;
use petgraph::{Graph, graph::NodeIndex};
use std::collections::{HashSet, VecDeque, HashMap};
use ndarray::{arr1, NdIndex};
use rand::seq::SliceRandom;

use helper_methods::*;
use dfa::*;
use mdp::*;
use gurobi_lp::*;

pub struct TeamMDP {
    pub initial: TeamState,
    pub states: Vec<TeamState>,
    pub transitions: Vec<TeamTransition>,
    pub labelling: Vec<TeamLabelling>,
    pub num_agents: usize,
    pub num_tasks: usize,
    pub task_alloc_states: Vec<TaskAllocStates>,
}

impl TeamMDP {
    pub fn create_states(&mut self, team_input: &Vec<TeamInput>) {

        for task_agent in team_input.iter() {
            for state in task_agent.product.states.iter() {
                self.states.push(TeamState {
                    state: DFAModelCheckingPair { s: state.s, q: state.q },
                    agent: task_agent.agent,
                    task: task_agent.task
                });
            }
            if task_agent.agent < self.num_agents - 1 {
                let alloc_state = TeamState{
                    state: DFAModelCheckingPair { s: task_agent.product.initial.s, q: task_agent.product.initial.q },
                    agent: task_agent.agent,
                    task: task_agent.task
                };
                let alloc_state_index: usize = self.states.iter().position(|x| *x == alloc_state).unwrap();
                self.task_alloc_states.push(
                    TaskAllocStates {
                        index: alloc_state_index,
                        state: alloc_state,
                        value: vec![0.0; self.num_agents]
                    }
                );
            }
        }

        let dfa_init = team_input[0].product.initial;
        self.initial = TeamState {
            state: DFAModelCheckingPair { s: dfa_init.s, q: dfa_init.q },
            agent: 0,
            task: 0
        };
    }

    pub fn create_transitions_and_labelling(&mut self, team_input: &Vec<TeamInput>, rewards_type: &Rewards) {
        let rewards_coeff: f64 = match rewards_type {
            Rewards::POSITIVE => 1.0,
            Rewards::NEGATIVE => -1.0
        };
        for state in self.states.iter() {
            for input in team_input.iter().
                filter(|x| x.task == state.task && x.agent == state.agent) {
                // Transitions
                for transition in input.product.transitions.iter().
                    filter(|x| x.sq == state.state) {
                    let mut state_prime: Vec<TeamTransitionPair> = Vec::new();
                    for s_prime in transition.sq_prime.iter() {
                        state_prime.push(TeamTransitionPair {
                            state: TeamState {
                                state: DFAModelCheckingPair { s: s_prime.state.s, q: s_prime.state.q },
                                agent: state.agent,
                                task: state.task
                            },
                            p: s_prime.p
                        });
                    }
                    // this transition is inherited from the input product model, and it will be zero
                    // everywhere else
                    let mut rewards: Vec<f64> = vec![0.0; self.num_tasks + self.num_agents];
                    let state_label = input.product.labelling.iter().
                        find(|x| x.sq == state.state).unwrap().w.to_vec();
                    //println!("state label: {:?}", state_label);
                    if state_label.iter().all(|x| *x != "suc" && *x != "fai") {
                        //println!("state: {:?}, agent: {}, task: {} contains no completion labels", state.state, state.agent, state.task);
                        if transition.reward != 0f64 && transition.a != "tau" {
                            rewards[input.agent] = rewards_coeff * transition.reward;
                        }
                    }
                    //println!("debug rewards: {:?}", rewards);
                    self.transitions.push(TeamTransition {
                        from: TeamState {
                            state: DFAModelCheckingPair { s: state.state.s, q: state.state.q },
                            agent: state.agent,
                            task: state.task
                        },
                        a: transition.a.to_string(),
                        to: state_prime,
                        // the question is how do we assign a value to the rewards vector
                        reward: rewards
                    });
                }
                // Labelling
                for label_pair in input.product.labelling.iter().
                    filter(|x| x.sq == state.state) {
                    //println!("working on agent: {0} of {2}; task: {1} of {3}", state.agent, state.task, self.num_agents - 1, self.num_tasks - 1);
                    // switching across agents, same task
                    if label_pair.w.iter().any(|x| *x == "ini" || *x == "suc" || *x == "fai") &&
                        state.agent < self.num_agents - 1 {
                        let next_agent_index = team_input.iter().
                            position(|x| x.agent == state.agent + 1).unwrap();
                        let new_switch_transition = TeamTransition {
                            from: TeamState {
                                state: DFAModelCheckingPair { s: state.state.s, q: state.state.q },
                                agent: state.agent,
                                task: state.task
                            },
                            a: "swi".to_string(),
                            to: vec![TeamTransitionPair{
                                state: TeamState {
                                    state: DFAModelCheckingPair {
                                        s: team_input[next_agent_index].product.initial.s,
                                        q: state.state.q
                                    },
                                    agent: input.agent + 1,
                                    task: input.task
                                },
                                p: 1.0
                            }],
                            reward: vec![0.0; self.num_agents + self.num_tasks]
                        };
                        //println!("creating switch transition for agent: {}, task: {}, of: {:?}", state.agent, state.task, new_switch_transition);
                        self.transitions.push(new_switch_transition);

                    } else if label_pair.w.iter().any(|x| *x == "suc" || *x == "fai") &&
                        state.agent == self.num_agents - 1 && state.task < self.num_tasks - 1 {
                        // the switch transition increases the number of tasks and passes it to the
                        // first agent
                        let first_agent_index = team_input.iter().
                            position(|x| x.agent == 0 && x.task == state.task + 1).unwrap();
                        self.transitions.push(TeamTransition {
                            from: TeamState {
                                state: DFAModelCheckingPair { s: state.state.s, q: state.state.q },
                                agent: state.agent,
                                task: state.task
                            },
                            a: "swi".to_string(),
                            to: vec![TeamTransitionPair {
                                state: TeamState {
                                    state: DFAModelCheckingPair {
                                        s: team_input[first_agent_index].product.initial.s,
                                        q: team_input[first_agent_index].product.initial.s
                                    },
                                    agent: 0,
                                    task: state.task + 1
                                },
                                p: 1.0
                            }],
                            reward: vec![0.0; self.num_agents + self.num_tasks]
                        });
                    } else if label_pair.w.iter().any(|x| *x == "suc" || *x == "fai") &&
                        state.agent == self.num_agents - 1 && state.task == self.num_tasks - 1 {
                        self.labelling.push(TeamLabelling {
                            state: TeamState {
                                state: DFAModelCheckingPair { s: state.state.s, q: state.state.q },
                                agent: state.agent,
                                task: state.task
                            },
                            label: vec!["done".to_string()]
                        });
                    } else if label_pair.w.iter().any(|x| *x == "com") {
                        self.labelling.push(TeamLabelling {
                            state: TeamState {
                                state: DFAModelCheckingPair { s: state.state.s, q: state.state.q },
                                agent: state.agent,
                                task: state.task
                            },
                            label: vec![format!("com_{}", state.task)]
                        });
                    }
                }
            }
        }
    }

    /// Adjusting the task rewards for finishing a task
    pub fn assign_task_rewards(&mut self) {
        for transition in self.transitions.iter_mut() {
            let task_index = self.num_agents + transition.from.task;
            //println!("task index: {}", task_index);
            let transition_copy: TeamTransition = TeamTransition {
                from: TeamState {
                    state: DFAModelCheckingPair { s: transition.from.state.s, q: transition.from.state.q },
                    agent: transition.from.agent,
                    task: transition.from.task
                },
                a: transition.a.to_string(),
                to: transition.to.to_vec(),
                reward: transition.reward.to_vec()
            };
            for label in self.labelling.iter().
                filter(|x| x.state.state == transition_copy.from.state) {
                if label.label.iter().any(|x| *x == format!("com_{}", transition_copy.from.task)) {
                    transition.reward[task_index] = 1000.0;
                }
            }
        }
    }

    pub fn generate_graph(&self) -> Graph<String, String> {
        let mut graph: Graph<String, String> = Graph::new();
        //
        for state in self.states.iter() {
            graph.add_node(format!("({},{},{},{})", state.state.s, state.state.q, state.agent, state.task));
        }
        for transition in self.transitions.iter() {
            let origin_index = graph.node_indices().
                find(|x| graph[*x] ==
                    format!(
                        "({},{},{},{})",
                        transition.from.state.s,
                        transition.from.state.q,
                        transition.from.agent,
                        transition.from.task
                    )).unwrap();
            for trans_to in transition.to.iter() {
                let destination_index = match graph.node_indices().
                    find(|x| graph[*x] == format!(
                        "({},{},{},{})",
                        trans_to.state.state.s,
                        trans_to.state.state.q,
                        trans_to.state.agent,
                        trans_to.state.task
                    )) {
                    None => {
                        println!("Transition {:?}", transition);
                        panic!("state: ({},{},{},{}) not found",
                               trans_to.state.state.s,
                               trans_to.state.state.q,
                               trans_to.state.agent,
                               trans_to.state.task
                        );}
                    Some(x) => {x}
                };
                graph.add_edge(origin_index, destination_index, transition.a.to_string());
            }
        }
        graph
    }

    pub fn team_ij_index_mapping(&self) -> HashMap<(usize,usize), TeamStateIndexHelper>{
        let mut state_index: HashMap<(usize, usize), TeamStateIndexHelper> = HashMap::new();
        for j in 0..self.num_tasks {
            for i in 0..self.num_agents {
                let state_indices: Vec<(&TeamState, TeamStateIndex)> = self.states.iter().enumerate().
                    filter(|(k, x)| x.task == j && x.agent == i).enumerate().
                    map(|(z,(k, x))| (x,TeamStateIndex{
                        local_index: z,
                        team_index: k
                    })).collect();
                let mut state_mapping_hash: HashMap<&TeamState, TeamStateIndex> = HashMap::new();
                for (state, k) in state_indices.iter() {
                    state_mapping_hash.insert(*state, TeamStateIndex{local_index: k.local_index, team_index: k.team_index});
                }
                state_index.insert(
                    (j,i),
                    TeamStateIndexHelper {
                        state_index_map: state_indices,
                        state_hashmap: state_mapping_hash
                    }
                );
            }
        }
        state_index
    }

    pub fn ij_mapping(&self) -> HashMap<(usize, usize), usize> {
        let mut mapping: HashMap<(usize, usize), usize> = HashMap::new();
        let mut counter: usize = 0;
        for j in 0..self.num_tasks {
            for i in 0..self.num_agents {
                mapping.insert((j,i), counter);
                counter += 1
            }
        }
        mapping
    }

    /// Fair scheduler value iteration
    /// ``` team_index``` is an index mapping of the filtered state spaces of a team MDP for a given
    /// (task, agent) ordered pair. The hashmap key is the ordered pair, and the value is a
    /// ```TeamStateIndexHelper``` or a struct containing a vector of tuples containing the state space
    /// and the enumerated filtered index of the state.
    pub fn exp_tot_cost<'a>(&self, w: &[f64], eps: &f64, team_index: &'a HashMap<(usize, usize), TeamStateIndexHelper>, rewards: &Rewards) -> Option<(HashMap<&'a TeamState, String>, Vec<f64>)> {
        let mut mu: HashMap<&'a TeamState, String> = HashMap::new();
        let mut r: Vec<f64> = vec![0.0; w.len()];
        let weight = arr1(w);
        let ij_k_mapping = self.ij_mapping();

        let test_state = TeamState {
            state: DFAModelCheckingPair { s: 0, q: 0 },
            agent: 0,
            task: 0
        };

        let mut x_cost_vectors: Vec<Vec<f64>> = vec![Vec::new(); ij_k_mapping.len()];
        let mut y_cost_vectors: Vec<Vec<f64>> = vec![Vec::new(); ij_k_mapping.len()];
        for j in 0..self.num_tasks {
            for i in 0..self.num_agents {
                let mapping_index: &usize = ij_k_mapping.get(&(j, i)).unwrap();
                let u = team_index.get(&(j, i)).unwrap();
                let vecsize = u.state_hashmap.len();
                x_cost_vectors[*mapping_index] = vec![0.0; vecsize];
                y_cost_vectors[*mapping_index] = vec![0.0; vecsize];
            }
        }

        let mut x_agent_cost_vector: Vec<Vec<f64>> = vec![vec![0.0; self.states.len()]; self.num_agents];
        let mut y_agent_cost_vector: Vec<Vec<f64>> = vec![vec![0.0; self.states.len()]; self.num_agents];

        let mut x_task_cost_vector: Vec<Vec<f64>> = vec![vec![0.0; self.states.len()]; self.num_tasks];
        let mut y_task_cost_vector: Vec<Vec<f64>> = vec![vec![0.0; self.states.len()]; self.num_tasks];

        for j in (0..self.num_tasks).rev() {
            for i in (0..self.num_agents).rev() {
                let ij_mapping_index: &usize = ij_k_mapping.get(&(j, i)).unwrap();
                let mut ij_plus_mapping_index: Option<&usize> = None;
                let helper_ij = team_index.get(&(j,i)).unwrap();
                let default_helper = TeamStateIndexHelper { state_index_map: vec![], state_hashmap: Default::default() };
                let helper_iplus1j: &TeamStateIndexHelper = if i < self.num_agents - 1 {
                    ij_plus_mapping_index = ij_k_mapping.get(&(j, i+1));
                    team_index.get(&(j,i+1)).unwrap()
                } else if j < self.num_tasks - 1 {
                    ij_plus_mapping_index = ij_k_mapping.get(&(j+1, 0));
                    team_index.get(&(j+1, 0)).unwrap()
                } else { &default_helper };
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
                                //println!("action: {}", transition.a);
                                if transition.a == "swi" {
                                    let x_sprime_index: &TeamStateIndex = helper_iplus1j.state_hashmap.get(&sprime.state).unwrap();
                                    sum_vect[z] = sprime.p * x_cost_vectors[*ij_plus_mapping_index.unwrap()][x_sprime_index.local_index];
                                } else {
                                    let x_sprime_index: &TeamStateIndex = helper_ij.state_hashmap.get(&sprime.state).unwrap();
                                    sum_vect[z] = sprime.p * x_cost_vectors[*ij_mapping_index][x_sprime_index.local_index];
                                }
                            }
                            let sum_vect_sum: f64 = sum_vect.iter().fold(0f64, |sum, &val| sum + val);
                            let mut action_reward = scaled_weight_rewards + sum_vect_sum;
                            min_action_values.push((transition.a.to_string(), action_reward));
                        }
                        //min_action_values.sort_by(|(a1, a2), (b1, b2)| a1.partial_cmp(b1).unwrap());
                        let mut v: Vec<_> = min_action_values.iter().
                            map(|(z, x)| (z, NonNan::new(*x).unwrap())).collect();

                        v.sort_by_key(|key| key.1);
                        let mut minmax_pair: &(&String, NonNan) = match rewards {
                            Rewards::NEGATIVE => &v.last().unwrap(),
                            Rewards::POSITIVE => &v[0]
                        };
                        let mut minmax_val: f64 = minmax_pair.1.inner();
                        let mut arg_minmax = minmax_pair.0;

                        /*if self.task_alloc_states.iter().any(|x| x.state == **state) {
                            let check_mult_argmin: Vec<_> = v.iter().
                                filter(|(k,x)| x.inner() == v.last().unwrap().1.inner()).
                                map(|(k, x)| (*k,*x)).collect();
                            if check_mult_argmin.len() > 1 && check_mult_argmin.iter().any(|(a, x)| **a == "swi") {
                                println!("min pair: {:?}, mult argmin: {:?}", minmax_pair, check_mult_argmin);
                                let rand_min_pair: &(&String, NonNan) = check_mult_argmin.choose(& mut rand::thread_rng()).unwrap();
                                //let rand_min_pair = check_mult_argmin.iter().find(|(a, x)| **a == "swi").unwrap();
                                println!("state: ({},{},{},{}) -> a: {:?}", state.state.s, state.state.q, state.agent, state.task, check_mult_argmin);
                                minmax_val = rand_min_pair.1.inner();
                                arg_minmax = rand_min_pair.0;
                            }
                        }*/

                        y_cost_vectors[*ij_mapping_index][k.local_index] = minmax_val;
                        mu.insert(*state, arg_minmax.to_string());
                    }
                    //println!("xbar:{:?}", x_cost_vectors[*ij_mapping_index]);

                    let y_bar_diff = absolute_diff_vect(&x_cost_vectors[*ij_mapping_index], &y_cost_vectors[*ij_mapping_index]);
                    let mut y_bar_diff_max_vect: Vec<NonNan> = y_bar_diff.iter().
                        map(|x| NonNan::new(*x).unwrap()).collect();
                    y_bar_diff_max_vect.sort();
                    epsilon = y_bar_diff_max_vect.last().unwrap().inner();
                    //println!("eps: {}", epsilon);
                    x_cost_vectors[*ij_mapping_index] = y_cost_vectors[*ij_mapping_index].to_vec();
                }
                epsilon = 1.0;
                while epsilon > *eps {
                    for (state, k) in helper_ij.state_index_map.iter() {
                        let chosen_action: &String = mu.get(*state).unwrap();
                        for transition in self.transitions.iter().
                            filter(|x| x.from == **state && x.a == *chosen_action) {
                            let mut sum_vect_agent: Vec<Vec<f64>> = vec![vec![0.0; transition.to.len()]; self.num_agents];
                            let mut sum_vect_task: Vec<Vec<f64>> = vec![vec![0.0; transition.to.len()]; self.num_tasks];
                            for (l, sprime) in transition.to.iter().enumerate(){
                                let x_sprime_index: &TeamStateIndex = if transition.a == "swi" {
                                    helper_iplus1j.state_hashmap.get(&sprime.state).unwrap()
                                } else {
                                    helper_ij.state_hashmap.get(&sprime.state).unwrap()
                                };
                                for agent in 0..self.num_agents {
                                    sum_vect_agent[agent][l] = sprime.p * x_agent_cost_vector[agent][x_sprime_index.team_index];
                                }
                                for task in 0..self.num_tasks {
                                    sum_vect_task[task][l] = sprime.p * x_task_cost_vector[task][x_sprime_index.team_index];
                                }
                            }
                            for agent in 0..self.num_agents {
                                let p_trans_agent: f64 = sum_vect_agent[agent].iter().sum();
                                y_agent_cost_vector[agent][k.team_index] = transition.reward[agent] + p_trans_agent;
                            }
                            for task in 0..self.num_tasks {
                                let p_trans_task: f64 = sum_vect_task[task].iter().sum();
                                y_task_cost_vector[task][k.team_index] = transition.reward[self.num_agents + task] + p_trans_task;
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
                        //println!("ytaskbar obj:{} = {:?}", task, ytaskbar[task]);
                    }
                    epsilon = eps_inner;
                }
            }
        }
        let helper_init = team_index.get(&(self.initial.task, self.initial.agent)).unwrap();
        let init_team_index = helper_init.state_hashmap.get(&self.initial).unwrap();
        for agent in 0..self.num_agents {
            r[agent] = y_agent_cost_vector[agent][init_team_index.team_index];
        }

        for task in 0..self.num_tasks {
            r[task + self.num_agents] = y_task_cost_vector[task][init_team_index.team_index];
        }

        //println!("new r: {:?}", r);

        Some((mu, r))
    }

    pub fn team_done_states(&self) -> Vec<TeamState> {
        let mut done_states: Vec<TeamState> = Vec::new();
        for state in self.states.iter().
            filter(|x| self.labelling.iter().any(|x| x.state == x.state &&
                x.label.iter().any(|y| *y == "done"))) {
            done_states.push(state.clone())
        }
        done_states
    }

    pub fn modify_final_rewards(&mut self, team_inputs: &Vec<TeamInput>) {
        let num_tasks: usize = self.num_tasks.clone();
        let num_agents: usize = self.num_agents.clone();
        for input in team_inputs.iter().
            filter(|x| x.task == num_tasks - 1 ){ // && x.agent == num_agents - 1) {
            for transition in self.transitions.iter_mut().
                filter(|x| x.from.agent == input.agent && x.from.task == input.task &&
                    (
                        input.dead.iter().any(|y| *y == x.from.state.q) ||
                            input.acc.iter().any(|y| *y == x.from.state.q)
                    ) && x.a != "tau")
            {
                //println!("modifying rewards for state: ({},{},{},{})",
                //         transition.from.state.s, transition.from.state.q, input.agent, input.task);
                transition.reward = vec![0.0; self.num_agents + self.num_tasks];
            }
        }
    }

    pub fn default() -> TeamMDP {
        TeamMDP {
            initial: TeamState {
                state: DFAModelCheckingPair { s: 0, q: 0 },
                agent: 0,
                task: 0
            },
            states: vec![],
            transitions: vec![],
            labelling: vec![],
            num_agents: 0,
            num_tasks: 0,
            task_alloc_states: vec![]
        }
    }

    pub fn multi_obj_sched_synth<'a>(&self, target: &Vec<f64>, eps: &f64, rewards: &Rewards, team_index_mapping: &'a HashMap<(usize, usize), TeamStateIndexHelper>) -> Alg1Output<'a> {
        let mut hullset: Vec<Vec<f64>> = Vec::new();
        let mut mu_vect: Vec<HashMap<&'a TeamState, String>> = Vec::new();
        let mut alg1_output: Alg1Output<'a> = Alg1Output{
            v: vec![],
            mu: vec![],
            hullset: vec![]
        };

        println!("num tasks: {}, num agents {}", self.num_tasks, self.num_agents);

        let mut extreme_points: Vec<Vec<f64>> = vec![vec![0.0; self.num_agents + self.num_tasks]; self.num_agents + self.num_tasks];

        /*for k in 0..(self.num_agents + self.num_tasks) {
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

        for k in 0..(self.num_tasks + self.num_agents) {
            extreme_points[k][k] = 1.0;
            let w_extr: &Vec<f64> = &extreme_points[k];
            println!("w: {:?}", w_extr);
            let safe_ret = self.exp_tot_cost(&w_extr, &eps, &team_index_mapping, rewards);
            match safe_ret {
                Some((mu_new, r)) => {
                    hullset.push(r);
                    mu_vect.push(mu_new);
                },
                None => panic!("No value was returned from the maximisation")
            }
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
        let dim = self.num_tasks + self.num_agents;
        let t_arr1 = arr1(target);
        let mut counter: u32 = 1;
        let mut w_new = lp5(&hullset, target, &dim);
        while w_new != None {
            println!("w' :{:?}", w_new);
            let safe_ret = self.exp_tot_cost(&w_new.as_ref().unwrap(), &eps, &team_index_mapping, rewards);
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
        println!("Constructing witness");
        let v = witness(&hullset, &target, &dim, &self.num_agents);
        println!("v: {:?}", v);
        alg1_output.v = v;
        alg1_output.mu = mu_vect;
        alg1_output.hullset =  hullset;
        alg1_output
    }

    pub fn dfs_merging(&self, sched: &Vec<Vec<Mu>>, v: &Vec<f64>) -> Graph<String, String> {
        let mut graph: Graph<String, String> = Graph::new();
        let mut s = Vec::new();
        let c0: Vec<u32> = (1..(v.len() as u32)).collect();
        s.push((&self.initial, c0));
        while !s.is_empty() {
            let (s_val, c_val) = s.pop().unwrap();
            let vertex = format!(
                "(({},{},{},{}),({:?})",
                s_val.state.s, s_val.state.q, s_val.agent, s_val.task,
                c_val
            );
            let from_node_index;
            match graph.node_indices().find(|x| graph[*x] == vertex) {
                None => {
                    // Add the vertex to the graph
                    //println!("vertex: {:?}", vertex);
                    from_node_index = graph.add_node(vertex);

                }
                Some(x) => {
                    // the note index already exists and we move on
                    from_node_index = x;
                }
            }

            for transition in self.transitions.iter().
                filter(|x| x.from == *s_val) {
                let mut c_prime: Vec<u32> = Vec::new();
                for (j, mu) in sched.iter().enumerate().
                    filter(|(j,x)| c_val.iter().any(|y| *y == *j as u32) ) {
                    let mu_s = mu.iter().find(|x| x.team_state == *s_val).unwrap();
                    match &mu_s.action {
                        None => { panic!("action not found in state: ({},{},{},{})", s_val.state.s, s_val.state.q, s_val.agent, s_val.task)}
                        Some(x) => {
                            if transition.a == *x {
                                c_prime.push(j as u32);
                            }
                        }
                    }
                }
                if !c_prime.is_empty() {
                    for sprime in transition.to.iter() {
                        //println!("s: {:?}", s);
                        let v_prime_vect: Vec<f64> = v.iter().enumerate().
                            filter(|(x, y)| c_prime.iter().any(|z| *x == *z as usize)).map(|(x,y)| *y).collect();
                        let v_prime_sum: f64 = v_prime_vect.iter().sum();
                        //println!("v_prime_vect: {:?}, v_prime: {}", v_prime_vect, v_prime_sum);
                        let v_vect: Vec<f64> = v.iter().enumerate().
                            filter(|(x,y)| c_val.iter().any(|z| *x == *z as usize)).map(|(x,y)| *y).collect();
                        let v_sum: f64 = v_vect.iter().sum();
                        //println!("v_vect: {:?}, v_prime: {}", v_vect, v_sum);
                        let p = v_prime_sum / v_sum;
                        let to_vertex = format!(
                            "(({},{},{},{}),({:?})",
                            sprime.state.state.s, sprime.state.state.q, sprime.state.agent, sprime.state.task,
                            c_prime
                        );
                        let destination_node_index;
                        match graph.node_indices().find(|x| graph[*x] == to_vertex) {
                            None => {
                                // The vertex wasn't found this means that we can add the vertex without complaint
                                destination_node_index = graph.add_node(to_vertex);
                                s.push((&sprime.state, c_prime.to_vec()));
                            }
                            Some(x) => {
                                // this doesn't necessarily mean that the algorithm doesn't work,
                                // let's do some investigation to work out if this is a problem or not
                                destination_node_index = x;
                            }
                        }
                        graph.add_edge(from_node_index, destination_node_index, format!("({},{1:.2$})", transition.a, p.to_string(), 5));
                    }
                }
            }

        }
        graph
    }
}

#[derive(Debug, PartialEq, Clone, Copy, Eq, Hash)]
pub struct TeamState {
    pub state: DFAModelCheckingPair,
    pub agent: usize,
    pub task: usize
}

impl TeamState {
    fn default() -> TeamState {
        TeamState {
            state: DFAModelCheckingPair { s: 0, q: 0 },
            agent: 0,
            task: 0
        }
    }
}

#[derive(Debug)]
pub struct TeamTransition {
    pub from: TeamState,
    pub a: String,
    pub to: Vec<TeamTransitionPair>,
    pub reward: Vec<f64>
}

#[derive(Debug, Clone)]
pub struct TeamTransitionPair {
    pub state: TeamState,
    pub p: f64
}

#[derive(Debug)]
pub struct TeamLabelling {
    pub state: TeamState,
    pub label: Vec<String>
}

#[derive(Debug, Clone)]
pub struct Mu {
    pub team_state: TeamState,
    pub action: Option<String>,
    pub task_local_index: usize,
    pub agent_local_index: usize
}

impl Mu {
    fn default() -> Mu {
        Mu {
            team_state: TeamState {
                state: DFAModelCheckingPair { s: 0, q: 0 },
                agent: 0,
                task: 0
            },
            action: None,
            task_local_index: 0,
            agent_local_index: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TeamInitState {
    pub state: TeamState,
    pub index: usize,
    pub obj_index: usize
}

impl TeamInitState {
    fn default() -> TeamInitState {
        TeamInitState {
            state: TeamState {
                state: DFAModelCheckingPair { s: 0, q: 0 },
                agent: 0,
                task: 0
            },
            index: 0,
            obj_index: 0
        }
    }
}

pub enum Rewards {
    NEGATIVE,
    POSITIVE
}

pub enum Fairness {
    FAIR,
    UNFAIR
}

pub struct TeamStateIndexHelper<'a> {
    pub state_index_map: Vec<(&'a TeamState, TeamStateIndex)>,
    pub state_hashmap: HashMap<&'a TeamState, TeamStateIndex>
}

pub struct TeamStateIndex{
    pub local_index: usize,
    pub team_index: usize
}

pub struct CostIndexHelper{
    pub x: Vec<f64>,
    pub y: Vec<f64>
}

#[derive(Debug)]
pub struct TaskAllocStates {
    pub index: usize,
    pub state: TeamState,
    pub value: Vec<f64>
}

pub struct Alg1Output<'a> {
    pub v: Vec<f64>,
    pub mu: Vec<HashMap<&'a TeamState, String>>,
    pub hullset: Vec<Vec<f64>>
}