mod model_checking;
use model_checking::decomp_team_mdp2::*;
use std::fs::File;
use std::io::Write;
use petgraph::{dot::Dot, Direction};
use std::time::{Duration, Instant};
use regex::Regex;

use model_checking::helper_methods::{parse_language, read_dfa_json, power_set, construct_labelling_vect, construct_hash_from_vect};
use model_checking::mdp2::*;
use model_checking::dfa2::*;
use model_checking::product_dfa::*;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

fn main() {
    let alphabet: Vec<&str> = vec!["start1", "l", "f1", "start2", "f2", "h", "s1", "down", "r"];
    let ps = power_set(&alphabet[..]);
    //println!("power set alphabet: {:?}", ps);
    //println!("{:?}", mdp_prod_states);
    let mut h1: HashSet<&str> = HashSet::with_capacity(1);
    h1.insert("s1");
    let mut h2: HashSet<&str> = HashSet::with_capacity(2);
    h2.insert("s1");
    h2.insert("r");
    let t_words = vec![h1, h2];
    // -------------------------
    // Construct DFA1
    // -------------------------
    let mut transition_map: HashMap<(u32, u32), Vec<&HashSet<&str>>> = HashMap::new();
    let mut transition_map = construct_dfa_transition(&ps[..], Some(&t_words[..]), 0, 1, None, &mut transition_map, None);
    let mut transition_map = construct_dfa_transition(&ps[..], None, 1, 3, Some((Some("h"), 2)), &mut transition_map, None);
    let mut not_words = transition_map.get(&(1,3)).unwrap().clone();
    let mut transition_map = construct_dfa_transition(&ps[..], None, 1, 2, Some((Some("down"), 2)), &mut transition_map, Some(&not_words[..]));
    let mut not_words1 = transition_map.get(&(1,2)).unwrap().clone();
    let mut not_words2 = transition_map.get(&(1,3)).unwrap().clone();
    not_words2.append(&mut not_words1);
    let mut transition_map = construct_dfa_transition(&ps[..], None, 1, 1, Some((None, 2)), &mut transition_map, Some(&not_words2[..]));
    let not_words = transition_map.get(&(0,1)).unwrap().clone();
    let mut transition_map = construct_dfa_transition(&ps[..], None, 0, 0, Some((None, 2)), &mut transition_map, Some(&not_words[..]));
    let mut transition_map  = construct_dfa_transition(&ps[..], None, 2, 2, Some((None, 2)), &mut transition_map, None);
    let mut transition_map  = construct_dfa_transition(&ps[..], None, 3, 3, Some((None, 2)), &mut transition_map, None);

    let grid_dim: (usize, usize) = (11, 11);
    let grid_state_space: HashMap<usize,(usize,usize)> = create_grid(grid_dim);
    let c_loc: (usize,usize) = (0,0);
    let c_loc2: (usize,usize) = (1,0);
    let act1: Vec<&str> = vec!["x", "l", "r"];
    let act2: Vec<&str> = vec!["n", "s", "e", "w"];
    let v = vec![vec![], vec!["h"], vec!["r"], vec!["l"], vec!["h", "r"], vec!["h", "l"], vec!["start1"],
                 vec!["f1"], vec!["start2"], vec!["f2"], vec!["s1"], vec!["down"], vec!["start1", "h"],
                 vec!["f1", "h"], vec!["start2", "h"], vec!["f2", "h"], vec!["s1", "h"], vec!["down", "h"]
    ];
    let mdp_labels = construct_labelling_vect(&v[..]);
    // Different agents may have different obstacles
    let obstacles: [(usize, usize); 17] = [(2,2),(2,3),(2,4),(4,4),(5,4),(6,4),(7,2),
        (8,2),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(8,7),(7,7),(6,7)];
    // hazards
    let hazards: [(usize, usize); 10] = [(3,0),(4,0),(4,1),(4,2),(3,7),(2,8),(2,9),(4,8),(4,9),(4,10)];
    // Different agents may have different probabilities of moving in cardinal directions
    let movement_p: f64 = 1.0;
    //
    let mut all_act: Vec<&str> = act1.to_vec();
    all_act.extend(act2.iter().copied());
    let mut obj_points: HashMap<MDPLongState,Vec<&str>> = HashMap::new();
    obj_points.insert(MDPLongState{ m: "l", g: (0, 2) }, vec!["start1"]);
    obj_points.insert(MDPLongState{ m: "l", g: (3, 9) }, vec!["f1"]);
    obj_points.insert(MDPLongState{ m: "l", g: (1, 0) }, vec!["start2"]);
    obj_points.insert(MDPLongState{ m: "l", g: (10, 9) }, vec!["f2"]);
    obj_points.insert(MDPLongState{ m: "r", g: (2, 0) }, vec!["s1"]);
    obj_points.insert(MDPLongState{ m: "r", g: (5, 0) }, vec!["down"]);
    //obj_points.insert(MDPLongState{ m: "r", g: (2, 1) }, "s");
    //obj_points.insert(MDPLongState{ m: "r", g: (2, 2)}, "d");

    // -------------------------------
    // Restrict DFA sensor Transitions
    // -------------------------------
    let mut dfa1_transitions: Vec<DFA2Transitions> = vec![DFA2Transitions{ q: 0, w: vec![], q_prime: 0 }; transition_map.len()];
    for (i, ((q1,q2),v)) in transition_map.iter().enumerate() {
        let filtered_labelling = v.into_iter().filter(|x| mdp_labels.iter().any(|y| y == **x)).collect::<Vec<_>>();
        dfa1_transitions[i] = DFA2Transitions { q: *q1, w: filtered_labelling, q_prime: *q2 }
    }

    let dfa_sensor: DFA2 = DFA2 {
        states: vec![0,1,2,3],
        sigma: &alphabet,
        initial: 0,
        delta: &dfa1_transitions,
        acc: vec![2],
        dead: vec![3]
    };

    // --------------------------------------
    // Construct the Agent-Environment Model
    // --------------------------------------
    let (mdp_state_hashmap, mdp_state_coords, mdp_states) =
        create_mdp_states(&grid_state_space, &act1);
    let mut labelling: Vec<model_checking::mdp2::MDPLabellingPair> = Vec::new();
    let (transitions, labelling) =
        create_mdp_transitions(&mdp_states[..], &mdp_state_hashmap, &mdp_state_coords,
                               &grid_dim, &movement_p,&obstacles[..], &hazards[..],
                               &obj_points, &all_act[..],&mdp_labels[..], &mut labelling);

    let mdps: Vec<MDP2> = vec![MDP2{
        states: mdp_states.to_vec(),
        initial: *mdp_state_coords.get(&MDPLongState{ m: "x", g: c_loc }).unwrap(),
        transitions: transitions.to_vec(),
        labelling: labelling.to_vec()
    }, MDP2{
        states: mdp_states.to_vec(),
        initial: *mdp_state_coords.get(&MDPLongState{ m: "x", g: c_loc2 }).unwrap(),
        transitions: transitions.to_vec(),
        labelling: labelling.to_vec()
    }];

    // ----------------------------
    // Construct Local Product MDPs
    // ----------------------------
    let mut completion_states: Vec<DFA2ModelCheckingPair> = Vec::with_capacity(dfa_sensor.states.len());
    for q in dfa_sensor.states.iter() {
        completion_states.push(DFA2ModelCheckingPair { s: 999, q: *q })
    }
    let completion_label: &str = "com";
    let completion_label_hash = HashSet::from_iter(vec![completion_label].iter().cloned());
    let initial_label: &str = "ini";
    let failure_label: &str = "fai";
    let success_label: &str = "suc";

    let mut r_p_m1_dfa1: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[0].states.len() * dfa_sensor.states.len());
    let mut t_m1_dfa1: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m1_dfa1: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m1_dfa1: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states = create_states(&mdps[0].states[..], &dfa_sensor.states[..]);
    let initial_prod_state = DFA2ModelCheckingPair { s: mdps[0].initial, q: dfa_sensor.initial };


    let (mut reach, mut trans, mut labels) =
        create_local_product(&mdps[0], &dfa_sensor, initial_label, failure_label, success_label,
                         &completion_label_hash, &mdp_prod_states[..],
                         &initial_prod_state,& mut r_p_m1_dfa1, &mut t_m1_dfa1, &mut l_m1_dfa1,
                         &mut mod_l_m1_dfa1, &mut additional_states, &mut additional_transitions, &mut additional_labels);
    let product_mdp1 = ProductMDP2 {
        states: reach.to_owned(),
        initial: initial_prod_state,
        transitions: trans,
        labelling: labels
    };

    let mut num_agents: usize = 1;
    let mut num_tasks: usize = 1;
    let mut team_input: Vec<TeamInput> = Vec::with_capacity(num_tasks * num_agents);
    team_input.push(TeamInput {
        agent: 0,
        task: 0,
        product: product_mdp1,
        dead: &dfa_sensor.dead,
        acc: &dfa_sensor.acc
    });

    // ----------------------------
    // Create Team Structure
    // ----------------------------
    let mut team_mdp = TeamMDP::default();
    team_mdp.num_agents = num_agents;
    team_mdp.num_tasks = num_tasks;
    team_mdp.create_states(&team_input);
    team_mdp.create_transitions_and_labelling(&team_input, &Rewards::NEGATIVE);
    team_mdp.assign_task_rewards();
    team_mdp.modify_final_rewards(&team_input);
    /*
    let tg = team_mdp.generate_graph();
    let dot = format!("{}", Dot::new(&tg));
    let mut file = File::create("diagnostics/team_mdp.dot").unwrap();
    file.write_all(&dot.as_bytes());
     */
    // --------------------------------
    // Parameters
    // --------------------------------
    let target: Vec<f64> = vec![-80.0, 800.0];
    let epsilon: f64 = 0.0001;
    let rewards: Rewards = Rewards::NEGATIVE;
    let team_index_mappings = team_mdp.team_ij_index_mapping();
    // ---------------------------------
    // Run
    // ---------------------------------
    let start = Instant::now();
    let output = team_mdp.multi_obj_sched_synth(&target, &epsilon, &rewards, &team_index_mappings);
    let duration = start.elapsed();
    println!("Model checking time: {:?}", duration);
    let (state_count, transition_count) = team_mdp.statistics();
    println!("Model Statistics: |S|: {}, |P|: {}", state_count, transition_count);
    for m in output.mu.iter() {
        println!("output");
        let ordered_output =  dfs_sched_debugger(m, &team_mdp.states, &team_mdp.transitions, &team_mdp.initial);
        for (s, a) in ordered_output.iter() {
            println!("state: ({},{},{},{}), a: {}", s.state.s, s.state.q, s.agent, s.task, a);
        }
    }
    // ---------------------------------
    // DFS Output
    // ---------------------------------
    let graph = TeamMDP::dfs_merging(&team_mdp.initial, &output.mu, &output.v,
                                     &team_mdp.transitions[..], Some(&mdp_state_hashmap));
    let dot = format!("{}", Dot::new(&graph));
    let mut file = File::create("diagnostics/merged_sched.dot").unwrap();
    file.write_all(&dot.as_bytes());
}

/// hazard states are not the grid locations of hazards but the corresponding internal agent states
/// resulting from hazards
fn create_mdp_states<'a>(grid_states: &'a HashMap<usize,(usize,usize)>, internal_states: &'a [&'a str])
    -> (HashMap<u32, MDPLongState<'a>>, HashMap<MDPLongState<'a>, u32>, Vec<u32>) {
    let state_space_size: u32 =  (internal_states.len() * grid_states.len()) as u32;
    let state_space: Vec<u32> = (0..state_space_size).collect();
    let mut mdp_state_hash: HashMap<u32, MDPLongState<'a>> = HashMap::new();
    let mut mdp_state_coords: HashMap<MDPLongState<'a>, u32> = HashMap::new();
    let mut counter: u32 = 0;
    for i in internal_states.iter() {
        for g in 0..grid_states.len() {
            mdp_state_hash.insert(counter, MDPLongState { m: *i, g: *grid_states.get(&g).unwrap() });
            mdp_state_coords.insert(MDPLongState { m: *i, g: *grid_states.get(&g).unwrap() }, counter);
            counter += 1;
        }
    }
    (mdp_state_hash, mdp_state_coords, state_space)
}

fn create_mdp_transitions<'a>(states: &'a [u32], state_hash: &'a HashMap<u32, MDPLongState<'a>>,
                              state_coords: &'a HashMap<MDPLongState<'a>, u32>, grid_dim: &'a (usize,usize),
                              movement_p: &f64, obstacles: &'a [(usize,usize)], hazards: &'a [(usize, usize)],
                              objectives: &'a HashMap<MDPLongState<'a>, Vec<&'a str>>, all_actions: &'a [&'a str],
                              labels: &'a [HashSet<&'a str>], new_labels: &'a mut Vec<MDPLabellingPair<'a>> )
                              -> (Vec<model_checking::mdp2::Transition>, &'a mut Vec<model_checking::mdp2::MDPLabellingPair<'a>>){
    // An mdp transition looks like s, a, sprime: [], reward
    let mut transitions: Vec<model_checking::mdp2::Transition> = Vec::new();
    for s in states.iter() {
        let long_state = state_hash.get(s).unwrap();
        // check if there is an objective on the grid, if there is then this overrides the default label from the
        // configuration of the agent
        match objectives.get(&long_state) {
            None => {
                let mut word: Vec<_> = vec![];
                let hazard = hazards.iter().any(|x| *x == long_state.g);
                if long_state.m.contains("x") {
                    // default configuration states
                    if hazard {
                        let v: Vec<&str> = vec!["h"];
                        let h: HashSet<_> = HashSet::from_iter(v.iter().cloned());
                        word.push(labels.iter().find(|x| **x == h).unwrap());
                    } else {
                        word.push(labels.iter().find(|x| **x == HashSet::new()).unwrap());
                    }
                } else if long_state.m.contains("l") {
                    // leading configuration states
                    if hazard {
                        let v: Vec<&str> = vec!["h", "l"];
                        let h: HashSet<_> = HashSet::from_iter(v.iter().cloned());
                        word.push(labels.iter().find(|x| **x == h).unwrap());
                    } else {
                        let v: Vec<&str> = vec!["l"];
                        let h: HashSet<_> = HashSet::from_iter(v.iter().cloned());
                        word.push(labels.iter().find(|x| **x == h).unwrap());
                    }
                } else if long_state.m.contains("r") {
                    // leading configuration states
                    if hazard {
                        let v: Vec<&str> = vec!["h"];
                        let h: HashSet<_> = HashSet::from_iter(v.iter().cloned());
                        word.push(labels.iter().find(|x| **x == h).unwrap());
                    } else {
                        let v: Vec<&str> = vec!["r"];
                        let h: HashSet<_> = HashSet::from_iter(v.iter().cloned());
                        word.push(labels.iter().find(|x| **x == h).unwrap());
                    }
                }
                // do labels here
                new_labels.push(model_checking::mdp2::MDPLabellingPair{ s: *s, w: word})
            }
            Some(x) => {
                let mut w: Vec<&str> = Vec::new();
                let mut word: Vec<_>= Vec::new();
                if hazards.iter().any(|y| *y == long_state.g) {
                    w.push("h");
                }
                w.extend(x);
                let h: HashSet<_> = HashSet::from_iter(w.iter().cloned());
                word.push(labels.iter().find(|y| **y == h).unwrap());
                new_labels.push(model_checking::mdp2::MDPLabellingPair{ s: *s, w: word });
            }
        };
        let movement = movement_coords(s, &state_hash, &grid_dim, obstacles);
        //let hazard_state = hazards.iter().any(|x| *x == long_state.g);
        //println!("state: {:?}, hazard {:?}", long_state, hazard_state);
        for a in all_actions.iter() {
            let mut transition = model_checking::mdp2::Transition {
                s: *s,
                a: a.to_string(),
                s_prime: vec![],
                rewards: 0.0
            };
            if *a == "n" {
                let sprime = s_prime_movement(long_state, &movement, movement_p, &MovementDirection::NORTH, &state_coords);
                transition.s_prime = sprime;
                if long_state.m == "x" {
                    transition.rewards = 1.0;
                } else {
                    transition.rewards = 3.0;
                }
            } else if *a == "s" {
                let sprime = s_prime_movement(long_state, &movement, movement_p, &MovementDirection::SOUTH, &state_coords);
                transition.s_prime = sprime;
                if long_state.m == "x" {
                    transition.rewards = 1.0;
                } else {
                    transition.rewards = 3.0;
                }
            } else if *a == "e" {
                let sprime = s_prime_movement(long_state, &movement, movement_p, &MovementDirection::EAST, &state_coords);
                transition.s_prime = sprime;
                if long_state.m == "x" {
                    transition.rewards = 1.0;
                } else {
                    transition.rewards = 3.0;
                }
            } else if *a == "w" {
                let sprime = s_prime_movement(long_state, &movement, movement_p, &MovementDirection::WEST, &state_coords);
                transition.s_prime = sprime;
                if long_state.m == "x" {
                    transition.rewards = 1.0;
                } else {
                    transition.rewards = 3.0;
                }
            } else if *a == "x" {
                let x_state = state_coords.get(&MDPLongState { m: "x", g: long_state.g }).unwrap();
                transition.s_prime = vec![model_checking::mdp2::TransitionPair { s: *x_state, p: 1.0 }];
                transition.rewards = 1.0;
            } else if *a == "l" {
                let x_state = state_coords.get( &MDPLongState { m: "l", g: long_state.g }).unwrap();
                transition.s_prime = vec![model_checking::mdp2::TransitionPair{ s: *x_state, p: 1.0 }];
                transition.rewards = 1.0;
            } else if *a == "r" {
                let x_state = state_coords.get( &MDPLongState { m: "r", g: long_state.g }).unwrap();
                transition.s_prime = vec![model_checking::mdp2::TransitionPair{ s: *x_state, p: 1.0 }];
                transition.rewards = 1.0;
            }
            if !transition.s_prime.is_empty() {
                transitions.push(transition);
            }
        }
    }
    (transitions, new_labels)
}

/// A movement transitions moves in a certain intended direction but can fail and move in any of the
/// remaining directions to a set of coordinates
fn movement_coords<'a>(state: &'a u32, state_hash: &'a HashMap<u32, MDPLongState<'a>>, grid_dim: &'a (usize,usize), obstacles: &'a [(usize, usize)]) -> Movement {
    let (x,y) = state_hash.get(state).unwrap().g;
    let movement = Movement {
        north: move_north(&x, &y, &grid_dim, obstacles),
        south: move_south(&x, &y, obstacles),
        east: move_east(&x, &y, &grid_dim, obstacles),
        west: move_west(&x, &y, obstacles)
    };
    movement
}

// ----------------------
// Movement
// ----------------------
fn move_north(x: &usize, y: &usize, grid_dim: &(usize,usize), obstacles: &[(usize,usize)]) -> (usize, usize) {
    if *y < grid_dim.1 - 1 {
        if obstacles.iter().any(|(x1,y1)| (*x,*y + 1) == (*x1,*y1)) {
            (*x,*y)
        } else {
            (*x,*y + 1)
        }
    } else {
        (*x,*y)
    }
}

fn move_south(x: &usize, y: &usize, obstacles: &[(usize,usize)]) -> (usize, usize) {
    if *y > 0 {
        // if the coordinate is not less than zero than process the obstacle
        if obstacles.iter().any(|(x1,y1)| (*x,*y - 1) == (*x1,*y1)) {
            (*x, *y)
        } else {
            (*x,*y - 1)
        }
    } else {
        (*x,*y)
    }
}

fn move_west(x: &usize, y: &usize, obstacles: &[(usize,usize)]) -> (usize, usize) {
    if *x > 0 {
        if obstacles.iter().any(|(x1, y1)| (*x - 1,*y) == (*x1,*y1)) {
            (*x, *y)
        } else {
            (*x - 1, *y)
        }
    } else {
        (*x, *y)
    }
}

fn move_east(x: &usize, y: &usize, grid_dim: &(usize,usize), obstacles: &[(usize,usize)]) -> (usize, usize) {
    if *x < grid_dim.0 - 1 {
        if obstacles.iter().any(|(x1, y1)| (*x + 1,*y) == (*x1,*y1)) {
            (*x,*y)
        } else {
            (*x + 1,*y)
        }
    } else {
        (*x,*y)
    }
}

fn s_prime_movement<'a, 'b>(state: &'a MDPLongState<'a>, movements: &'a Movement, p_dir: &'b f64,
                            direction: &'b MovementDirection, state_coords: &'a HashMap<MDPLongState<'a>, u32>)
    -> Vec<model_checking::mdp2::TransitionPair> {
    let mut sprime: Vec<model_checking::mdp2::TransitionPair> = match direction {
        MovementDirection::NORTH => {
            vec![ TransitionPair{ s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.north }).unwrap(), p: 1.0 }]
        }
        MovementDirection::SOUTH => {
            vec![ TransitionPair{ s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.south }).unwrap(), p: 1.0 }]
        }
        MovementDirection::EAST => {
            vec![ TransitionPair{ s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.east }).unwrap(), p: 1.0 }]
        }
        MovementDirection::WEST => {
            vec![ TransitionPair{ s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.west }).unwrap(), p: 1.0 }]
        }
    };
    /*vec![
        TransitionPair{
            s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.north }).unwrap(),
            p: match direction {
                MovementDirection::NORTH => *p_dir,
                MovementDirection::SOUTH => 0.0,
                _ => { (1f64 - *p_dir) / 2f64 }
            }},
        TransitionPair{
            s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.south }).unwrap(),
            p: match direction {
                MovementDirection::SOUTH => *p_dir,
                MovementDirection::NORTH => 0.0,
                _ => { (1f64 - *p_dir) / 2f64 }
            }},
        TransitionPair{
            s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.east }).unwrap(),
            p: match direction {
                MovementDirection::EAST => *p_dir,
                MovementDirection::WEST => 0.0,
                _ => { (1f64 - *p_dir) / 2f64 }
            }},
        TransitionPair{
            s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.west }).unwrap(),
            p: match direction {
                MovementDirection::WEST => *p_dir,
                MovementDirection::EAST => 0.0,
                _ => { (1f64 - *p_dir) / 2f64 }
            }}
    ];

         */
    sprime
}

// ----------------------
// Grid Definition
// ----------------------
fn create_grid(grid: (usize,usize)) -> HashMap<usize,(usize,usize)> {
    let x: Vec<usize> = (0..grid.0).collect();
    let y: Vec<usize> = (0..grid.1).collect();
    let mut states: HashMap<usize, (usize,usize)> = HashMap::new();
    let cp = x.into_iter().cartesian_product(y.into_iter());
    for (k, (i,j)) in cp.into_iter().enumerate() {
        states.insert(k, (i,j));
    }
    states
}

enum MovementDirection {
    NORTH,
    SOUTH,
    EAST,
    WEST
}

#[derive(Debug, Clone)]
struct Movement {
    north: (usize,usize),
    south: (usize,usize),
    east: (usize,usize),
    west: (usize,usize),
}
