mod model_checking;
use model_checking::decomp_team_mdp::*;
use std::fs::File;
use std::io::Write;
use petgraph::{dot::Dot, Direction};
use std::time::{Duration, Instant};
use regex::Regex;

use model_checking::helper_methods::{parse_language, read_dfa_json};
use model_checking::mdp::*;
use model_checking::dfa::*;
use model_checking::product_dfa::*;
use itertools::Itertools;
use std::collections::HashMap;

fn main() {
    let grid_dim: (usize, usize) = (30,5);
    let grid_state_space: HashMap<usize,(usize,usize)> = create_grid(grid_dim);
    let c_loc: (usize,usize) = (0,2);
    let c_loc2: (usize,usize) = (3,0);
    let act1: Vec<&str> = vec!["x", "l", "r"];
    let act2: Vec<&str> = vec!["n", "s", "e", "w"];
    // Different agents may have different obstacles
    let obstacles: [(usize, usize); 3] = [(1,1),(1,2),(1,3)];
    // hazards
    let hazards: [(usize, usize); 1] = [(1,0)];
    // Different agents may have different probabilities of moving in cardinal directions
    let movement_p: f64 = 0.95;
    //
    let mut all_act: Vec<&str> = act1.to_vec();
    all_act.extend(act2.iter().copied());
    let mut obj_points: HashMap<MDPLongState,&str> = HashMap::new();
    obj_points.insert(MDPLongState{ m: "l", g: (0, 1) }, "start");
    //obj_points.insert(MDPLongState{ m: "l", d: "d2", g: (0, 0) }, "start");
    obj_points.insert(MDPLongState{ m: "l", g: (25, 3) }, "finish");
    obj_points.insert(MDPLongState{ m: "r", g: (4, 3) }, "sensor");
    obj_points.insert(MDPLongState{ m: "r", g: (25, 0)}, "download");
    // --------------------------------------
    // Construct the Agent-Environment Model
    // --------------------------------------
    let (mdp_state_hashmap, mdp_state_coords, mdp_states) =
        create_mdp_states(&grid_state_space, &act1);
    let (transitions, labelling) =
        create_mdp_transitions(&mdp_states[..], &mdp_state_hashmap, &mdp_state_coords,
                               &grid_dim, &movement_p,&obstacles[..], &hazards[..],
                               &obj_points, &all_act[..]);

    let mdps: Vec<MDP> = vec![MDP{
        states: mdp_states.to_vec(),
        initial: *mdp_state_coords.get(&MDPLongState{ m: "x", g: c_loc }).unwrap(),
        transitions: transitions.to_vec(),
        labelling: labelling.to_vec()
    }, MDP{
        states: mdp_states,
        initial: *mdp_state_coords.get(&MDPLongState{ m: "x", g: c_loc2 }).unwrap(),
        transitions: transitions.to_vec(),
        labelling: labelling.to_vec()
    }];
    // ----------------------------
    // Parse DFAs
    // ----------------------------
    let dfas = read_dfas();
    // ----------------------------
    // Construct Local Product MDPs
    // ----------------------------
    let mut num_agents: usize = mdps.len();
    let mut num_tasks: usize = dfas.len();
    let mut team_input: Vec<TeamInput> = vec![TeamInput::default(); num_tasks * num_agents];
    let mut team_counter: usize = 0;
    for (i, mdp) in mdps.iter().enumerate() {
        for (j, dfa) in dfas.iter() {
            let initial_state = DFAModelCheckingPair { s: mdp.initial, q: dfa.initial };
            let local_product = model_checking::dfa::create_local_product(&initial_state, mdp, dfa);
            //println!("local product states: {:?}", local_product.states);
            //println!("local product transitions: {:?}", local_product.transitions);
            team_input[team_counter] = TeamInput {
                agent: i,
                task: *j,
                product: local_product,
                dead: dfa.dead.to_vec(),
                acc: dfa.acc.to_vec()
            };
            team_counter += 1;
        }
    }

    // ----------------------------
    // Create Team Structure
    // ----------------------------
    let target: Vec<f64> = vec![-120.0, -120.0, 800.0];
    let epsilon: f64 = 0.0001;
    let mut team_mdp = TeamMDP::default();
    team_mdp.num_agents = num_agents;
    team_mdp.num_tasks = num_tasks;
    team_mdp.create_states(&team_input);
    team_mdp.create_transitions_and_labelling(&team_input, &Rewards::NEGATIVE);
    team_mdp.assign_task_rewards();
    team_mdp.modify_final_rewards(&team_input);

    let tg = team_mdp.generate_graph();
    let dot = format!("{}", Dot::new(&tg));
    let mut file = File::create("diagnostics/team_mdp.dot").unwrap();
    file.write_all(&dot.as_bytes());

    let rewards: Rewards = Rewards::NEGATIVE;
    let team_index_mappings = team_mdp.team_ij_index_mapping();
    let start = Instant::now();
    let output = team_mdp.multi_obj_sched_synth(&target, &epsilon, &rewards, &team_index_mappings);
    let duration = start.elapsed();
    println!("Model checking time: {:?}", duration);
    let (state_count, transition_count) = team_mdp.statistics();
    println!("Model Statistics: |S|: {}, |P|: {}", state_count, transition_count);
    let graph = TeamMDP::dfs_merging(&team_mdp.initial, &output.mu, &output.v, &team_mdp.transitions[..], Some(&mdp_state_hashmap));
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
                              objectives: &'a HashMap<MDPLongState<'a>,&'a str>, all_actions: &'a [&'a str])
                              -> (Vec<model_checking::mdp::Transition>, Vec<model_checking::mdp::MDPLabellingPair>){
    // An mdp transition looks like s, a, sprime: [], reward
    let mut transitions: Vec<model_checking::mdp::Transition> = Vec::new();
    let mut labels: Vec<model_checking::mdp::MDPLabellingPair> = Vec::new();
    for s in states.iter() {
        let long_state = state_hash.get(s).unwrap();
        // check if there is an objective on the grid, if there is then this overrides the default label from the
        // configuration of the agent
        match objectives.get(&long_state) {
            None => {
                if long_state.m.contains("x") {
                    // default configuration states
                    labels.push(model_checking::mdp::MDPLabellingPair{ s: *s, w: "".to_string()})
                } else if long_state.m.contains("l") {
                    // leading configuration states
                    labels.push(model_checking::mdp::MDPLabellingPair{ s: *s, w: "lead".to_string()})
                } else if long_state.m.contains("r") {
                    // leading configuration states
                    labels.push(model_checking::mdp::MDPLabellingPair{ s: *s, w: "sensor".to_string()})
                }
            }
            Some(x) => labels.push(model_checking::mdp::MDPLabellingPair{ s: *s, w: x.to_string() })
        };
        let movement = movement_coords(s, &state_hash, &grid_dim, obstacles);
        let hazard_state = hazards.iter().any(|x| *x == long_state.g);
        //println!("state: {:?}, hazard {:?}", long_state, hazard_state);
        for a in all_actions.iter() {
            let mut transition = model_checking::mdp::Transition {
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
                transition.s_prime = vec![model_checking::mdp::TransitionPair { s: *x_state, p: 1.0 }];
            } else if *a == "l" {
                let x_state = state_coords.get( &MDPLongState { m: "l", g: long_state.g }).unwrap();
                transition.s_prime = vec![model_checking::mdp::TransitionPair{ s: *x_state, p: 1.0 }];
            } else if *a == "r" {
                let x_state = state_coords.get( &MDPLongState { m: "r", g: long_state.g }).unwrap();
                transition.s_prime = vec![model_checking::mdp::TransitionPair{ s: *x_state, p: 1.0 }];
            }
            if !transition.s_prime.is_empty() {
                transitions.push(transition);
            }
        }
    }
    (transitions, labels)
}

fn read_dfas() -> Vec<(usize, DFA)> {
    let mut dfa_parse: Vec<(usize, DFA)> = Vec::new();
    let dfas: Option<Vec<DFA>> = match read_dfa_json("examples/uav_tasks.json") {
        Ok(u) => {
            Some(u)
        },
        Err(e) => {println!("Error: {}", e); None}
    };
    match dfas {
        None => {println!("There was an error reading the DFAs from examples/uav_tasks.json")}
        Some(x) => {
            for (i, mut aut) in x.into_iter().enumerate() {
                for transition in aut.delta.iter_mut() {
                    //println!("w: {:?}", transition.w);
                    match parse_language(&aut.sigma, &transition.w) {
                        None => {}
                        Some(x) => {
                            //println!("parsed words: {:?}", x);
                            transition.w = x;
                        }
                    }
                }
                dfa_parse.push((i, aut));
            }
        }
    }
    dfa_parse
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
    if obstacles.iter().any(|(x1,y1)| (*x,*y + 1) == (*x1,*y1)) {
        (*x,*y)
    } else {
        if *y < grid_dim.1 - 1 {
            (*x,*y + 1)
        } else {
            (*x,*y)
        }
    }
}

fn move_south(x: &usize, y: &usize, obstacles: &[(usize,usize)]) -> (usize, usize) {
    if obstacles.iter().any(|(x1,y1)| (*x,*y - 1) == (*x1,*y1)) {
        (*x, *y)
    } else {
        if *y > 0 {
            (*x,*y - 1)
        } else {
            (*x,*y)
        }
    }
}

fn move_west(x: &usize, y: &usize, obstacles: &[(usize,usize)]) -> (usize, usize) {
    if obstacles.iter().any(|(x1, y1)| (*x - 1,*y) == (*x1,*y1)) {
        (*x, *y)
    } else {
        if *x > 0 {
            (*x - 1, *y)
        } else {
            (*x, *y)
        }

    }

}

fn move_east(x: &usize, y: &usize, grid_dim: &(usize,usize), obstacles: &[(usize,usize)]) -> (usize, usize) {
    if obstacles.iter().any(|(x1, y1)| (*x + 1,*y) == (*x1,*y1)) {
        (*x,*y)
    } else {
        if *x < grid_dim.0 - 1 {
            (*x + 1,*y)
        } else {
            (*x,*y)
        }
    }

}

fn s_prime_movement<'a, 'b>(state: &'a MDPLongState<'a>, movements: &'a Movement, p_dir: &'b f64,
                            direction: &'b MovementDirection, state_coords: &'a HashMap<MDPLongState<'a>, u32>)
    -> Vec<model_checking::mdp::TransitionPair> {
    let mut sprime: Vec<model_checking::mdp::TransitionPair> = vec![
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

