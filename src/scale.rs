mod model_checking;
use model_checking::dfa::*;
use model_checking::mdp::*;
use model_checking::decomp_team_mdp::*;
use model_checking::helper_methods::*;
use std::time::{Duration, Instant};

fn main() {
    // -------------------------------
    // Task Scalability
    // -------------------------------
    //let mdp_path_val: &str = "agents2.json";
    let mdps: Option<Vec<MDP>> = match read_mdp_json(format!("examples/{}",mdp_path_val)) {
        Ok(u) => {
            Some(u)
        },
        Err(e) => {println!("Error: {}", e); None}
    };
    let mdp_parse: Vec<MDP> = mdps.unwrap();
    let mut dfas: Vec<DFA> = Vec::with_capacity(100);
    let num_agents: usize = 2;
    let mut num_tasks: usize = 50;
    let mut team_input: Vec<TeamInput> = vec![TeamInput::default(); num_tasks * num_agents];
    let mut team_counter: usize = 0;
    let target_parse: Vec<f64> = vec![
        -6000.0, -6000.0,
        900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 800.0,
        800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 700.0,
        700.0, 700.0, 700.0, 700.0, 700.0, 700.0, 700.0, 700.0, 700.0, 600.0,
        600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 500.0,
        500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 400.0,
    ];
    let epsilon: f64 = 0.00001;
    for k in 1..(num_tasks + 1) {
        dfas.push(construct_message_dfa(&k));
        for (i,mdp) in mdp_parse.iter().enumerate(){
            let mut local_product: DFAProductMDP = DFAProductMDP::default();
            local_product.initial = DFAModelCheckingPair{ s: mdp.initial, q: dfas[k-1].initial };
            local_product.create_states(&mdp, &dfas[k-1]);
            local_product.create_transitions(&mdp, &dfas[k-1]);
            let mut g = local_product.generate_graph();
            let initially_reachable = local_product.reachable_from_initial(&g);
            let (prune_states_indices, prune_states) : (Vec<usize>, Vec<DFAModelCheckingPair>) =
                local_product.prune_candidates(&initially_reachable);
            local_product.prune_states_transitions(&prune_states_indices, &prune_states);
            local_product.create_labelling(&mdp);
            local_product.modify_complete(&dfas[k-1]);
            //println!("modifying agent: {} task: {}", i, j);
            local_product.edit_labelling(&dfas[k-1], &mdp);

            team_input[team_counter] = TeamInput {
                agent: i,
                task: k-1,
                product: local_product,
                dead: dfas[k-1].dead.to_vec(),
                acc: dfas[k-1].acc.to_vec()
            };
            team_counter += 1;
        }
    }
    let mut team_mdp = TeamMDP::default();
    team_mdp.num_agents = num_agents;
    team_mdp.num_tasks = num_tasks;
    team_mdp.create_states(&team_input);
    team_mdp.create_transitions_and_labelling(&team_input, &Rewards::NEGATIVE);
    team_mdp.assign_task_rewards();
    team_mdp.modify_final_rewards(&team_input);
    let rewards: Rewards = Rewards::NEGATIVE;
    /**/
    let mut v: Vec<Duration> = Vec::with_capacity(100);
    let samples: usize = 1;
    for _i in 0..samples {
        let start = Instant::now();
        let output = team_mdp.multi_obj_sched_synth(&target_parse, &epsilon, &rewards);
        //let output = team_mdp.multi_obj_sched_synth_non_iter(&target_parse, &epsilon, &rewards);
        let duration = start.elapsed();
        v.push(duration);
    }
    let sum_duration = v.iter().fold(0.0, |acc, &sum| acc + sum.as_secs_f64());
    println!("ave duration:{}", sum_duration/ samples as f64);
    let (s, p) = team_mdp.statistics();
    println!("s: {}, p: {}", s, p);
    // -------------------------------
    // Agent Scalability
    // -------------------------------
    // construct DFA
    /*
    let num_tasks: usize = 2;
    let mut dfas: Vec<DFA> = Vec::with_capacity(2);
    for k in 1..3 {
        dfas.push(construct_message_dfa(&k));
    }
    let num_agents: usize = 3;
    let mut team_input: Vec<TeamInput> = vec![TeamInput::default(); num_tasks * num_agents];
    let mut mdps: Vec<MDP> = Vec::with_capacity(num_agents);
    let mut team_counter: usize = 0;
    let target_parse: Vec<f64> = vec![
        -5.0, -5.0,-5.0,
        900.0, 900.0
    ];
    let epsilon: f64 = 0.00001;
    for j in 0..dfas.iter().len() {
        for i in 0..num_agents {
            mdps.push(construct_mdp());
            let mut local_product: DFAProductMDP = DFAProductMDP::default();
            local_product.initial = DFAModelCheckingPair{ s: mdps[i].initial, q: dfas[j].initial };
            local_product.create_states(&mdps[i], &dfas[j]);
            local_product.create_transitions(&mdps[i], &dfas[j]);
            let mut g = local_product.generate_graph();
            let initially_reachable = local_product.reachable_from_initial(&g);
            let (prune_states_indices, prune_states) : (Vec<usize>, Vec<DFAModelCheckingPair>) =
                local_product.prune_candidates(&initially_reachable);
            local_product.prune_states_transitions(&prune_states_indices, &prune_states);
            local_product.create_labelling(&mdps[i]);
            local_product.modify_complete(&dfas[j]);
            local_product.edit_labelling(&dfas[j], &mdps[i]);

            team_input[team_counter] = TeamInput {
                agent: i,
                task: j,
                product: local_product,
                dead: dfas[j].dead.to_vec(),
                acc: dfas[j].acc.to_vec()
            };
            team_counter += 1;
        }
    }

    let mut team_mdp = TeamMDP::default();
    team_mdp.num_agents = num_agents;
    team_mdp.num_tasks = num_tasks;
    team_mdp.create_states(&team_input);
    team_mdp.create_transitions_and_labelling(&team_input, &Rewards::NEGATIVE);
    team_mdp.assign_task_rewards();
    team_mdp.modify_final_rewards(&team_input);

    let rewards: Rewards = Rewards::NEGATIVE;
    /**/
    let mut v: Vec<Duration> = Vec::with_capacity(100);
    let samples: usize = 100;
    for _i in 0..samples {
        let start = Instant::now();
        //let output = team_mdp.multi_obj_sched_synth(&target_parse, &epsilon, &rewards);
        let output = team_mdp.multi_obj_sched_synth_non_iter(&target_parse, &epsilon, &rewards);
        let duration = start.elapsed();
        v.push(duration);
    }
    let sum_duration = v.iter().fold(0.0, |acc, &sum| acc + sum.as_secs_f64());
    println!("ave duration:{}", sum_duration/ samples as f64);
    let (s, p) = team_mdp.statistics();
    println!("s: {}, p: {}", s, p);
    */

    // -------------------------------
    // Task - Agent Scalability
    // -------------------------------
    /*
    // construct an agent grid
    let mut stats: Vec<((usize,usize),(usize,usize))> = Vec::new();
    let mut runtime_x: Vec<usize> = Vec::new();
    let mut runtime_y: Vec<usize> = Vec::new();
    let mut runtime_z: Vec<f64> = Vec::new();
    for k1 in 1..15 {
        for k2 in 1..15 {
            println!("Building task: {}, agent: {} experiment", k1, k2);
            let num_tasks: usize = k1;
            let mut dfas: Vec<DFA> = Vec::with_capacity(k1);
            for k in 1..k1 + 1 {
                dfas.push(construct_message_dfa(&k));
            }
            let num_agents: usize = k2;
            let mut team_input: Vec<TeamInput> = vec![TeamInput::default(); num_tasks * num_agents];
            let mut mdps: Vec<MDP> = Vec::with_capacity(num_agents);
            let mut team_counter: usize = 0;
            let mut target_parse: Vec<f64> = Vec::with_capacity(num_tasks + num_agents);
            for _ in 0..k2{
                target_parse.push(-(100.0 * k1 as f64) / k2 as f64 );
            }
            for _ in 0..k1 {
                target_parse.push(500.0);
            }
            let epsilon: f64 = 0.00001;
            for j in 0..dfas.iter().len() {
                for i in 0..num_agents {
                    mdps.push(construct_mdp());
                    let mut local_product: DFAProductMDP = DFAProductMDP::default();
                    local_product.initial = DFAModelCheckingPair{ s: mdps[i].initial, q: dfas[j].initial };
                    local_product.create_states(&mdps[i], &dfas[j]);
                    local_product.create_transitions(&mdps[i], &dfas[j]);
                    let mut g = local_product.generate_graph();
                    let initially_reachable = local_product.reachable_from_initial(&g);
                    let (prune_states_indices, prune_states) : (Vec<usize>, Vec<DFAModelCheckingPair>) =
                        local_product.prune_candidates(&initially_reachable);
                    local_product.prune_states_transitions(&prune_states_indices, &prune_states);
                    local_product.create_labelling(&mdps[i]);
                    local_product.modify_complete(&dfas[j]);
                    local_product.edit_labelling(&dfas[j], &mdps[i]);

                    team_input[team_counter] = TeamInput {
                        agent: i,
                        task: j,
                        product: local_product,
                        dead: dfas[j].dead.to_vec(),
                        acc: dfas[j].acc.to_vec()
                    };
                    team_counter += 1;
                }
            }
            let mut team_mdp = TeamMDP::default();
            team_mdp.num_agents = num_agents;
            team_mdp.num_tasks = num_tasks;
            team_mdp.create_states(&team_input);
            team_mdp.create_transitions_and_labelling(&team_input, &Rewards::NEGATIVE);
            team_mdp.assign_task_rewards();
            team_mdp.modify_final_rewards(&team_input);

            let rewards: Rewards = Rewards::NEGATIVE;
            /**/
            let mut v: Vec<Duration> = Vec::with_capacity(100);
            let samples: usize = if k1 < 10 && k2 < 10 {
                10
            } else {
                1
            };
            for _i in 0..samples {
                let start = Instant::now();
                //let output = team_mdp.multi_obj_sched_synth_non_iter(&target_parse, &epsilon, &rewards, &false);
                let output = team_mdp.multi_obj_sched_synth_non_iter(&target_parse, &epsilon, &rewards, &false);
                let duration = start.elapsed();
                v.push(duration);
            }
            let sum_duration = v.iter().fold(0.0, |acc, &sum| acc + sum.as_secs_f64());
            //println!("ave duration:{}", sum_duration/ samples as f64);
            let (s, p) = team_mdp.statistics();
            stats.push(((k2,k1),(s,p)));
            runtime_x.push(k2);
            runtime_y.push(k1);
            runtime_z.push( ((sum_duration / samples as f64) * 10000.0 ).round() / 10000.0);
        }
    }
    //println!("stats: {:?}", stats);
    println!("run time x: {:?}", runtime_x);
    println!("run time y: {:?}", runtime_y);
    println!("run time z: {:?}", runtime_z);
    */
}

fn construct_message_dfa(k: &usize) -> DFA {
    // we are only concerned with k - 1 as there will always be one sprint;
    let base_states: u32 = 4; // the number of states with one sprint
    let states: Vec<u32> = (0..(3 + *k as u32)).collect();
    // the second to last state will always be accepting
    let acc: u32 = states.len() as u32 - 2;
    let rej: u32 = states.len() as u32 - 1;
    // the last state will always be rejecting
    let sigma : Vec<String> = vec![String::from("initiate"), String::from("ready"), 
                                   String::from("exit"), String::from("sprint"), String::from("")];
    // There will always be 4 transitions + 3 * k
    let mut delta: Vec<model_checking::dfa::DFATransitions> = Vec::with_capacity(4 + 3 * *k);
    // we can start with the four static transitions and then move onto the dynamic transitions
    // static transitions
    delta.push(DFATransitions {
        q: 0,
        w: vec![String::from("ready"), String::from("exit"), String::from("sprint"), String::from("")],
        q_prime: 0
    });
    delta.push(DFATransitions {
        q: 0,
        w: vec![String::from("initiate")],
        q_prime: 1
    });
    delta.push(DFATransitions {
        q: acc,
        w: sigma.to_vec(),
        q_prime: acc
    });
    delta.push(DFATransitions {
        q: rej,
        w: sigma.to_vec(),
        q_prime: rej
    });
    let mut init_dyn_state: u32 = 1;
    for _i in 0..*k {
        delta.push(DFATransitions {
            q: init_dyn_state,
            w: vec![String::from("ready"), String::from("initiate"), String::from("")],
            q_prime: init_dyn_state
        });
        delta.push(DFATransitions {
            q: init_dyn_state,
            w: vec![String::from("exit")],
            q_prime: rej
        });
        delta.push(DFATransitions{
            q: init_dyn_state,
            w: vec![String::from("sprint")],
            q_prime: init_dyn_state + 1
        });
        init_dyn_state += 1;
    }
    DFA {
        states,
        sigma,
        initial: 0,
        delta,
        acc: vec![acc],
        dead: vec![rej]
    }
}

fn construct_mdp() -> MDP {
    let states: Vec<u32> = vec![0,1,2,3,4];
    let initial: u32 = 0;
    let transitions: Vec<model_checking::mdp::Transition> = vec![Transition{
        s: 0,
        a: "a".to_string(),
        s_prime: vec![TransitionPair{ s: 0, p: 0.05 }, TransitionPair{ s: 1, p: 0.95 }],
        rewards: 3.0
    },Transition {
        s: 1,
        a: "a".to_string(),
        s_prime: vec![TransitionPair{s: 2, p: 1.0}],
        rewards: 2.0
    },Transition {
         s: 2,
         a: "a".to_string(),
         s_prime: vec![ TransitionPair {s: 3, p: 0.99}, TransitionPair {s: 4, p: 0.01}],
         rewards: 3.0
     },Transition {
         s: 2,
         a: "c".to_string(),
         s_prime: vec![ TransitionPair {s: 3, p: 0.8}, TransitionPair {s: 4, p: 0.2}],
         rewards: 1.0
     },Transition {
         s: 2,
         a: "b".to_string(),
         s_prime: vec![ TransitionPair {s: 4, p: 1.0}],
         rewards: 0.5
     },Transition{
         s: 3,
         a: "a".to_string(),
         s_prime: vec![TransitionPair {s: 2, p: 1.0}],
         rewards: 0.5
     },Transition {
         s: 4,
         a: "a".to_string(),
         s_prime: vec![ TransitionPair {s: 0, p: 1.0}],
         rewards: 3.0
     }];

    let labelling: Vec<model_checking::mdp::MDPLabellingPair> = vec![
        MDPLabellingPair { s: 0, w: "".to_string() },
        MDPLabellingPair { s: 1, w: "initiate".to_string() },
        MDPLabellingPair { s: 2, w: "ready".to_string() },
        MDPLabellingPair { s: 3, w: "sprint".to_string() },
        MDPLabellingPair { s: 4, w: "exit".to_string() }
    ];

    MDP {
        states,
        initial,
        transitions,
        labelling
    }
}