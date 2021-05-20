//use itertools::{Itertools, enumerate};
use clap::{clap_app, Values};
use lib::{read_mdp_json, MDP, DFA, DFAProductMDP, read_dfa_json, DFAModelCheckingPair, TeamInput, TeamMDP, NonNan, absolute_diff_vect, read_target, Target, Mu, Rewards, Alg1Output};
use std::fs::File;
use std::io::Write;
use petgraph::{dot::Dot};

fn main() {
    let matches = clap_app!(motap =>
        (version: "0.1")
        (author: "Thomas Robinson")
        (@subcommand motap =>
            (about: "Fast Multiagent System Task Allocation and Planning with Multiple Objectives")
            (@arg mPATH: --mpath <PATH_REF1> "MDP Model, takes a json file in the form of transitions e.g.
            {
              \"states\": [0,1,2,3,4],
              \"initial\": 1,
              \"transitions\":
               [{
                  \"s\": 0,
                  \"a\": \"a\",
                  \"s_prime\": [{\"s\": 0,\"p\": 0.1},{\"s\": 1,\"p\": 0.9}],
                  \"rewards\": 1
               },{
                  \"s\": 1,
                  \"a\": \"a\",
                  \"s_prime\": [{\"s\":2, \"p\": 1.0}],
                  \"rewards\": 1.0
               },...]
              \"labelling\": [
                {\"s\":  0, \"w\": \"\"},
                {\"s\":  1, \"w\": \"initiate(k)\"},
                ...
              ]
            }
           ")
           (@arg dPATH: --dpath <PATH_REF2> "DRA Model, vector of DRA models to apply\
           {
              \"states\": [0,1,2,3],
              \"initial\": 1,
              \"delta\": [
                {\"q\":  0, \"w\": [\"initiate(1)\"], \"q_prime\": 1},
                {\"q\":  0, \"w\": [\"sprint(1)\", \"exit\", \"ready\"], \"q_prime\": 0},
                {\"q\":  1, \"w\": [\"ready\", \"initiate(1)\"], \"q_prime\": 1},
                {\"q\":  1, \"w\": [\"sprint(1)\"], \"q_prime\": 2},
                {\"q\":  1, \"w\": [\"exit\"], \"q_prime\": 3},
                {\"q\":  2, \"w\": [\"ready\", \"exit\", \"sprint(1)\", \"initiate(1)\"], \"q_prime\": 2},
                {\"q\":  3, \"w\": [\"ready\", \"exit\", \"sprint(1)\", \"initiate(1)\"], \"q_prime\": 3}
              ],
              \"acc\": [
                {\"l\":  [], \"k\":  [2]}
              ]
            }
           ")
            (@arg TARGET: --target <PATH_REF3> "Input of the target vector")
            (@arg VERBOSE: -v --verbose [VERBOSITY] default_value("0") "Level of verbosity \
               0 - errors\
               1 - inputs
               2 - algorithm debugging
               3 - results
               "
            )
            (@arg TEST: -t --test "test some code")
            (@arg LABEL: -l --label "present the labelling of a product MDP")
            (@arg GRAPH: -g --graph [GRAPH_TYPE] default_value("0") "generates a product graph type: \
               1 - Product Graph
               2 - Modified Product Graph
               3 - Team MDP Graph
               4 - All of the above
               "
            )
            (@arg RUN: -r --run "run task allocation and planning in a team MDP")
            (@arg EPS: --eps [EPSILON] default_value("0.005"))
        )
    ).get_matches();

    let test: bool = match matches.subcommand() {
        ("motap", Some(f)) => {
            match f.occurrences_of("TEST") {
                0 => false,
                1 | _ => true
            }
        },
        (_,_) => false
    };

    let labelling: bool = match matches.subcommand() {
        ("motap", Some(f)) => {
            match f.occurrences_of("LABEL") {
                0 => false,
                1 | _ => true
            }
        },
        (_,_) => false
    };

    let graph_type: u32 = match matches.subcommand() {
        ("motap", Some(f)) => {
            f.value_of("GRAPH").unwrap().parse().unwrap()
        },
        (_,_) => 0
    };

    // If block used for testing certain inputs with the motap CLI

    let verbose: u32 = match matches.subcommand() {
        ("motap", Some(f)) => {
            f.value_of("VERBOSE").unwrap().parse().unwrap()
        },
        (_,_) => {0}
    };

    let mdp_path_val = match matches.subcommand() {
        ("motap", Some(f)) => {
            if verbose >= 1 {
                println!("path: {}", f.value_of("mPATH").unwrap());
            }
            f.value_of("mPATH").unwrap()
        },
        (_,_) => {""}
    };

    let dra_path_val = match matches.subcommand() {
        ("motap", Some(f)) => {
            if verbose >= 1 {
                println!("path: {}", f.value_of("dPATH").unwrap());
            }
            f.value_of("dPATH").unwrap()
        },
        (_,_) => {""}
    };

    let mdps: Option<Vec<MDP>> = match read_mdp_json(mdp_path_val) {
        Ok(u) => {
            if verbose == 1 {
                println!("{:?}", u);
            }
            Some(u)
        },
        Err(e) => {println!("Error: {}", e); None}
    };

    let target_path = match matches.subcommand() {
        ("motap", Some(f)) => {
            f.value_of("TARGET").unwrap()
        }
        (_,_) => {""}
    };

    let dfas: Option<Vec<DFA>> = match read_dfa_json(dra_path_val) {
        Ok(u) => {
            if verbose == 1 {
                println!("{:?}", u);
            }
            Some(u)
        },
        Err(e) => {println!("Error: {}", e); None}
    };

    let targets: Option<Target> = match read_target(target_path) {
        Ok(u) => {
            Some(u)
        },
        Err(e) => {println!("Error: {}", e); None}
    };

    let run: bool = match matches.subcommand() {
        ("motap", Some(f)) => {
            match f.occurrences_of("RUN") {
                0 => false,
                1 | _ => true
            }
        },
        (_,_) => false
    };

    let epsilon: f64 = match matches.subcommand() {
        ("motap", Some(f)) => {
            f.value_of("EPS").unwrap().parse().unwrap()
        },
        (_,_) => 0.0
    };

    //let safety_present: bool = false;
    let mut dfa_parse: Vec<(usize, DFA)> = Vec::new();
    let mut mdp_parse: Vec<(usize, MDP)> = Vec::new();
    let mut target_parse: Vec<f64> = Vec::new();
    let mut num_agents: usize = 0;
    let mut num_tasks: usize = 0;
    if test {
        /*
        //lp3();

        let result = lp4(&hullset, &target, &4usize);
        println!("result: {:?}", result);

         */
        let target: Vec<f64> = vec![-7.0, -7.0, 0.5, 0.5];
        let hullset: Vec<Vec<f64>> = vec![
            vec![-8.5, 0.0, 0.0, 0.0],
            vec![0.0, -13.66, 0.0, 0.0],
            vec![-6.85, -6.33, 0.8, 0.0],
            vec![-13.18, 0.0, 0.0, 0.64]
        ];
        //let val = lp6(&hullset, &target, &4, &num_agents);
        //println!("r: {:?}", val);
    }
    match dfas {
        None => {println!("There was an error reading the DFAs from {}", dra_path_val)}
        Some(x) => {
            num_tasks = x.len();
            for (i, aut) in x.into_iter().enumerate() {
                dfa_parse.push((i, aut));
            }
        }
    }
    match mdps {
        Some(z) => {
            num_agents = z.len();
            for (j, mdp) in z.into_iter().enumerate() {
                mdp_parse.push((j, mdp));
            }
        },
        None => {println!("There was an error reading the mdp from {}", mdp_path_val); return }
    }
    match targets {
        None => {println!("There was an error reading the targets from {}", target_path)},
        Some(z) => {
            target_parse = z.target;
        }
    }
    println!("target: {:?}", target_parse);
    let mut team_input: Vec<TeamInput> = vec![TeamInput::default(); num_tasks * num_agents];
    let mut team_counter: usize = 0;
    for (i, mdp) in mdp_parse.iter() {
        for (j, task) in dfa_parse.iter() {
            let mut local_product: DFAProductMDP = DFAProductMDP::default();
            local_product.create_states(&mdp, task);
            local_product.create_transitions(&mdp, task);
            let mut g = local_product.generate_graph();
            let initially_reachable = local_product.reachable_from_initial(&g);
            let (prune_states_indices, prune_states) : (Vec<usize>, Vec<DFAModelCheckingPair>) =
                local_product.prune_candidates(&initially_reachable);
            if graph_type > 0 {
                local_product.prune_graph(&mut g, &prune_states_indices);
                let dot = format!("{}", Dot::new(&g));
                let mut file = File::create(format!("product_mdp_{}.dot", j)).unwrap();
                file.write_all(&dot.as_bytes());
            }
            local_product.prune_states_transitions(&prune_states_indices, &prune_states);
            local_product.create_labelling(&mdp);
            local_product.modify_complete(task);
            //println!("modifying agent: {} task: {}", i, j);
            local_product.edit_labelling(task, &mdp);
            //local_product.modify_rewards(task);
            if graph_type > 0 {
                let g = local_product.generate_graph();
                let dot = format!("{}", Dot::new(&g));
                let mut file = File::create(format!("mod_product_mdp_{}.dot", j)).unwrap();
                file.write_all(&dot.as_bytes());
            }
            if verbose == 2 {
                println!("Modified product labelling");
                for label in local_product.labelling.iter() {
                    println!("{:?}", label);
                }
                for transition in local_product.transitions.iter() {
                    println!("{:?}", transition)
                }
            }

            team_input[team_counter] = TeamInput {
                agent: *i,
                task: *j,
                product: local_product,
                dead: task.dead.to_vec(),
                acc: task.acc.to_vec()
            };
            team_counter += 1;
        }
    }
    let mut team_mdp = TeamMDP::default();
    team_mdp.num_agents = num_agents;
    team_mdp.num_tasks = num_tasks;
    team_mdp.create_states(&team_input);
    if verbose == 2 {
        for state in team_mdp.states.iter() {
            println!("team state: ({},{},{},{})", state.state.s, state.state.q, state.agent, state.task);
        }
    }
    team_mdp.create_transitions_and_labelling(&team_input, &Rewards::POSITIVE);
    team_mdp.assign_task_rewards();
    team_mdp.modify_final_rewards(&team_input);
    if verbose == 2 {
        for transition in team_mdp.transitions.iter() {
            println!("state: ({},{},{},{}), rewards: {:?}", transition.from.state.s, transition.from.state.q,
            transition.from.agent, transition.from.task, transition.reward);
        }
    }
    if graph_type > 0 {
        let tg = team_mdp.generate_graph();
        let dot = format!("{}", Dot::new(&tg));
        let mut file = File::create("team_mdp.dot").unwrap();
        file.write_all(&dot.as_bytes());
        //let team_done_states: Vec<TeamState> = team_mdp.team_done_states();
        //team_mdp.all_paths(&tg, &team_mdp.initial, &team_done_states);
    }

    if run {
        let output = team_mdp.multi_obj_sched_synth(&target_parse, &epsilon, &Rewards::POSITIVE);
        match output {
            Some(x) => {
                println!("v: {:?}",x.v);
                let graph = team_mdp.dfs_merging(&x.mu, &x.v);
                let dot = format!("{}", Dot::new(&graph));
                let mut file = File::create("merged_sched.dot").unwrap();
                file.write_all(&dot.as_bytes());
            },
            _ => {println!("No output from scheduler synthesis!");}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
}


