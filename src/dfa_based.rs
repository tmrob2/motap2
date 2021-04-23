//use itertools::{Itertools, enumerate};
use clap::clap_app;
use lib::{read_mdp_json, MDP, DFA, DFAProductMDP, read_dfa_json, DFAModelCheckingPair, TeamState, TeamInput, TeamMDP, NonNan, absolute_diff_vect};
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
           (@arg VERBOSE: -v --verbose [VERBOSITY] default_value("0") "Level of verbosity \
           0 - errors\
           1 - inputs
           2 - algorithm debugging
           3 - results
           ")
           (@arg TEST: -t --test "test some code")
           (@arg LABEL: -l --label "present the labelling of a product MDP")
           (@arg GRAPH: -g --graph [GRAPH_TYPE] default_value("0") "generates a product graph type: \
           1 - Product Graph
           2 - Modified Product Graph
           3 - Team MDP Graph
           4 - All of the above
           ")
            (@arg RUN: -r --run "run task allocation and planning in a team MDP")
            (@arg DEBUG_LOOP_MAX: --debug_max [DEBUG_MAX] default_value("0")
                "the number of loops allowed in a while loop in calculating the expected total returns")
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

    let debug_max_loops: u32 = match matches.subcommand() {
        ("motap", Some(f)) => {
            f.value_of("DEBUG_LOOP_MAX").unwrap().parse().unwrap()
        },
        (_,_) => 0
    };

    // If block used for testing certain inputs with the motap CLI
    if test {
        let vbar: Vec<f64> = vec![2.0, 1.0, 3.0];
        let mut v: Vec<_> = vbar.iter().map(|x| NonNan::new(*x).unwrap()).collect();
        v.sort();
        println!("min of vbar:{:?} is {:?}", &vbar, &v[0].inner());

        let a: Vec<f64> = vec![0.1, 2.3, 4.3];
        let b: Vec<f64> = vec![0.4, -1.0, 3.3];
        let c = absolute_diff_vect(&a, &b);
        println!("{:?}", c);
    }

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

    let dfas: Option<Vec<DFA>> = match read_dfa_json(dra_path_val) {
        Ok(u) => {
            if verbose == 1 {
                println!("{:?}", u);
            }
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
    let mut num_agents: usize = 0;
    let mut num_tasks: usize = 0;
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
    let mut team_input: Vec<TeamInput> = vec![TeamInput::default(); num_tasks * num_agents];
    let mut team_counter: usize = 0;
    for (i, mdp) in mdp_parse.iter() {
        for (j, task) in dfa_parse.iter() {
            let mut local_product: DFAProductMDP = DFAProductMDP::default();
            local_product.create_states(&mdp, task);
            local_product.create_transitions(&mdp, task);
            let mut g = local_product.generate_graph();
            let initially_reachable = local_product.reachable_from_initial(&g);
            let (prune_states_indices, prune_states) : (Vec<usize>, Vec<DFAModelCheckingPair>) = local_product.prune_candidates(&initially_reachable);
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
                product: local_product
            };
            team_counter += 1;
        }
    }
    let mut team_mdp = TeamMDP::default();
    team_mdp.num_agents = num_agents;
    team_mdp.num_tasks = num_tasks;
    team_mdp.create_states(&team_input);
    team_mdp.create_transitions_and_labelling(&team_input);
    team_mdp.assign_task_rewards();
    team_mdp.modify_final_transition();
    if verbose == 2 {
        for transition in team_mdp.transitions.iter().
            filter(|x| team_mdp.labelling.iter().
                any(|y| y.label.iter().any(|z| *z == "done")
                    && y.state.state == x.from.state) && x.a == "tau") {
            println!("mission complete transitions: {:?}", transition);
        }
    }
    if graph_type > 0 {
        let tg = team_mdp.generate_graph();
        let dot = format!("{}", Dot::new(&tg));
        let mut file = File::create("team_mdp.dot").unwrap();
        file.write_all(&dot.as_bytes());
    }

    if run {
        let w = vec![0.25,0.25,0.25,0.25];
        team_mdp.min_exp_tot(&team_input, &w, &epsilon, &debug_max_loops, &verbose);
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
}


