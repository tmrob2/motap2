mod model_checking;
use clap::{clap_app, Values};
use model_checking::helper_methods::*;
use model_checking::decomp_team_mdp::*;
use std::fs::File;
use std::io::Write;
use petgraph::{dot::Dot};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use regex::Regex;

use model_checking::mdp::*;
use model_checking::dfa::*;
use model_checking::product_dfa_product_mdp::*;
use model_checking::product_dfa::*;
use std::collections::hash_map::RandomState;
use model_checking::gurobi_lp::witness;
use crate::model_checking::team_mdp::*;

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
            (@arg STAT: -s [STATISTICS] default_value("0") "Statistics of the model")
            (@arg TERM: --term [TERMINATE] default_value("10000") "Termination value of value iteration")
            (@arg TEAM_TYPE: --team [TEAM_TYPE] default_value("0") "The type of team MDP to be created, \
            0 is the decomposed team MDP linear size in both agents and tasks, 1 is a partially decomposed MDP linear in the number of agents")
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

    let stats: bool = match matches.subcommand() {
        ("motap", Some(f)) => {
            match f.occurrences_of("STAT") {
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

    let termination: u32 = match matches.subcommand() {
        ("motap", Some(f)) => {
            f.value_of("TERM").unwrap().parse().unwrap()
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

    let mdps: Option<Vec<MDP>> = match read_mdp_json(format!("examples/{}",mdp_path_val)) {
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

    let dfas: Option<Vec<DFA>> = match read_dfa_json(format!("examples/{}",dra_path_val)) {
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

    let team_type: u8 = match matches.subcommand() {
        ("motap", Some(f)) => {
            f.value_of("TEAM_TYPE").unwrap().parse().unwrap()
        },
        (_,_) => 0
    };

    //let safety_present: bool = false;
    let mut dfa_parse: Vec<(usize, DFA)> = Vec::new();
    let mut mdp_parse: Vec<(usize, MDP)> = Vec::new();
    let mut target_parse: Vec<f64> = Vec::new();
    let mut num_agents: usize = 0;
    let mut num_tasks: usize = 0;
    match dfas {
        None => {println!("There was an error reading the DFAs from {}", dra_path_val)}
        Some(x) => {
            num_tasks = x.len();
            for (i, mut aut) in x.into_iter().enumerate() {
                if team_type == 1u8 {
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
                }
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



    if team_type == 1 {
        let mut product_dfa: ProductDFA = ProductDFA::default(); // Product DFA has a lifetime of a and lives for the entirety of Team
        product_dfa.sigma = dfa_parse[0].1.sigma.to_vec();
        let mut transitions: Vec<ProductDFATransition> = Vec::new();
        for (j, task) in dfa_parse.iter() {
            println!("Constructing automata: {}", j);
            println!("Constructing states:");
            let state_start = Instant::now();
            let new_states = ProductDFA::create_states(&task, &mut product_dfa.states);
            let duration = state_start.elapsed();
            println!("{:?}", duration);
            product_dfa.initial.push(task.initial);
            println!("Constructing Transitions");
            let trans_start = Instant::now();
            let new_transition: Vec<ProductDFATransition> = product_dfa.create_transitions(&task, &transitions[..], &new_states);
            let duration = trans_start.elapsed();
            println!("{:?}", duration);
            product_dfa.states = new_states;
            transitions = new_transition;
        }
        product_dfa.delta = transitions;

        if graph_type == 1 {
            let g = product_dfa.create_automaton_graph();
            let dot = format!("{}", Dot::new(&g));
            let mut file = File::create("diagnostics/product_dfa_automaton.dot").unwrap();
            file.write_all(&dot.as_bytes());
        }
        let mut team_inputs: Vec<TeamInputs> = vec![TeamInputs{ states: vec![], initial: StatePair { s: 0, q: vec![] }, transitions: vec![], labelling: vec![], agent: 0 }; mdp_parse.len()];
        for i in 0..mdp_parse.len() {
            println!("Constructing {}", i);
            let states = model_checking::product_dfa_product_mdp::create_states(&mdp_parse[i].1, &product_dfa);
            let (transitions, labels) = model_checking::product_dfa_product_mdp::create_transitions(&states[..], &mdp_parse[i].1, &product_dfa, &dfa_parse, &verbose);
            let initial = StatePair{ s: mdp_parse[i].1.initial, q: product_dfa.initial.to_vec()};
            let reachable_states =
                model_checking::product_dfa_product_mdp::reachable_from_initial(&states[..], &transitions[..], &initial);
            let transitions = model_checking::product_dfa_product_mdp::prune_transitions(&reachable_states[..], &transitions[..]);
            let (mod_trans_incompl_cand, mod_trans_compl_cand) = model_checking::product_dfa_product_mdp::identify_mod_transitions(&transitions[..], &labels[..]);
            let (mod_trans_incompl,rm_transitions_incompl, mod_states_incompl, mod_labels_incompl) =
                model_checking::product_dfa_product_mdp::modify_incomplete_tasks(&mod_trans_incompl_cand[..], &dfa_parse[..], &labels[..]);
            let (mod_trans_compl,rm_transitions_compl, mod_states_compl, mod_labels_compl) =
                model_checking::product_dfa_product_mdp::modify_complete_tasks(&mod_trans_compl_cand[..], &dfa_parse[..], &labels[..]);
            let reachable_states = model_checking::product_dfa_product_mdp::append_states(&reachable_states[..], &mod_states_incompl[..]);
            let reachable_states = model_checking::product_dfa_product_mdp::append_states(&reachable_states[..], &mod_states_compl);
            let transitions = model_checking::product_dfa_product_mdp::remove_transitions(&transitions[..], &rm_transitions_incompl[..]);
            let transitions = model_checking::product_dfa_product_mdp::remove_transitions(&transitions[..], &rm_transitions_compl[..]);
            let transitions = model_checking::product_dfa_product_mdp::append_transitions(&transitions[..], &mod_trans_incompl[..]);
            let transitions = model_checking::product_dfa_product_mdp::append_transitions(&transitions[..], &mod_trans_compl[..]);
            let labels = model_checking::product_dfa_product_mdp::append_labels(&labels[..], &mod_labels_incompl[..]);
            let labels = model_checking::product_dfa_product_mdp::append_labels(&labels, &mod_labels_compl[..]);
            team_inputs[i].states = reachable_states;
            team_inputs[i].initial = initial;
            team_inputs[i].transitions = transitions;
            team_inputs[i].labelling = labels;
            team_inputs[i].agent = i;
            /*for l in team_inputs[i].labelling.iter() {
                println!("s: {},{:?},{}, l:{:?}", l.s.s, l.s.q, i, l.w);
            }*/
            if graph_type == 3 {
                let g = model_checking::product_dfa_product_mdp::create_graph(&team_inputs[i].states[..], &team_inputs[i].transitions[..]);
                let dot = format!("{}", Dot::new(&g));
                let mut file = File::create(format!("diagnostics/local_product_{}.dot", i)).unwrap();
                file.write_all(&dot.as_bytes());
            }
        }
        let team_initial_default: StatePair = StatePair{ s: 0, q: vec![] };
        let mut dfa_prod_team_mdp: DFAProductTeamMDP = DFAProductTeamMDP::default(&team_initial_default);
        dfa_prod_team_mdp.num_agents = mdp_parse.len();
        dfa_prod_team_mdp.num_tasks = dfa_parse.len();
        let (team_states, team_initial_state) = model_checking::team_mdp::create_states(&team_inputs[..]);
        dfa_prod_team_mdp.states = team_states;
        dfa_prod_team_mdp.initial = team_initial_state;
        let team_transitions = model_checking::team_mdp::create_transitions(&dfa_prod_team_mdp.states[..], &team_inputs[..], &Rewards::NEGATIVE, dfa_prod_team_mdp.num_tasks, dfa_prod_team_mdp.num_agents);
        dfa_prod_team_mdp.transitions = team_transitions;
        /*
        for s in dfa_prod_team_mdp.states.iter() {
            println!("s:{},{:?},{}", s.s, s.q, s.agent);
        }
        for t in dfa_prod_team_mdp.transitions.iter().filter(|x| *x.from.s >= 999) {
            println!("state: {},{:?},{}, rewards: {:?}", t.from.s, t.from.q, t.from.agent, t.reward);
        }*/
        if graph_type == 4 {
            let g = dfa_prod_team_mdp.generate_team_graph();
            let dot = format!("{}", Dot::new(&g));
            let mut file = File::create("diagnostics/product_dfa_product_team_mdp.dot").unwrap();
            file.write_all(&dot.as_bytes());
        }
        //let w = vec![0.25, 0.25, 0.25, 0.25];
        let team_index_mappings = dfa_prod_team_mdp.team_ij_index_mapping();
        if stats {
            dfa_prod_team_mdp.statistics();
        }
        if run {
            /*let safe_ret = dfa_prod_team_mdp.exp_tot_cost(&w[..], &epsilon, &team_index_mappings, &Rewards::NEGATIVE);
            match safe_ret {
                None => {}
                Some((mu, r)) => {
                    /*for (s,a) in mu.iter() {
                        println!("s: {:?}, a:{}", s, a);
                    }*/
                    println!("r: {:?}", r);
                    let team_initial = &dfa_prod_team_mdp.initial;
                    let ordered_mu = model_checking::team_mdp::dfs_sched_debugger(&mu, &dfa_prod_team_mdp.states[..], &dfa_prod_team_mdp.transitions[..], team_initial);
                    for (s,a) in ordered_mu.iter() {
                        println!("s: ({},{:?},{}), a: {}", s.s, s.q, s.agent, a);
                    }
                }
            }*/
            let start = Instant::now();
            let output = dfa_prod_team_mdp.multi_obj_sched_synth(&target_parse, &epsilon,  &team_index_mappings, &Rewards::NEGATIVE);
            let duration = start.elapsed();
            println!("Model checking time: {:?}", duration);
        }
        return;
    }
    println!("target: {:?}", target_parse);
    let mut team_input: Vec<TeamInput> = vec![TeamInput::default(); num_tasks * num_agents];
    let mut team_counter: usize = 0;
    for (i, mdp) in mdp_parse.iter() {
        for (j, task) in dfa_parse.iter() {
            let mut local_product: DFAProductMDP = DFAProductMDP::default();
            local_product.initial = DFAModelCheckingPair{ s: mdp.initial, q: task.initial };
            local_product.create_states(&mdp, task);
            local_product.create_transitions(&mdp, task);
            let mut g = local_product.generate_graph();
            let initially_reachable = local_product.reachable_from_initial(&g);
            let (prune_states_indices, prune_states) : (Vec<usize>, Vec<DFAModelCheckingPair>) =
                local_product.prune_candidates(&initially_reachable);
            if graph_type > 0 {
                local_product.prune_graph(&mut g, &prune_states_indices);
                let dot = format!("{}", Dot::new(&g));
                let mut file = File::create(format!("diagnostics/product_mdp{}_{}.dot",i,j)).unwrap();
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
                let mut file = File::create(format!("diagnostics/mod_product_mdp_{}.dot", j)).unwrap();
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
    team_mdp.create_transitions_and_labelling(&team_input, &Rewards::NEGATIVE);
    team_mdp.assign_task_rewards();
    team_mdp.modify_final_rewards(&team_input);
    if verbose == 2 {
        for transition in team_mdp.transitions.iter() {
            println!("state: ({},{},{},{}) -> {:?}", transition.from.state.s, transition.from.state.q,
                transition.from.agent, transition.from.task,
            transition.reward);
        }
        for l in team_mdp.labelling.iter() {
            println!("s: ({},{},{},{}), l: {:?}", l.state.state.s, l.state.state.q, l.state.agent, l.state.task, l.label)
        }
    }
    if graph_type > 0 {
        let tg = team_mdp.generate_graph();
        let dot = format!("{}", Dot::new(&tg));
        let mut file = File::create("diagnostics/team_mdp.dot").unwrap();
        file.write_all(&dot.as_bytes());
        //let team_done_states: Vec<TeamState> = team_mdp.team_done_states();
        //team_mdp.all_paths(&tg, &team_mdp.initial, &team_done_states);
    }

    if stats {
        println!("Model Stats");
        let(s,t) = team_mdp.statistics();
        println!("states: {}, transitions: {}", s, t);
    }

    if test {
        //let w: Vec<f64> = vec![1.0 / (team_mdp.num_agents + team_mdp.num_tasks) as f64; team_mdp.num_agents + team_mdp.num_tasks];
        let w = vec![0.0, 0.0, 1.0, 0.0];
        let mut v: Vec<Duration> = Vec::with_capacity(100);
        let (transition_map, state_to_trans_cardinality,
            state_to_trans_start_fin_map, sprime_state_index_map)
            = model_checking::decomp_team_mdp::vector_index_mapping_non_iter(&team_mdp.states[..],
                                                                    &team_mdp.transitions[..], &team_mdp.num_agents, &team_mdp.num_tasks);
        let team_init_index = team_mdp.states.iter().position(|x| *x == team_mdp.initial).unwrap();
        for i in 0..1 {
            let start = Instant::now();
            let safe_ret = opt_exp_tot_cost_non_iter(&w, &epsilon, &team_mdp.states[..], &team_mdp.transitions[..],
                             &Rewards::NEGATIVE, &team_mdp.num_agents, &team_mdp.num_tasks, &transition_map, &state_to_trans_cardinality,
                                                     &state_to_trans_start_fin_map, &sprime_state_index_map[..], &team_init_index);
            let duration = start.elapsed();
            println!("duration: {:?}", duration);
            v.push(duration);
            match safe_ret {
                None => {}
                Some((mu, r)) => {println!("r: {:?}", r)}
            }
        }
    }

    if run {
        let rewards: Rewards = Rewards::NEGATIVE;
        /**/
        let mut v: Vec<Duration> = Vec::with_capacity(100);
        /*for i in 0..100 {
            let start = Instant::now();
            let output = team_mdp.multi_obj_sched_synth(&target_parse, &epsilon, &rewards, &true);
            //let output = team_mdp.multi_obj_sched_synth_non_iter(&target_parse, &epsilon, &rewards);
            let duration = start.elapsed();
            v.push(duration);
        }*/
        let sum_duration = v.iter().fold(0.0, |acc, &sum| acc + sum.as_secs_f64());
        println!("ave duration:{}", sum_duration/ 100.0);
        let start = Instant::now();
        let output = team_mdp.multi_obj_sched_synth(&target_parse, &epsilon, &rewards, &true);
        //let output = team_mdp.multi_obj_sched_synth_non_iter(&target_parse, &epsilon, &rewards);
        let duration = start.elapsed();
        println!("Model checking time: {:?}", duration);
        println!("v: {:?}", v);
        let graph = dfs_merging(&team_mdp.initial, &output.mu,
                                &team_mdp.states[..], &output.v, &team_mdp.transitions[..], None);
        let dot = format!("{}", Dot::new(&graph));
        let mut file = File::create("diagnostics/merged_sched.dot").unwrap();
        file.write_all(&dot.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
}


