use criterion::{black_box, criterion_group, criterion_main, Criterion};
use criterion::measurement::WallTime;
mod model_checking;
use model_checking::helper_methods::*;
use model_checking::decomp_team_mdp::*;
use model_checking::dfa::*;
use model_checking::mdp::*;

use std::fs::File;
use std::io::Write;
use petgraph::{dot::Dot};
use std::time::{Duration, Instant};

enum Agent {
    TWO,
    THREE,
    FOUR,
    FIVE,
    SIX,
    SEVEN,
    TEN,
    TWENTY,
    FIFTY
}

enum Tasks {
    TWO,
    THREE,
    FOUR,
    FIVE,
    SIX,
    SEVEN,
    EIGHT,
    NINE,
    TEN,
    TWENTY,
    FIFTY
}

fn setup(agent: Agent, tasks: Tasks) -> TeamMDP {
    let mut dfa_parse: Vec<(usize, DFA)> = Vec::new();
    let mut mdp_parse: Vec<(usize, MDP)> = Vec::new();
    let mut num_agents: usize = 0;
    let mut num_tasks: usize = 0;

    let mdp_path_val = match agent {
        Agent::TWO => {"examples/agents2.json"}
        Agent::THREE => {"examples/agents3.json"}
        Agent::FOUR => {"examples/agents4.json"}
        Agent::FIVE => {"examples/agents5.json"}
        Agent::SIX => {"examples/agents6.json"}
        Agent::SEVEN => {"examples/agents7.json"}
        Agent::TEN => {""}
        Agent::TWENTY => {""}
        Agent::FIFTY => {""}
    };

    let dfa_path_val = match tasks {
        Tasks::TWO => {"examples/dfa_2task.json"}
        Tasks::THREE => {"examples/dfa_3task.json"}
        Tasks::FOUR => {"examples/dfa_4task.json"}
        Tasks::FIVE => {"examples/dfa_5task.json"}
        Tasks::SIX => {"examples/dfa_6task.json"}
        Tasks::SEVEN => {"examples/dfa_7task.json"}
        Tasks::EIGHT => {"examples/dfa_8task.json"}
        Tasks::NINE => {"examples/dfa_9task.json"}
        Tasks::TEN => {"examples/dfa_10task.json"}
        Tasks::TWENTY => {""}
        Tasks::FIFTY => {""}
    };

    let mdps: Option<Vec<MDP>> = match read_mdp_json(mdp_path_val) {
        Ok(u) => {
            Some(u)
        },
        Err(e) => {println!("Error: {}", e); None}
    };

    let dfas: Option<Vec<DFA>> = match read_dfa_json(dfa_path_val) {
        Ok(u) => {
            Some(u)
        },
        Err(e) => {println!("Error: {}", e); None}
    };

    match mdps {
        Some(z) => {
            num_agents = z.len();
            for (j, mdp) in z.into_iter().enumerate() {
                mdp_parse.push((j, mdp));
            }
        },
        None => {panic!("There was an error reading the mdp from {}", mdp_path_val) }
    }

    match dfas {
        None => {panic!("There was an error reading the DFAs from {}", dfa_path_val)}
        Some(x) => {
            num_tasks = x.len();
            for (i, aut) in x.into_iter().enumerate() {
                dfa_parse.push((i, aut));
            }
        }
    }

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
            local_product.prune_states_transitions(&prune_states_indices, &prune_states);
            local_product.create_labelling(&mdp);
            local_product.modify_complete(task);
            //println!("modifying agent: {} task: {}", i, j);
            local_product.edit_labelling(task, &mdp);
            //local_product.modify_rewards(task);
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
    team_mdp.create_transitions_and_labelling(&team_input, &Rewards::POSITIVE);
    team_mdp.assign_task_rewards();
    team_mdp.modify_final_rewards(&team_input);
    team_mdp

}

pub fn sched_synth_benchmark(c: &mut Criterion) -> &mut Criterion<WallTime> {
    let target: Vec<f64> = vec![2800.0, 2800.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 80.0, 90.0];
    let regularisation: f64 = 0.1;
    let epsilon: f64 = 0.00001;
    let a: Agent = Agent::TWO;
    let t: Tasks = Tasks::NINE;
    let mut team_mdp = setup(a, t);

    let team_index_mappings = team_mdp.team_ij_index_mapping();

    c.bench_function("scheduler synthesis",|b|
        b.iter(|| team_mdp.multi_obj_sched_synth(&target, &epsilon, &Rewards::POSITIVE, &regularisation, &team_index_mappings)))
}

criterion_group!(benches, sched_synth_benchmark);
criterion_main!(benches);