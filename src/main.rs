//use itertools::{Itertools, enumerate};
use clap::clap_app;
use lib::{read_mdp_json, read_dra_json, MDP, DRA, ProductMDP, ProductDRA};
use std::fs::File;
use std::io::Write;
use petgraph::{algo::kosaraju_scc,dot::Dot};
use regex::{Regex};

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
           (@arg GRAPH: -g --graph [GRAPH_TYPE] default_value("0") "generates a product graph type: \
           1 - Product Graph
           2 - Modified Product Graph
           3 - Team MDP Graph
           4 - All of the above
           ")
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

    let graph_type: u32 = match matches.subcommand() {
        ("motap", Some(f)) => {
            f.value_of("GRAPH").unwrap().parse().unwrap()
        },
        (_,_) => 0
    };

    if test {
        let test_state: &str = "(1,[0,0])";
        let re: Regex = Regex::new(r"^\((?P<input1>[0-9]+),(?P<input2>[\[\]0-9]+\))").unwrap();
        let input1 = re.captures(test_state).and_then(|x|{
            x.name("input1").map(|y| y.as_str())
        });
        let input2 = re.captures(test_state).and_then(|x|{
            x.name("input2").map(|y| y.as_str())
        });
        println!("{:?}", input1);
        println!("{:?}", input2);
        return
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

    let mdp: Option<MDP> = match read_mdp_json(mdp_path_val) {
        Ok(u) => {
            if verbose == 1 {
                println!("{:?}", u);
            }
            Some(u)
        },
        Err(e) => {println!("Error: {}", e); None}
    };

    let dras: Option<Vec<DRA>> = match read_dra_json(dra_path_val) {
        Ok(u) => {
            if verbose == 1 {
                println!("{:?}", u);
            }
            Some(u)
        },
        Err(e) => {println!("Error: {}", e); None}
    };

    let mut product_mdp: ProductMDP = ProductMDP::default();
    let mut dra_product: ProductDRA = ProductDRA::default();
    match mdp {
        Some(m) => {
            match dras {
                Some(x) => {
                    for aut in x.iter() {
                        dra_product.create_states(aut);
                        if verbose == 2 {
                            println!("DRA Product: {:?}", dra_product);
                        }
                    }
                    dra_product.sigma = x[0].sigma.clone();
                    dra_product.create_transitions(&x);
                    dra_product.set_initial(&x);
                    let task_count = ProductDRA::task_count(&x);
                    if verbose == 3 {
                        println!("DRA: {:?}", dra_product);
                    }
                    product_mdp.create_states(&m, &dra_product);
                    product_mdp.create_transitions(&m, &dra_product, &task_count, &verbose);
                    product_mdp.set_initial(&m, &dra_product);
                    product_mdp.prune(&verbose);
                },
                None => {println!("There was an error reading DRAs"); return}
            }
        },
        None => {println!("There was an error reading the mdp"); return }
    }

    if verbose == 3 {
        for s in product_mdp.states.iter(){
            println!("Product MDP states: {:?}", s);
        }
        println!("Transitions");
        for t in product_mdp.transitions.iter() {
            println!("{:?}", t);
        }
        for l in product_mdp.labelling.iter() {
            println!("{:?}", l);
        }
    }

    if graph_type > 0 {
        println!("Processing Graph");
        let g = product_mdp.generate_graph();
        println!("graph: ");
        println!("{:?}", g);
        let scc: Vec<Vec<petgraph::prelude::NodeIndex>> = kosaraju_scc(&g);
        let dot = format!("{}", Dot::new(&g));
        let mut count: u32 = 0;
        for ni_vect in scc.iter() {
            for ni in ni_vect.iter() {
                let i = ni;
                println!("{}:{:?}", count, g[*i]);
            }
            count += 1;
        }
        let mut file = File::create("product_MDP.dot").unwrap();
        file.write_all(&dot.as_bytes());
        product_mdp.find_mecs(&g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    /// Test that an empty hashmap produces a lenght of zero
    fn cart_product() {
        let it = (0..2).cartesian_product((3..5));
        itertools::assert_equal(it, vec![(0,3),(0,4),(1,3),(1,4)])
    }

    #[test]
    pub fn convert_string_node_to_state() {
        let test_state: &str = "(1,[0, 0])";
        let re: Regex = Regex::new(r"^\((?P<input1>[0-9]+),(?P<input2>\[[0-9+],\s[0-9+]\])\)").unwrap();
        let input1 = re.captures(test_state).and_then(|x|{
            x.name("input1").map(|y| y.as_str())
        });
        let input2 = re.captures(test_state).and_then(|x|{
            x.name("input2").map(|y| y.as_str())
        });
        assert_eq!(input1, Some("1"));
        assert_eq!(input2, Some("[0, 0]"));
    }
}


