mod model_checking;
use model_checking::decomp_team_mdp2::*;
use std::fs::File;
use std::io::Write;
use petgraph::{dot::Dot};
use std::time::{Duration, Instant};

use model_checking::helper_methods::{power_set, construct_labelling_vect};
use model_checking::mdp2::*;
use model_checking::dfa2::*;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::hash::Hash;

fn main() {
    let alphabet: Vec<&str> = vec!["start1", "l", "f1", "start2", "f2", "h", "s1", "down1", "r",
                                   "start3", "start4", "start5", "s2", "s3", "s4"];
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
    let mut transition_map = construct_dfa_transition(&ps[..], Some(&t_words[..]), 0, 1, None, &mut transition_map, None, None);
    let mut transition_map = construct_dfa_transition(&ps[..], None, 1, 3, Some((Some("h"), 2)), &mut transition_map, None, None);
    let mut not_words = transition_map.get(&(1,3)).unwrap().clone();
    let mut transition_map = construct_dfa_transition(&ps[..], None, 1, 2, Some((Some("down1"), 2)), &mut transition_map, Some(&not_words[..]), None);
    let mut not_words1 = transition_map.get(&(1,2)).unwrap().clone();
    let mut not_words2 = transition_map.get(&(1,3)).unwrap().clone();
    not_words2.append(&mut not_words1);
    let mut transition_map = construct_dfa_transition(&ps[..], None, 1, 1, Some((None, 2)), &mut transition_map, Some(&not_words2[..]), None);
    let not_words = transition_map.get(&(0,1)).unwrap().clone();
    let mut transition_map = construct_dfa_transition(&ps[..], None, 0, 0, Some((None, 2)), &mut transition_map, Some(&not_words[..]), None);
    let mut transition_map  = construct_dfa_transition(&ps[..], None, 2, 2, Some((None, 2)), &mut transition_map, None, None);
    let mut transition_map  = construct_dfa_transition(&ps[..], None, 3, 3, Some((None, 2)), &mut transition_map, None, None);

    // ----------------------------
    // Construct DFA2
    // ---------------------------
    let mut transition_map2: HashMap<(u32, u32), Vec<&HashSet<&str>>> = HashMap::new();
    let mut transition_map2 = construct_dfa_transition(&ps[..], None, 0, 1, Some((Some("start1"), 2)), &mut transition_map2, None, None);
    let mut not_words = transition_map2.get(&(0,1)).unwrap().clone();
    let mut transition_map2 = construct_dfa_transition(&ps[..], None, 0, 0, Some((None, 2)), &mut transition_map2, Some(&not_words[..]), None);
    let mut transition_map2 = construct_dfa_transition(&ps[..], None, 1,1, Some((Some("l"), 2)), &mut transition_map2, None, Some(("f2", 2)));
    let mut transition_map2 = construct_dfa_transition(&ps[..], None, 1,2, Some((Some("f2"), 2)), &mut transition_map2, None, Some(("l", 2)));
    let mut not_words = transition_map2.get(&(1,2)).unwrap().clone();
    let mut not_words2 = transition_map2.get(&(1,1)).unwrap().clone();
    not_words2.append(&mut not_words);
    let mut transition_map2 = construct_dfa_transition(&ps[..], None, 1,3,Some((None, 2)), &mut transition_map2, Some(&not_words2[..]), None);
    let mut transition_map2 = construct_dfa_transition(&ps[..], None, 2,2,Some((None,2)), &mut transition_map2, None, None);
    let mut transition_map2 = construct_dfa_transition(&ps[..], None, 3,3,Some((None,2)), &mut transition_map2, None, None);
    // ----------------------------
    // Construct DFA3
    // ----------------------------
    let mut transition_map3: HashMap<(u32, u32), Vec<&HashSet<&str>>> = HashMap::new();
    let mut transition_map3 = construct_dfa_transition(&ps[..], None, 0, 1, Some((Some("start2"), 2)), &mut transition_map3, None, None);
    let mut not_words = transition_map3.get(&(0,1)).unwrap().clone();
    let mut transition_map3 = construct_dfa_transition(&ps[..], None, 0, 0, Some((None, 2)), &mut transition_map3, Some(&not_words[..]), None);
    let mut transition_map3 = construct_dfa_transition(&ps[..], None, 1,1, Some((Some("l"), 2)), &mut transition_map3, None, Some(("f2", 2)));
    let mut transition_map3 = construct_dfa_transition(&ps[..], None, 1,2, Some((Some("f2"), 2)), &mut transition_map3, None, Some(("l", 2)));
    let mut not_words = transition_map3.get(&(1,2)).unwrap().clone();
    let mut not_words2 = transition_map3.get(&(1,1)).unwrap().clone();
    not_words2.append(&mut not_words);
    let mut transition_map3 = construct_dfa_transition(&ps[..], None, 1,3,Some((None, 2)), &mut transition_map3, Some(&not_words2[..]), None);
    let mut transition_map3 = construct_dfa_transition(&ps[..], None, 2,2,Some((None,2)), &mut transition_map3, None, None);
    let mut transition_map3 = construct_dfa_transition(&ps[..], None, 3,3,Some((None,2)), &mut transition_map3, None, None);
    // ----------------------------
    // Construct DFA4
    // ----------------------------
    let mut transition_map4: HashMap<(u32, u32), Vec<&HashSet<&str>>> = HashMap::new();
    let mut transition_map4 = construct_dfa_transition(&ps[..], None, 0, 1, Some((Some("start3"), 2)), &mut transition_map4, None, None);
    let mut not_words = transition_map4.get(&(0,1)).unwrap().clone();
    let mut transition_map4 = construct_dfa_transition(&ps[..], None, 0, 0, Some((None, 2)), &mut transition_map4, Some(&not_words[..]), None);
    let mut transition_map4 = construct_dfa_transition(&ps[..], None, 1,1, Some((Some("l"), 2)), &mut transition_map4, None, Some(("f2", 2)));
    let mut transition_map4 = construct_dfa_transition(&ps[..], None, 1,2, Some((Some("f2"), 2)), &mut transition_map4, None, Some(("l", 2)));
    let mut not_words = transition_map4.get(&(1,2)).unwrap().clone();
    let mut not_words2 = transition_map4.get(&(1,1)).unwrap().clone();
    not_words2.append(&mut not_words);
    let mut transition_map4 = construct_dfa_transition(&ps[..], None, 1,3,Some((None, 2)), &mut transition_map4, Some(&not_words2[..]), None);
    let mut transition_map4 = construct_dfa_transition(&ps[..], None, 2,2,Some((None,2)), &mut transition_map4, None, None);
    let mut transition_map4 = construct_dfa_transition(&ps[..], None, 3,3,Some((None,2)), &mut transition_map4, None, None);
    // ----------------------------
    // Construct DFA5
    // ----------------------------
    let mut transition_map5: HashMap<(u32, u32), Vec<&HashSet<&str>>> = HashMap::new();
    let mut transition_map5 = construct_dfa_transition(&ps[..], None, 0, 1, Some((Some("start4"), 2)), &mut transition_map5, None, None);
    let mut not_words = transition_map5.get(&(0,1)).unwrap().clone();
    let mut transition_map5 = construct_dfa_transition(&ps[..], None, 0, 0, Some((None, 2)), &mut transition_map5, Some(&not_words[..]), None);
    let mut transition_map5 = construct_dfa_transition(&ps[..], None, 1,1, Some((Some("l"), 2)), &mut transition_map5, None, Some(("f1", 2)));
    let mut transition_map5 = construct_dfa_transition(&ps[..], None, 1,2, Some((Some("f1"), 2)), &mut transition_map5, None, Some(("l", 2)));
    let mut not_words = transition_map5.get(&(1,2)).unwrap().clone();
    let mut not_words2 = transition_map5.get(&(1,1)).unwrap().clone();
    not_words2.append(&mut not_words);
    let mut transition_map5 = construct_dfa_transition(&ps[..], None, 1,3,Some((None, 2)), &mut transition_map5, Some(&not_words2[..]), None);
    let mut transition_map5 = construct_dfa_transition(&ps[..], None, 2,2,Some((None,2)), &mut transition_map5, None, None);
    let mut transition_map5 = construct_dfa_transition(&ps[..], None, 3,3,Some((None,2)), &mut transition_map5, None, None);
    // ----------------------------
    // Construct DFA6
    // ----------------------------
    let mut transition_map6: HashMap<(u32, u32), Vec<&HashSet<&str>>> = HashMap::new();
    let mut transition_map6 = construct_dfa_transition(&ps[..], None, 0, 1, Some((Some("start5"), 2)), &mut transition_map6, None, None);
    let mut not_words = transition_map6.get(&(0,1)).unwrap().clone();
    let mut transition_map6 = construct_dfa_transition(&ps[..], None, 0, 0, Some((None, 2)), &mut transition_map6, Some(&not_words[..]), None);
    let mut transition_map6 = construct_dfa_transition(&ps[..], None, 1,1, Some((Some("l"), 2)), &mut transition_map6, None, Some(("f1", 2)));
    let mut transition_map6 = construct_dfa_transition(&ps[..], None, 1,2, Some((Some("f1"), 2)), &mut transition_map6, None, Some(("l", 2)));
    let mut not_words = transition_map6.get(&(1,2)).unwrap().clone();
    let mut not_words2 = transition_map6.get(&(1,1)).unwrap().clone();
    not_words2.append(&mut not_words);
    let mut transition_map6 = construct_dfa_transition(&ps[..], None, 1,3,Some((None, 2)), &mut transition_map6, Some(&not_words2[..]), None);
    let mut transition_map6 = construct_dfa_transition(&ps[..], None, 2,2,Some((None,2)), &mut transition_map6, None, None);
    let mut transition_map6 = construct_dfa_transition(&ps[..], None, 3,3,Some((None,2)), &mut transition_map6, None, None);
    // ----------------------------
    // Construct DFA6
    // ----------------------------
    let h1: HashSet<&str> = HashSet::from_iter(vec!["s2"].iter().cloned());
    let h2: HashSet<&str> = HashSet::from_iter(vec!["s2","r"].iter().cloned());
    let s6_words: Vec<HashSet<&str>> = vec![h1,h2];
    let mut transition_map7: HashMap<(u32, u32), Vec<&HashSet<&str>>> = HashMap::new();
    let mut transition_map7 = construct_dfa_transition(&ps[..], Some(&s6_words[..]), 0, 1, None, &mut transition_map7, None, None);
    let mut transition_map7 = construct_dfa_transition(&ps[..], None, 1, 3, Some((Some("h"), 2)), &mut transition_map7, None, None);
    let mut not_words = transition_map7.get(&(1,3)).unwrap().clone();
    let mut transition_map7 = construct_dfa_transition(&ps[..], None, 1, 2, Some((Some("down1"), 2)), &mut transition_map7, Some(&not_words[..]), None);
    let mut not_words1 = transition_map7.get(&(1,2)).unwrap().clone();
    let mut not_words2 = transition_map7.get(&(1,3)).unwrap().clone();
    not_words2.append(&mut not_words1);
    let mut transition_map7 = construct_dfa_transition(&ps[..], None, 1, 1, Some((None, 2)), &mut transition_map7, Some(&not_words2[..]), None);
    let not_words = transition_map7.get(&(0,1)).unwrap().clone();
    let mut transition_map7 = construct_dfa_transition(&ps[..], None, 0, 0, Some((None, 2)), &mut transition_map7, Some(&not_words[..]), None);
    let mut transition_map7  = construct_dfa_transition(&ps[..], None, 2, 2, Some((None, 2)), &mut transition_map7, None, None);
    let mut transition_map7  = construct_dfa_transition(&ps[..], None, 3, 3, Some((None, 2)), &mut transition_map7, None, None);
    // ----------------------------
    // Construct DFA7
    // ----------------------------
    let h1: HashSet<&str> = HashSet::from_iter(vec!["s3"].iter().cloned());
    let h2: HashSet<&str> = HashSet::from_iter(vec!["s3","r"].iter().cloned());
    let s7_words: Vec<HashSet<&str>> = vec![h1,h2];
    let mut transition_map8: HashMap<(u32, u32), Vec<&HashSet<&str>>> = HashMap::new();
    let mut transition_map8 = construct_dfa_transition(&ps[..], Some(&s7_words[..]), 0, 1, None, &mut transition_map8, None, None);
    let mut transition_map8 = construct_dfa_transition(&ps[..], None, 1, 3, Some((Some("h"), 2)), &mut transition_map8, None, None);
    let mut not_words = transition_map8.get(&(1,3)).unwrap().clone();
    let mut transition_map8 = construct_dfa_transition(&ps[..], None, 1, 2, Some((Some("down1"), 2)), &mut transition_map8, Some(&not_words[..]), None);
    let mut not_words1 = transition_map8.get(&(1,2)).unwrap().clone();
    let mut not_words2 = transition_map8.get(&(1,3)).unwrap().clone();
    not_words2.append(&mut not_words1);
    let mut transition_map8 = construct_dfa_transition(&ps[..], None, 1, 1, Some((None, 2)), &mut transition_map8, Some(&not_words2[..]), None);
    let not_words = transition_map8.get(&(0,1)).unwrap().clone();
    let mut transition_map8 = construct_dfa_transition(&ps[..], None, 0, 0, Some((None, 2)), &mut transition_map8, Some(&not_words[..]), None);
    let mut transition_map8  = construct_dfa_transition(&ps[..], None, 2, 2, Some((None, 2)), &mut transition_map8, None, None);
    let mut transition_map8  = construct_dfa_transition(&ps[..], None, 3, 3, Some((None, 2)), &mut transition_map8, None, None);
    // ----------------------------
    // Construct DFA8
    // ----------------------------
    let h1: HashSet<&str> = HashSet::from_iter(vec!["s4"].iter().cloned());
    let h2: HashSet<&str> = HashSet::from_iter(vec!["s4","r"].iter().cloned());
    let s8_words: Vec<HashSet<&str>> = vec![h1,h2];
    let mut transition_map9: HashMap<(u32, u32), Vec<&HashSet<&str>>> = HashMap::new();
    let mut transition_map9 = construct_dfa_transition(&ps[..], Some(&s7_words[..]), 0, 1, None, &mut transition_map9, None, None);
    let mut transition_map9 = construct_dfa_transition(&ps[..], None, 1, 3, Some((Some("h"), 2)), &mut transition_map9, None, None);
    let mut not_words = transition_map9.get(&(1,3)).unwrap().clone();
    let mut transition_map9 = construct_dfa_transition(&ps[..], None, 1, 2, Some((Some("down1"), 2)), &mut transition_map9, Some(&not_words[..]), None);
    let mut not_words1 = transition_map9.get(&(1,2)).unwrap().clone();
    let mut not_words2 = transition_map9.get(&(1,3)).unwrap().clone();
    not_words2.append(&mut not_words1);
    let mut transition_map9 = construct_dfa_transition(&ps[..], None, 1, 1, Some((None, 2)), &mut transition_map9, Some(&not_words2[..]), None);
    let not_words = transition_map9.get(&(0,1)).unwrap().clone();
    let mut transition_map9 = construct_dfa_transition(&ps[..], None, 0, 0, Some((None, 2)), &mut transition_map9, Some(&not_words[..]), None);
    let mut transition_map9  = construct_dfa_transition(&ps[..], None, 2, 2, Some((None, 2)), &mut transition_map9, None, None);
    let mut transition_map9  = construct_dfa_transition(&ps[..], None, 3, 3, Some((None, 2)), &mut transition_map9, None, None);
    // -----------------------------
    // Construct Map
    // -----------------------------
    let grid_dim: (usize, usize) = (21, 11);
    let grid_state_space: HashMap<usize,(usize,usize)> = create_grid(grid_dim);
    let c_loc: (usize,usize) = (0,0);
    let c_loc2: (usize,usize) = (0,0);
    let act1: Vec<&str> = vec!["x", "l", "r"];
    let act2: Vec<&str> = vec!["n", "ne", "nw", "s", "se", "sw", "e", "w"];
    let v = vec![vec![], vec!["h"], vec!["r"], vec!["l"], vec!["h", "r"], vec!["h", "l"], vec!["start1"],
                 vec!["f1"], vec!["start2"], vec!["f2"], vec!["s1"], vec!["down1"], vec!["start1", "h"],
                 vec!["f1", "h"], vec!["start2", "h"], vec!["f2", "h"], vec!["s1", "h"], vec!["down1", "h"], vec!["start3"],
                 vec!["start3", "h"],vec!["start4"], vec!["start4", "h"], vec!["start5"], vec!["start5", "h"],
                 vec!["s2"], vec!["s2", "h"], vec!["s3"], vec!["s3", "h"], vec!["s4"], vec!["s4", "h"]
    ];
    let mdp_labels = construct_labelling_vect(&v[..]);
    // Different agents may have different obstacles
    let obstacles: [(usize, usize); 38] = [(2,2),(2,3),(2,4),(4,4),(5,4),(6,4),(7,2),
        (8,2),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(8,7),(7,7),(6,7),(11,10),(11,9),(11,8),(11,7),(11,6),(11,5),
        (12,5),(13,5),(14,5),(14,8),(14,2),(15,2),(16,2),(17,2),(17,3),(17,4),(17,5),(17,6),(17,7),(12,1),(12,3)];
    // hazards
    let hazards: [(usize, usize); 29] = [(3,0),(4,0),(4,1),(4,2),(3,7),(2,8),(2,9),(4,8),(4,9),(4,10),
        (5,0),(5,1),(5,2),(7,1),(8,1),(9,1),(6,5),(6,6),(7,6),(7,5),(15,3),(16,3),(15,4),(16,4),(16,5),
        (15,6),(16,6),(15,7),(16,7)];
    // Different agents may have different probabilities of moving in cardinal directions
    let movement_p: f64 = 0.90;
    //
    let mut all_act: Vec<&str> = act1.to_vec();
    all_act.extend(act2.iter().copied());
    let mut obj_points: HashMap<MDPLongState,Vec<&str>> = HashMap::new();
    obj_points.insert(MDPLongState{ m: "l", g: (0, 2) }, vec!["start1"]);
    obj_points.insert(MDPLongState{ m: "l", g: (3, 9) }, vec!["f1"]);
    obj_points.insert(MDPLongState{ m: "l", g: (1, 0) }, vec!["start2"]);
    obj_points.insert(MDPLongState{ m: "l", g: (18, 10) }, vec!["start3"]);
    obj_points.insert(MDPLongState{ m: "l", g: (20, 0) }, vec!["start4"]);
    obj_points.insert(MDPLongState{ m: "l", g: (14, 6) }, vec!["start5"]);
    obj_points.insert(MDPLongState{ m: "l", g: (10, 9) }, vec!["f2"]);
    obj_points.insert(MDPLongState{ m: "r", g: (2, 0) }, vec!["s1"]);
    obj_points.insert(MDPLongState{ m: "r", g: (9, 0) }, vec!["down1"]);
    obj_points.insert(MDPLongState{ m: "r", g: (3, 10) }, vec!["s2"]);
    obj_points.insert(MDPLongState{ m: "r", g: (14, 7) }, vec!["s3"]);
    obj_points.insert(MDPLongState{ m: "r", g: (8, 6) }, vec!["s4"]);
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
    let mut dfa2_transitions: Vec<DFA2Transitions> = vec![DFA2Transitions{ q: 0, w: vec![], q_prime: 0}; transition_map2.len()];
    for (i, ((q1, q2), v)) in transition_map2.iter().enumerate() {
        let filtered_labelling = v.into_iter().filter(|x| mdp_labels.iter().any(|y| y == **x)).collect::<Vec<_>>();
        dfa2_transitions[i] = DFA2Transitions { q: *q1, w: filtered_labelling, q_prime: *q2}
    }
    let mut dfa3_transitions: Vec<DFA2Transitions> = vec![DFA2Transitions{ q: 0, w: vec![], q_prime: 0}; transition_map3.len()];
    for (i, ((q1, q2), v)) in transition_map3.iter().enumerate() {
        let filtered_labelling = v.into_iter().filter(|x| mdp_labels.iter().any(|y| y == **x)).collect::<Vec<_>>();
        dfa3_transitions[i] = DFA2Transitions { q: *q1, w: filtered_labelling, q_prime: *q2}
    }
    let mut dfa4_transitions: Vec<DFA2Transitions> = vec![DFA2Transitions{ q: 0, w: vec![], q_prime: 0}; transition_map4.len()];
    for (i, ((q1, q2), v)) in transition_map4.iter().enumerate() {
        let filtered_labelling = v.into_iter().filter(|x| mdp_labels.iter().any(|y| y == **x)).collect::<Vec<_>>();
        dfa4_transitions[i] = DFA2Transitions { q: *q1, w: filtered_labelling, q_prime: *q2}
    }
    let mut dfa5_transitions: Vec<DFA2Transitions> = vec![DFA2Transitions{ q: 0, w: vec![], q_prime: 0}; transition_map5.len()];
    for (i, ((q1, q2), v)) in transition_map5.iter().enumerate() {
        let filtered_labelling = v.into_iter().filter(|x| mdp_labels.iter().any(|y| y == **x)).collect::<Vec<_>>();
        dfa5_transitions[i] = DFA2Transitions { q: *q1, w: filtered_labelling, q_prime: *q2}
    }
    let mut dfa6_transitions: Vec<DFA2Transitions> = vec![DFA2Transitions{ q: 0, w: vec![], q_prime: 0}; transition_map6.len()];
    for (i, ((q1, q2), v)) in transition_map6.iter().enumerate() {
        let filtered_labelling = v.into_iter().filter(|x| mdp_labels.iter().any(|y| y == **x)).collect::<Vec<_>>();
        dfa6_transitions[i] = DFA2Transitions { q: *q1, w: filtered_labelling, q_prime: *q2}
    }
    let mut dfa7_transitions: Vec<DFA2Transitions> = vec![DFA2Transitions{ q: 0, w: vec![], q_prime: 0}; transition_map7.len()];
    for (i, ((q1, q2), v)) in transition_map7.iter().enumerate() {
        let filtered_labelling = v.into_iter().filter(|x| mdp_labels.iter().any(|y| y == **x)).collect::<Vec<_>>();
        dfa7_transitions[i] = DFA2Transitions { q: *q1, w: filtered_labelling, q_prime: *q2}
    }
    let mut dfa8_transitions: Vec<DFA2Transitions> = vec![DFA2Transitions{ q: 0, w: vec![], q_prime: 0}; transition_map8.len()];
    for (i, ((q1, q2), v)) in transition_map8.iter().enumerate() {
        let filtered_labelling = v.into_iter().filter(|x| mdp_labels.iter().any(|y| y == **x)).collect::<Vec<_>>();
        dfa8_transitions[i] = DFA2Transitions { q: *q1, w: filtered_labelling, q_prime: *q2}
    }
    let mut dfa9_transitions: Vec<DFA2Transitions> = vec![DFA2Transitions{ q: 0, w: vec![], q_prime: 0}; transition_map9.len()];
    for (i, ((q1, q2), v)) in transition_map9.iter().enumerate() {
        let filtered_labelling = v.into_iter().filter(|x| mdp_labels.iter().any(|y| y == **x)).collect::<Vec<_>>();
        dfa9_transitions[i] = DFA2Transitions { q: *q1, w: filtered_labelling, q_prime: *q2}
    }

    let dfa_sensor: DFA2 = DFA2 {
        states: vec![0,1,2,3],
        sigma: &alphabet,
        initial: 0,
        delta: &dfa1_transitions,
        acc: vec![2],
        dead: vec![3]
    };
    let dfa_sensor2: DFA2 = DFA2 {
        states: vec![0,1,2,3],
        sigma: &alphabet,
        initial: 0,
        delta: &dfa7_transitions,
        acc: vec![2],
        dead: vec![3]
    };
    let dfa_sensor3: DFA2 = DFA2 {
        states: vec![0,1,2,3],
        sigma: &alphabet,
        initial: 0,
        delta: &dfa8_transitions,
        acc: vec![2],
        dead: vec![3]
    };
    let dfa_sensor4: DFA2 = DFA2 {
        states: vec![0,1,2,3],
        sigma: &alphabet,
        initial: 0,
        delta: &dfa9_transitions,
        acc: vec![2],
        dead: vec![3]
    };
    let dfa_convoy1: DFA2 = DFA2 {
        states: vec![0,1,2,3],
        sigma: &alphabet,
        initial: 0,
        delta: &dfa2_transitions,
        acc: vec![2],
        dead: vec![3]
    };
    let dfa_convoy2: DFA2 = DFA2 {
        states: vec![0,1,2,3],
        sigma: &alphabet,
        initial: 0,
        delta: &dfa3_transitions,
        acc: vec![2],
        dead: vec![3]
    };
    let dfa_convoy3: DFA2 = DFA2 {
        states: vec![0,1,2,3],
        sigma: &alphabet,
        initial: 0,
        delta: &dfa4_transitions,
        acc: vec![2],
        dead: vec![3]
    };
    let dfa_convoy4: DFA2 = DFA2 {
        states: vec![0,1,2,3],
        sigma: &alphabet,
        initial: 0,
        delta: &dfa5_transitions,
        acc: vec![2],
        dead: vec![3]
    };
    let dfa_convoy5: DFA2 = DFA2 {
        states: vec![0,1,2,3],
        sigma: &alphabet,
        initial: 0,
        delta: &dfa6_transitions,
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
    // local product 1: agent 1, task 1
    let completion_label: &str = "com";
    let completion_label_hash = HashSet::from_iter(vec![completion_label].iter().cloned());
    let initial_label: &str = "ini";
    let failure_label: &str = "fai";
    let success_label: &str = "suc";

    let mut r_p_m1_dfa1: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[0].states.len() * dfa_sensor.states.len());
    let mut t_m1_dfa1: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m1_dfa1: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m1_dfa1: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states11: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions11: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels11: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states11 = create_states(&mdps[0].states[..], &dfa_sensor.states[..]);
    let initial_prod_state11 = DFA2ModelCheckingPair { s: mdps[0].initial, q: dfa_sensor.initial };


    let (mut reach11, mut trans11, mut labels11) =
        create_local_product(&mdps[0], &dfa_sensor, initial_label, failure_label, success_label,
                         &completion_label_hash, &mdp_prod_states11[..],
                         &initial_prod_state11,& mut r_p_m1_dfa1, &mut t_m1_dfa1, &mut l_m1_dfa1,
                         &mut mod_l_m1_dfa1, &mut additional_states11, &mut additional_transitions11, &mut additional_labels11);
    let product_mdp11 = ProductMDP2 {
        states: reach11.to_owned(),
        initial: initial_prod_state11,
        transitions: trans11,
        labelling: labels11
    };
    // local product 2: agent 1, task 2
    let mut r_p_m1_dfa2: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[0].states.len() * dfa_sensor.states.len());
    let mut t_m1_dfa2: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m1_dfa2: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m1_dfa2: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states12: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions12: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels12: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states12 = create_states(&mdps[0].states[..], &dfa_convoy1.states[..]);
    let initial_prod_state12 = DFA2ModelCheckingPair { s: mdps[0].initial, q: dfa_convoy1.initial };


    let (mut reach12, mut trans12, mut labels12) =
        create_local_product(&mdps[0], &dfa_convoy1, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states12[..],
                             &initial_prod_state12,& mut r_p_m1_dfa2, &mut t_m1_dfa2, &mut l_m1_dfa2,
                             &mut mod_l_m1_dfa2, &mut additional_states12, &mut additional_transitions12, &mut additional_labels12);
    let product_mdp12 = ProductMDP2 {
        states: reach12.to_owned(),
        initial: initial_prod_state12,
        transitions: trans12,
        labelling: labels12
    };
    // local product 3: agent 1, task 3
    let mut r_p_m1_dfa3: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[0].states.len() * dfa_convoy2.states.len());
    let mut t_m1_dfa3: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m1_dfa3: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m1_dfa3: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states13: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions13: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels13: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states13 = create_states(&mdps[0].states[..], &dfa_convoy2.states[..]);
    let initial_prod_state13 = DFA2ModelCheckingPair { s: mdps[0].initial, q: dfa_convoy2.initial };


    let (mut reach13, mut trans13, mut labels13) =
        create_local_product(&mdps[0], &dfa_convoy2, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states13[..],
                             &initial_prod_state13,& mut r_p_m1_dfa3, &mut t_m1_dfa3, &mut l_m1_dfa3,
                             &mut mod_l_m1_dfa3, &mut additional_states13, &mut additional_transitions13, &mut additional_labels13);
    let product_mdp13 = ProductMDP2 {
        states: reach13.to_owned(),
        initial: initial_prod_state13,
        transitions: trans13,
        labelling: labels13
    };
    // local product 3: agent 1, task 4
    let mut r_p_m1_dfa4: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[0].states.len() * dfa_convoy3.states.len());
    let mut t_m1_dfa4: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m1_dfa4: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m1_dfa4: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states14: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions14: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels14: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states14 = create_states(&mdps[0].states[..], &dfa_convoy3.states[..]);
    let initial_prod_state14 = DFA2ModelCheckingPair { s: mdps[0].initial, q: dfa_convoy3.initial };


    let (mut reach14, mut trans14, mut labels14) =
        create_local_product(&mdps[0], &dfa_convoy3, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states14[..],
                             &initial_prod_state14,& mut r_p_m1_dfa4, &mut t_m1_dfa4, &mut l_m1_dfa4,
                             &mut mod_l_m1_dfa4, &mut additional_states14, &mut additional_transitions14, &mut additional_labels14);
    let product_mdp14 = ProductMDP2 {
        states: reach14.to_owned(),
        initial: initial_prod_state14,
        transitions: trans14,
        labelling: labels14
    };
    // local product 3: agent 1, task 5
    let mut r_p_m1_dfa5: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[0].states.len() * dfa_convoy4.states.len());
    let mut t_m1_dfa5: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m1_dfa5: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m1_dfa5: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states15: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions15: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels15: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states15 = create_states(&mdps[0].states[..], &dfa_convoy4.states[..]);
    let initial_prod_state15 = DFA2ModelCheckingPair { s: mdps[0].initial, q: dfa_convoy4.initial };


    let (mut reach15, mut trans15, mut labels15) =
        create_local_product(&mdps[0], &dfa_convoy4, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states15[..],
                             &initial_prod_state15,& mut r_p_m1_dfa5, &mut t_m1_dfa5, &mut l_m1_dfa5,
                             &mut mod_l_m1_dfa5, &mut additional_states15, &mut additional_transitions15, &mut additional_labels15);
    let product_mdp15 = ProductMDP2 {
        states: reach15.to_owned(),
        initial: initial_prod_state15,
        transitions: trans15,
        labelling: labels15
    };
    // local product 3: agent 1, task 6
    let mut r_p_m1_dfa6: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[0].states.len() * dfa_convoy5.states.len());
    let mut t_m1_dfa6: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m1_dfa6: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m1_dfa6: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states16: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions16: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels16: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states16 = create_states(&mdps[0].states[..], &dfa_convoy5.states[..]);
    let initial_prod_state16 = DFA2ModelCheckingPair { s: mdps[0].initial, q: dfa_convoy5.initial };


    let (mut reach16, mut trans16, mut labels16) =
        create_local_product(&mdps[0], &dfa_convoy5, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states16[..],
                             &initial_prod_state16,& mut r_p_m1_dfa6, &mut t_m1_dfa6, &mut l_m1_dfa6,
                             &mut mod_l_m1_dfa6, &mut additional_states16, &mut additional_transitions16, &mut additional_labels16);
    let product_mdp16 = ProductMDP2 {
        states: reach16.to_owned(),
        initial: initial_prod_state16,
        transitions: trans16,
        labelling: labels16
    };
    // local product: agent 1 task 7
    let mut r_p_m1_dfa7: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[0].states.len() * dfa_sensor2.states.len());
    let mut t_m1_dfa7: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m1_dfa7: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m1_dfa7: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states17: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions17: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels17: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states17 = create_states(&mdps[0].states[..], &dfa_sensor2.states[..]);
    let initial_prod_state17 = DFA2ModelCheckingPair { s: mdps[0].initial, q: dfa_sensor2.initial };

    let (mut reach17, mut trans17, mut labels17) =
        create_local_product(&mdps[0], &dfa_sensor2, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states17[..],
                             &initial_prod_state17,& mut r_p_m1_dfa7, &mut t_m1_dfa7, &mut l_m1_dfa7,
                             &mut mod_l_m1_dfa7, &mut additional_states17, &mut additional_transitions17, &mut additional_labels17);
    let product_mdp17 = ProductMDP2 {
        states: reach17.to_owned(),
        initial: initial_prod_state17,
        transitions: trans17,
        labelling: labels17
    };
    // local product: agent 1 task 8
    let mut r_p_m1_dfa8: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[0].states.len() * dfa_sensor3.states.len());
    let mut t_m1_dfa8: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m1_dfa8: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m1_dfa8: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states18: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions18: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels18: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states18 = create_states(&mdps[0].states[..], &dfa_sensor3.states[..]);
    let initial_prod_state18 = DFA2ModelCheckingPair { s: mdps[0].initial, q: dfa_sensor3.initial };

    let (mut reach18, mut trans18, mut labels18) =
        create_local_product(&mdps[0], &dfa_sensor3, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states18[..],
                             &initial_prod_state18,& mut r_p_m1_dfa8, &mut t_m1_dfa8, &mut l_m1_dfa8,
                             &mut mod_l_m1_dfa8, &mut additional_states18, &mut additional_transitions18, &mut additional_labels18);
    let product_mdp18 = ProductMDP2 {
        states: reach18.to_owned(),
        initial: initial_prod_state18,
        transitions: trans18,
        labelling: labels18
    };
    // local product: agent 1 task 9
    let mut r_p_m1_dfa9: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[0].states.len() * dfa_sensor4.states.len());
    let mut t_m1_dfa9: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m1_dfa9: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m1_dfa9: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states19: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions19: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels19: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states19 = create_states(&mdps[0].states[..], &dfa_sensor4.states[..]);
    let initial_prod_state19 = DFA2ModelCheckingPair { s: mdps[0].initial, q: dfa_sensor4.initial };

    let (mut reach19, mut trans19, mut labels19) =
        create_local_product(&mdps[0], &dfa_sensor4, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states19[..],
                             &initial_prod_state19,& mut r_p_m1_dfa9, &mut t_m1_dfa9, &mut l_m1_dfa9,
                             &mut mod_l_m1_dfa9, &mut additional_states19, &mut additional_transitions19, &mut additional_labels19);
    let product_mdp19 = ProductMDP2 {
        states: reach19.to_owned(),
        initial: initial_prod_state19,
        transitions: trans19,
        labelling: labels19
    };
    // local product 4: agent 2, task 1
    let mut r_p_m2_dfa1: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[0].states.len() * dfa_sensor.states.len());
    let mut t_m2_dfa1: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m2_dfa1: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m2_dfa1: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states21: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions21: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels21: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states21 = create_states(&mdps[1].states[..], &dfa_sensor.states[..]);
    let initial_prod_state21 = DFA2ModelCheckingPair { s: mdps[1].initial, q: dfa_sensor.initial };


    let (mut reach21, mut trans21, mut labels21) =
        create_local_product(&mdps[1], &dfa_sensor, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states21[..],
                             &initial_prod_state21,& mut r_p_m2_dfa1, &mut t_m2_dfa1, &mut l_m2_dfa1,
                             &mut mod_l_m2_dfa1, &mut additional_states21, &mut additional_transitions21, &mut additional_labels21);
    let product_mdp21 = ProductMDP2 {
        states: reach21.to_owned(),
        initial: initial_prod_state21,
        transitions: trans21,
        labelling: labels21
    };
    // local product 5,: agent 2, task 2
    let mut r_p_m2_dfa2: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[1].states.len() * dfa_convoy1.states.len());
    let mut t_m2_dfa2: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m2_dfa2: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m2_dfa2: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states22: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions22: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels22: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states22 = create_states(&mdps[1].states[..], &dfa_convoy1.states[..]);
    let initial_prod_state22 = DFA2ModelCheckingPair { s: mdps[1].initial, q: dfa_convoy1.initial };


    let (mut reach22, mut trans22, mut labels22) =
        create_local_product(&mdps[1], &dfa_convoy1, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states22[..],
                             &initial_prod_state22,& mut r_p_m2_dfa2, &mut t_m2_dfa2, &mut l_m2_dfa2,
                             &mut mod_l_m2_dfa2, &mut additional_states22, &mut additional_transitions22,
                             &mut additional_labels22);
    let product_mdp22 = ProductMDP2 {
        states: reach22.to_owned(),
        initial: initial_prod_state22,
        transitions: trans22,
        labelling: labels22
    };
    // local product 6,: agent 2, task 3
    let mut r_p_m2_dfa3: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[1].states.len() * dfa_convoy2.states.len());
    let mut t_m2_dfa3: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m2_dfa3: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m2_dfa3: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states23: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions23: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels23: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states23 = create_states(&mdps[1].states[..], &dfa_convoy2.states[..]);
    let initial_prod_state23 = DFA2ModelCheckingPair { s: mdps[1].initial, q: dfa_convoy2.initial };


    let (mut reach23, mut trans23, mut labels23) =
        create_local_product(&mdps[1], &dfa_convoy2, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states23[..],
                             &initial_prod_state23,& mut r_p_m2_dfa3, &mut t_m2_dfa3, &mut l_m2_dfa3,
                             &mut mod_l_m2_dfa3, &mut additional_states23, &mut additional_transitions23,
                             &mut additional_labels23);
    let product_mdp23 = ProductMDP2 {
        states: reach23.to_owned(),
        initial: initial_prod_state23,
        transitions: trans23,
        labelling: labels23
    };
    // local product 3: agent 2, task 4
    let mut r_p_m2_dfa4: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[1].states.len() * dfa_convoy3.states.len());
    let mut t_m2_dfa4: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m2_dfa4: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m2_dfa4: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states24: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions24: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels24: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states24 = create_states(&mdps[1].states[..], &dfa_convoy3.states[..]);
    let initial_prod_state24 = DFA2ModelCheckingPair { s: mdps[1].initial, q: dfa_convoy3.initial };


    let (mut reach24, mut trans24, mut labels24) =
        create_local_product(&mdps[1], &dfa_convoy3, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states24[..],
                             &initial_prod_state24,& mut r_p_m2_dfa4, &mut t_m2_dfa4, &mut l_m2_dfa4,
                             &mut mod_l_m2_dfa4, &mut additional_states24, &mut additional_transitions24, &mut additional_labels24);
    let product_mdp24 = ProductMDP2 {
        states: reach24.to_owned(),
        initial: initial_prod_state24,
        transitions: trans24,
        labelling: labels24
    };
    // local product 3: agent 2, task 5
    let mut r_p_m2_dfa5: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[1].states.len() * dfa_convoy4.states.len());
    let mut t_m2_dfa5: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m2_dfa5: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m2_dfa5: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states25: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions25: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels25: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states25 = create_states(&mdps[1].states[..], &dfa_convoy4.states[..]);
    let initial_prod_state25 = DFA2ModelCheckingPair { s: mdps[1].initial, q: dfa_convoy4.initial };


    let (mut reach25, mut trans25, mut labels25) =
        create_local_product(&mdps[1], &dfa_convoy4, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states25[..],
                             &initial_prod_state25,& mut r_p_m2_dfa5, &mut t_m2_dfa5, &mut l_m2_dfa5,
                             &mut mod_l_m2_dfa5, &mut additional_states25, &mut additional_transitions25, &mut additional_labels25);
    let product_mdp25 = ProductMDP2 {
        states: reach25.to_owned(),
        initial: initial_prod_state25,
        transitions: trans25,
        labelling: labels25
    };
    // local product 3: agent 2, task 6
    let mut r_p_m2_dfa6: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[1].states.len() * dfa_convoy5.states.len());
    let mut t_m2_dfa6: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m2_dfa6: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m2_dfa6: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states26: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions26: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels26: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states26 = create_states(&mdps[1].states[..], &dfa_convoy5.states[..]);
    let initial_prod_state26 = DFA2ModelCheckingPair { s: mdps[1].initial, q: dfa_convoy5.initial };


    let (mut reach26, mut trans26, mut labels26) =
        create_local_product(&mdps[1], &dfa_convoy5, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states26[..],
                             &initial_prod_state26,& mut r_p_m2_dfa6, &mut t_m2_dfa6, &mut l_m2_dfa6,
                             &mut mod_l_m2_dfa6, &mut additional_states26, &mut additional_transitions26, &mut additional_labels26);
    let product_mdp26 = ProductMDP2 {
        states: reach26.to_owned(),
        initial: initial_prod_state26,
        transitions: trans26,
        labelling: labels26
    };
    // local product: agent 2 task 7
    let mut r_p_m2_dfa7: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[1].states.len() * dfa_sensor2.states.len());
    let mut t_m2_dfa7: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m2_dfa7: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m2_dfa7: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states27: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions27: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels27: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states27 = create_states(&mdps[1].states[..], &dfa_sensor2.states[..]);
    let initial_prod_state27 = DFA2ModelCheckingPair { s: mdps[1].initial, q: dfa_sensor2.initial };

    let (mut reach27, mut trans27, mut labels27) =
        create_local_product(&mdps[1], &dfa_sensor2, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states27[..],
                             &initial_prod_state27,& mut r_p_m2_dfa7, &mut t_m2_dfa7, &mut l_m2_dfa7,
                             &mut mod_l_m2_dfa7, &mut additional_states27, &mut additional_transitions27, &mut additional_labels27);
    let product_mdp27 = ProductMDP2 {
        states: reach27.to_owned(),
        initial: initial_prod_state17,
        transitions: trans27,
        labelling: labels27
    };
    // local product: agent 1 task 8
    let mut r_p_m2_dfa8: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[1].states.len() * dfa_sensor3.states.len());
    let mut t_m2_dfa8: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m2_dfa8: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m2_dfa8: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states28: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions28: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels28: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states28 = create_states(&mdps[1].states[..], &dfa_sensor3.states[..]);
    let initial_prod_state28 = DFA2ModelCheckingPair { s: mdps[1].initial, q: dfa_sensor3.initial };

    let (mut reach28, mut trans28, mut labels28) =
        create_local_product(&mdps[1], &dfa_sensor3, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states28[..],
                             &initial_prod_state28,& mut r_p_m2_dfa8, &mut t_m2_dfa8, &mut l_m2_dfa8,
                             &mut mod_l_m2_dfa8, &mut additional_states28, &mut additional_transitions28, &mut additional_labels28);
    let product_mdp28 = ProductMDP2 {
        states: reach28.to_owned(),
        initial: initial_prod_state18,
        transitions: trans28,
        labelling: labels28
    };
    // local product: agent 1 task 9
    let mut r_p_m2_dfa9: Vec<&DFA2ModelCheckingPair> = Vec::with_capacity(mdps[1].states.len() * dfa_sensor4.states.len());
    let mut t_m2_dfa9: Vec<DFA2ProductTransition> = Vec::new();
    let mut l_m2_dfa9: Vec<DFA2ProductLabellingPair> = Vec::new();
    let mut mod_l_m2_dfa9: Vec<DFA2ModLabelPair> = Vec::new();
    let mut additional_states29: HashSet<DFA2ModelCheckingPair> = HashSet::new();
    let mut additional_transitions29: Vec<DFA2ProductTransition> = Vec::new();
    let mut additional_labels29: Vec<NonRefDFA2ProductLabellingPair> = Vec::new();
    let mdp_prod_states29 = create_states(&mdps[1].states[..], &dfa_sensor4.states[..]);
    let initial_prod_state29 = DFA2ModelCheckingPair { s: mdps[1].initial, q: dfa_sensor4.initial };

    let (mut reach29, mut trans29, mut labels29) =
        create_local_product(&mdps[1], &dfa_sensor4, initial_label, failure_label, success_label,
                             &completion_label_hash, &mdp_prod_states29[..],
                             &initial_prod_state29,& mut r_p_m2_dfa9, &mut t_m2_dfa9, &mut l_m2_dfa9,
                             &mut mod_l_m2_dfa9, &mut additional_states29, &mut additional_transitions29, &mut additional_labels29);
    let product_mdp29 = ProductMDP2 {
        states: reach29.to_owned(),
        initial: initial_prod_state29,
        transitions: trans29,
        labelling: labels29
    };

    let mut num_agents: usize = 2;
    let mut num_tasks: usize = 9;
    let mut team_input: Vec<TeamInput> = Vec::with_capacity(num_tasks * num_agents);
    team_input.push(TeamInput {
        agent: 0,
        task: 0,
        product: product_mdp11,
        dead: &dfa_sensor.dead,
        acc: &dfa_sensor.acc
    });
    team_input.push(TeamInput {
        agent: 0,
        task: 1,
        product: product_mdp12,
        dead: &dfa_convoy1.dead,
        acc: &dfa_convoy1.acc
    });
    team_input.push(TeamInput {
        agent: 0,
        task: 2,
        product: product_mdp13,
        dead: &dfa_convoy2.dead,
        acc: &dfa_convoy2.acc
    });
    team_input.push(TeamInput {
        agent: 0,
        task: 3,
        product: product_mdp14,
        dead: &dfa_convoy3.dead,
        acc: &dfa_convoy3.acc
    });
    team_input.push(TeamInput {
        agent: 0,
        task: 4,
        product: product_mdp15,
        dead: &dfa_convoy3.dead,
        acc: &dfa_convoy3.acc
    });
    team_input.push(TeamInput {
        agent: 0,
        task: 5,
        product: product_mdp16,
        dead: &dfa_convoy3.dead,
        acc: &dfa_convoy3.acc
    });
    team_input.push(TeamInput {
        agent: 0,
        task: 6,
        product: product_mdp17,
        dead: &dfa_sensor2.dead,
        acc: &dfa_sensor2.acc
    });
    team_input.push(TeamInput {
        agent: 0,
        task: 7,
        product: product_mdp18,
        dead: &dfa_sensor3.dead,
        acc: &dfa_sensor3.acc
    });
    team_input.push(TeamInput {
        agent: 0,
        task: 8,
        product: product_mdp19,
        dead: &dfa_sensor4.dead,
        acc: &dfa_sensor4.acc
    });
    team_input.push(TeamInput {
        agent: 1,
        task: 0,
        product: product_mdp21,
        dead: &dfa_sensor.dead,
        acc: &dfa_sensor.acc
    });
    team_input.push(TeamInput {
        agent: 1,
        task: 1,
        product: product_mdp22,
        dead: &dfa_convoy1.dead,
        acc: &dfa_convoy1.acc
    });
    team_input.push(TeamInput {
        agent: 1,
        task: 2,
        product: product_mdp23,
        dead: &dfa_convoy2.dead,
        acc: &dfa_convoy2.acc
    });
    team_input.push(TeamInput {
        agent: 1,
        task: 3,
        product: product_mdp24,
        dead: &dfa_convoy3.dead,
        acc: &dfa_convoy3.acc
    });
    team_input.push(TeamInput {
        agent: 1,
        task: 4,
        product: product_mdp25,
        dead: &dfa_convoy3.dead,
        acc: &dfa_convoy3.acc
    });
    team_input.push(TeamInput {
        agent: 1,
        task: 5,
        product: product_mdp26,
        dead: &dfa_convoy3.dead,
        acc: &dfa_convoy3.acc
    });
    team_input.push(TeamInput {
        agent: 1,
        task: 6,
        product: product_mdp27,
        dead: &dfa_sensor2.dead,
        acc: &dfa_sensor2.acc
    });
    team_input.push(TeamInput {
        agent: 1,
        task: 7,
        product: product_mdp28,
        dead: &dfa_sensor3.dead,
        acc: &dfa_sensor3.acc
    });
    team_input.push(TeamInput {
        agent: 1,
        task: 8,
        product: product_mdp29,
        dead: &dfa_sensor4.dead,
        acc: &dfa_sensor4.acc
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
    let target: Vec<f64> = vec![-900.0, -900.0, 600.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0, 800.0];
    let epsilon: f64 = 0.0001;
    let rewards: Rewards = Rewards::NEGATIVE;
    // ---------------------------------
    // Run
    // ---------------------------------
    let start = Instant::now();
    let output = team_mdp.multi_obj_sched_synth(&target, &epsilon, &rewards);
    let duration = start.elapsed();
    println!("Model checking time: {:?}", duration);
    let (state_count, transition_count) = team_mdp.statistics();
    println!("Model Statistics: |S|: {}, |P|: {}", state_count, transition_count);
    /*for m in output.mu.iter() {
        println!("output");
        let (_ordered_output, grid_ordered_output) =
            dfs_sched_debugger(&m[..], &team_mdp.states,  &team_mdp.transitions[..],
                               &team_mdp.initial, Some(&mdp_state_hashmap));
        /*for (s, a) in ordered_output.iter() {
            println!("state: ({},{},{},{}), a: {}", s.state.s, s.state.q, s.agent, s.task, a);
        }*/
        match grid_ordered_output {
            None => {}
            Some(x) => {
                for (s, a) in x.iter() {
                    println!("state: ({:?},{},{},{}), a: {}", s.s, s.q, s.agent, s.task, a);
                }
            }
        }
    }*/
    // ---------------------------------
    // DFS Output
    // ---------------------------------
    let graph = TeamMDP::dfs_merging(&team_mdp.initial, &output.mu[..], &team_mdp.states[..], &output.v,
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
            } else if *a == "ne" {
                let sprime = s_prime_movement(long_state, &movement, movement_p, &MovementDirection::NE, &state_coords);
                transition.s_prime = sprime;
                if long_state.m == "x" {
                    transition.rewards = 1.0;
                } else {
                    transition.rewards = 3.0;
                }
            } else if *a == "nw" {
                let sprime = s_prime_movement(long_state, &movement, movement_p, &MovementDirection::NW, &state_coords);
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
            } else if *a == "se" {
                let sprime = s_prime_movement(long_state, &movement, movement_p, &MovementDirection::SE, &state_coords);
                transition.s_prime = sprime;
                if long_state.m == "x" {
                    transition.rewards = 1.0;
                } else {
                    transition.rewards = 3.0;
                }
            } else if *a == "sw" {
                let sprime = s_prime_movement(long_state, &movement, movement_p, &MovementDirection::SW, &state_coords);
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
fn movement_coords<'a>(state: &'a u32, state_hash: &'a HashMap<u32, MDPLongState<'a>>, grid_dim: &'a (usize,usize),
                       obstacles: &'a [(usize, usize)]) -> Movement {
    let (x,y) = state_hash.get(state).unwrap().g;
    let movement = Movement {
        north: move_north(&x, &y, &grid_dim, obstacles),
        north_east: move_north_east(&x,&y,&grid_dim,obstacles),
        north_west: move_north_west(&x,&y,&grid_dim,obstacles),
        south: move_south(&x, &y, obstacles),
        south_east: move_south_east(&x,&y,&grid_dim,obstacles),
        south_west: move_south_west(&x,&y,obstacles),
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

fn move_north_east(x: &usize, y: &usize, grid_dim: &(usize,usize), obstacles: &[(usize,usize)]) -> (usize, usize) {
    // make a conscious design choice not to wall skate in the direction that does not fail the obstacle or the wall
    if *y < grid_dim.1 - 1 && *x < grid_dim.1 - 1 {
        if obstacles.iter().any(|(x1,y1)| (*x+1,*y + 1) == (*x1,*y1)) {
            (*x,*y)
        } else {
            (*x + 1,*y + 1)
        }
    } else {
        (*x,*y)
    }
}

fn move_north_west(x: &usize, y: &usize, grid_dim: &(usize,usize), obstacles: &[(usize,usize)]) -> (usize, usize) {
    // make a conscious design choice not to wall skate in the direction that does not fail the obstacle or the wall
    if *y < grid_dim.1 - 1 && *x > 0 {
        if obstacles.iter().any(|(x1,y1)| (*x-1,*y + 1) == (*x1,*y1)) {
            (*x,*y)
        } else {
            (*x - 1,*y + 1)
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

fn move_south_east(x: &usize, y: &usize, grid_dim: &(usize,usize), obstacles: &[(usize,usize)]) -> (usize, usize) {
    // make a conscious design choice not to wall skate in the direction that does not fail the obstacle or the wall
    if *y > 0 && *x < grid_dim.1 - 1 {
        if obstacles.iter().any(|(x1,y1)| (*x+1,*y-1) == (*x1,*y1)) {
            (*x,*y)
        } else {
            (*x + 1,*y - 1)
        }
    } else {
        (*x,*y)
    }
}

fn move_south_west(x: &usize, y: &usize, obstacles: &[(usize,usize)]) -> (usize, usize) {
    // make a conscious design choice not to wall skate in the direction that does not fail the obstacle or the wall
    if *y > 0 && *x > 0 {
        if obstacles.iter().any(|(x1,y1)| (*x-1,*y-1) == (*x1,*y1)) {
            (*x,*y)
        } else {
            (*x - 1,*y - 1)
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
    /*
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
    };*/

    /*let sprime: Vec<model_checking::mdp2::TransitionPair> = vec![
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
    sprime*/
    match direction {
        MovementDirection::NORTH => {
            vec![TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.north }).unwrap(),
                p: *p_dir,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.north_east }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.north_west }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            }]
        }
        MovementDirection::NE => {
            vec![TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.north_east }).unwrap(),
                p: *p_dir,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.north }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.east }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            }]
        }
        MovementDirection::NW => {
            vec![TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.north_west }).unwrap(),
                p: *p_dir,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.north }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.west }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            }]
        }
        MovementDirection::SOUTH => {
            vec![TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.south }).unwrap(),
                p: *p_dir,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.south_east }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.south_west }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            }]
        }
        MovementDirection::SE => {
            vec![TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.south_east }).unwrap(),
                p: *p_dir,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.east }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.south }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            }]
        }
        MovementDirection::SW => {
            vec![TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.south_west }).unwrap(),
                p: *p_dir,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.south }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.west }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            }]
        }
        MovementDirection::EAST => {
            vec![TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.east }).unwrap(),
                p: *p_dir,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.north_east }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.south_east }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            }]
        }
        MovementDirection::WEST => {
            vec![TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.west }).unwrap(),
                p: *p_dir,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.north_west }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            },TransitionPair{
                s: *state_coords.get(&MDPLongState{ m: state.m, g: movements.south_west }).unwrap(),
                p: (1f64-*p_dir)/2f64,
            }]
        }
    }
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
    NE,
    NW,
    SOUTH,
    SE,
    SW,
    EAST,
    WEST
}

#[derive(Debug, Clone)]
struct Movement {
    north: (usize,usize),
    north_east: (usize,usize),
    north_west: (usize, usize),
    south: (usize,usize),
    south_east: (usize,usize),
    south_west: (usize,usize),
    east: (usize,usize),
    west: (usize,usize),
}
