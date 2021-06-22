extern crate serde_json;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct MDP {
    pub states: Vec<u32>,
    pub initial: u32,
    pub transitions: Vec<Transition>,
    pub labelling: Vec<MDPLabellingPair>
}

#[derive(Debug, Deserialize, Clone)]
pub struct MDPLabellingPair {
    pub s: u32,
    pub w: String
}

#[derive(Debug, Deserialize, Clone)]
pub struct Transition {
    pub s: u32,
    pub a: String,
    pub s_prime: Vec<TransitionPair>,
    pub rewards: f64
}

#[derive(Debug, Deserialize, Clone)]
pub struct TransitionPair {
    pub s: u32,
    pub p: f64
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct MDPLongState<'a> {
    pub m: &'a str,
    pub g: (usize,usize)
}