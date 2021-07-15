use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct MDP2<'a> {
    pub states: Vec<u32>,
    pub initial: u32,
    pub transitions: Vec<Transition>,
    pub labelling: Vec<MDPLabellingPair<'a>>
}

#[derive(Debug, Clone)]
pub struct MDPLabellingPair<'a> {
    pub s: u32,
    pub w: Vec<&'a HashSet<&'a str>>
}

#[derive(Debug, Clone)]
pub struct Transition {
    pub s: u32,
    pub a: String,
    pub s_prime: Vec<TransitionPair>,
    pub rewards: f64
}

#[derive(Debug, Clone)]
pub struct TransitionPair {
    pub s: u32,
    pub p: f64
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct MDPLongState<'a> {
    pub m: &'a str,
    pub g: (usize,usize)
}