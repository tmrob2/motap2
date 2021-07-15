//use std::io::BufReader;
//use std::path::Path;
//use std::error::Error;
//use std::fs::File;
//extern crate serde_json;
//use serde::{Deserialize};
//use super::mdp;
//use super::dfa;

//use mdp::*;
//use dfa::*;
//use std::collections::{HashSet};
//use std::iter::FromIterator;
//use std::hash::Hash;
use ndarray::{arr1, Array1};
//use crate::model_checking::mdp2::MDP2;

/*pub fn absolute_diff_vect(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let a_b: Vec<_> = a.iter().zip(b.into_iter()).collect();
    let mut c: Vec<f64> = vec![0.0; a.len()];
    for (i, (val1,val2)) in a_b.iter().enumerate() {
        c[i] = (**val1 - **val2).abs();
    }
    c
}

 */

#[allow(dead_code)]
pub enum Rewards {
    NEGATIVE,
    POSITIVE
}

pub fn opt_absolute_diff_vect(a: &[f64], b: &[f64]) -> Array1<f64> {
    let c: Array1<f64> = arr1(b) - &arr1(a);
    c
}

/*pub fn read_mdp_json<'a, P: AsRef<Path>>(path:P) -> std::result::Result<Vec<MDP>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}*/

/*pub fn read_target<'a, P: AsRef<Path>>(path: P) -> std::result::Result<Target, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}*/

/*pub fn power_set<T: Clone + Eq + Hash>(a: &[T])-> Vec<HashSet<T>> {
    a.iter().fold(vec![HashSet::new()], |mut p, x| {
        let i = p.clone().into_iter()
            .map(|mut s| {s.insert(x.clone()); s});
        p.extend(i); p})
}*/

/*#[derive(PartialOrd, PartialEq, Debug, Clone, Copy)]
pub struct NonNan(f64);

impl NonNan {
    pub fn new(val: f64) -> Option<NonNan> {
        if val.is_nan() {
            None
        } else {
            Some(NonNan(val))
        }
    }

    pub fn inner(self) -> f64 {
        self.0
    }
}

impl Eq for NonNan {}

impl Ord for NonNan {
    fn cmp(&self, other: &NonNan) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Deserialize)]
pub struct Target {
    pub target: Vec<f64>
}*/

/*pub fn construct_labelling_vect<'a>(v: &'a[Vec<&'a str>]) -> Vec<HashSet<&'a str>> {
    let mut v_new: Vec<_> = Vec::new();
    for w in v.iter() {
        let h = construct_hash_from_vect(&w[..]);
        v_new.push(h)
    }
    v_new
}

pub fn construct_hash_from_vect<'a>(v: &[&'a str]) -> HashSet<&'a str> {
    let h: HashSet<&'a str> = HashSet::from_iter(v.iter().cloned());
    h
}*/