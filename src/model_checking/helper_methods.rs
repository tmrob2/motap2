use std::num::ParseIntError;
use std::io::BufReader;
use std::path::Path;
use std::error::Error;
use std::fs::File;
extern crate serde_json;
use serde::Deserialize;
use super::mdp;
use super::dfa;

use mdp::*;
use dfa::*;
use std::collections::HashSet;
use std::iter::FromIterator;

pub fn absolute_diff_vect(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let a_b: Vec<_> = a.iter().zip(b.into_iter()).collect();
    let mut c: Vec<f64> = vec![0.0; a.len()];
    for (i, (val1,val2)) in a_b.iter().enumerate() {
        c[i] = (**val1 - **val2).abs();
    }
    c
}

pub fn parse_int(s: &str) -> std::result::Result<u32, ParseIntError> {
    s.parse::<u32>()
}

pub fn parse_str_vect(s: &str) -> serde_json::Result<Vec<u32>> {
    let u: Vec<u32> = serde_json::from_str(s)?;
    Ok(u)
}

pub fn read_mdp_json<'a, P: AsRef<Path>>(path:P) -> std::result::Result<Vec<MDP>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}

/*pub fn read_dra_json<'a, P: AsRef<Path>>(path:P) -> std::result::Result<Vec<DRA>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}
 */

pub fn read_dfa_json<'a, P: AsRef<Path>>(path: P) -> std::result::Result<Vec<DFA>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}

pub fn read_target<'a, P: AsRef<Path>>(path: P) -> std::result::Result<Target, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let u = serde_json::from_reader(reader)?;
    Ok(u)
}

pub fn parse_language(sigma: &Vec<String>, w: &Vec<String>) -> Option<Vec<String>>{
    let mut valid_words: Option<Vec<String>> = None;
    if w.len() == 1 {
        for word in w.iter() {
            if word.chars().next().unwrap() == '!' {
                let expression = &word[1..];
                let a: Vec<String> = expression.split("&").map(|x| x.to_string()).collect();
                let sigma_hash: HashSet<String> = HashSet::from_iter(sigma.iter().cloned());
                let a_hash: HashSet<String> = HashSet::from_iter(a.into_iter());
                let sigma_diff: Vec<_> = sigma_hash.difference(&a_hash).collect();
                valid_words = Some(sigma_diff.into_iter().map(|x| x.to_string()).collect());
                //valid_words = Some(sigma_diff);
            } else if word == "true" {
                valid_words = Some(sigma.to_vec());
            }
        }
    }
    valid_words
}

#[derive(PartialOrd, PartialEq, Debug, Clone, Copy)]
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
}