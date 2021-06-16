#[derive(Debug, Deserialize)]
pub struct DRA {
    pub states: Vec<u32>,
    pub sigma: Vec<String>,
    pub safety: bool,
    pub initial: u32,
    pub delta: Vec<DRATransitions>,
    pub acc: Vec<AcceptanceSet>,
}

impl DRAMod {
    pub fn generate_graph(&mut self) -> Graph<String, String> {
        let mut graph: Graph<String, String> = Graph::new();

        for state in self.states.iter() {
            graph.add_node(format!("{}", state));
        }
        //println!("{:?}", graph.raw_nodes());
        for transition in self.delta.iter() {
            let origin_index = graph.node_indices().
                find(|x| graph[*x] == format!("{}", transition.q)).unwrap();
            let destination_index = match graph.node_indices().find(|x| graph[*x] == format!("{}", transition.q_prime)){
                None => {panic!("state: {} not found!", transition.q_prime)}
                Some(x) => {x}
            };
            graph.add_edge(origin_index, destination_index, format!("{:?}", transition.w.to_vec()));
        }
        graph
    }

    pub fn accepting_states(&self, g: &Graph<String, String>) -> Vec<NodeIndex> {
        let mut acc_indices: Vec<NodeIndex> = Vec::new();
        for state in self.acc.iter() {
            for k in state.k.iter() {
                let k_index = g.node_indices().
                    find(|x| g[*x] == format!("{}", k)).unwrap();
                acc_indices.push(k_index);
            }
        }
        acc_indices
    }

    pub fn reachable_from_states(&self, g: &Graph<String, String>, acc: &Vec<NodeIndex>) -> Vec<(u32, bool)> {
        let reachable: Vec<bool> = vec![true; self.states.len()];
        let mut reachable_states: Vec<(u32, bool)> = self.states.iter().cloned().zip(reachable.into_iter()).collect();
        for (state, truth) in reachable_states.iter_mut() {
            let from_node_index: NodeIndex = g.node_indices().find(|x| g[*x] == format!("{}", state)).unwrap();
            for k in acc.iter() {
                *truth = has_path_connecting(g, from_node_index, *k, None);
                //println!("Path from {} to {} is {}", g[from_node_index], g[*k], truth);
            }
        }
        reachable_states
    }
}

#[derive(Debug)]
pub struct ProductTransition {
    pub sq: ModelCheckingPair,
    pub a: String,
    pub sq_prime: Vec<ModelCheckingPair>
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ProductLabellingPair {
    pub sq: ModelCheckingPair,
    pub w: Vec<String>
}

#[derive(Debug, Deserialize, Clone)]
pub struct AcceptanceSet {
    pub l: Vec<u32>,
    pub k: Vec<u32>
}

#[derive(Debug, Deserialize, Clone)]
pub struct DRATransitions {
    q: u32,
    w: Vec<String>,
    q_prime: u32,
}

#[derive(Debug, Clone)]
pub struct DRAMod {
    pub states: Vec<u32>,
    pub sigma: Vec<String>,
    pub safety: bool,
    pub initial: u32,
    pub delta: Vec<DRATransitions>,
    pub acc: Vec<AcceptanceSet>,
    pub dead: Vec<u32>
}

pub struct ProductMDP {
    pub states: Vec<ModelCheckingPair>,
    pub initial: ModelCheckingPair,
    pub transitions: Vec<ProductTransition>,
    pub labelling: Vec<ProductLabellingPair>
}

impl ProductMDP {

    /// Create states should only be used for the initial product creation
    pub fn create_states(&mut self, mdp: &MDP, dra: &ProductDRA) {
        self.states = Vec::new();
        let cp= mdp.states.clone().into_iter().cartesian_product(dra.states.clone().into_iter());
        for (s_m, q_a) in cp.into_iter() {
            self.states.push(ModelCheckingPair{ s: s_m, q: q_a.clone() });
        }
    }

    pub fn create_transitions(&mut self, mdp: &MDP, dra: &ProductDRA, verbose: &u32, dras: &Vec<DRAMod>, safety_present: &bool) {
        for state_from in self.states.iter() {
            if *verbose == 2 {
                println!("state from: {:?}", state_from);
            }
            // For each state we want to filter the mdp transitions to that state
            for t in mdp.transitions.iter().filter(|x| x.s == state_from.s) {
                // We then want to know, given some state in M, what state does it transition to?
                // there are exactly m DRAs currently added to the product (in the initial case this is 1)
                // and we want to know that given DRA (task j), in the loop signature above, we enumerate over q
                // for each task in the vector of DRAs minus the safety task, we want to determine what happens
                // when we transition from s -> a -> s' such that the trace will be L(s')
                // but the MDP wil have a non specific labelling, such as initiate(k)

                // Observe the state and determine which task is active, if the MDP state is 0 and the automata state is final
                // then any of the remaining tasks can be chosen. We must always be thinking that the MDP is projecting its states
                // onto some automata representing a task, but the MDP cannot switch from one task to another, it must continue with
                // some active task to completion

                // determine the active task
                let mut inactive_tasks: Vec<usize> = Vec::new();
                let mut active_task: bool = false;
                let mut active_tasks: Vec<usize> = Vec::new();
                let task_elements: &[u32] = if *safety_present {
                    state_from.q.split_last().unwrap().1
                } else {
                    &state_from.q[..]
                };
                let new_task = task_elements.iter().enumerate().
                    all(|(i, x)| dras[i].initial == *x
                        || dras[i].dead.iter().any(|y| y == x)
                        || dras[i].acc.iter().any(|y| y.k.iter().any(|z|z == x)));
                if new_task {
                    if task_elements.iter().enumerate().all(|(i, x)| dras[i].dead.iter().any(|y| y == x)
                        || dras[i].acc.iter().any(|y| y.k.iter().any(|z|z == x))) {
                        // every task has finished
                        if *verbose == 2 {
                            println!("Every task has finished at state: {:?}", task_elements);
                        }
                        for k in 0..task_elements.len(){
                            inactive_tasks.push(k + 1);
                        }
                    } else {
                        if *verbose == 2 {
                            println!("There are tasks remaining: {:?}", task_elements);
                        }
                        for (i,q) in task_elements.iter().enumerate() {
                            if dras[i].initial == *q {
                                if *verbose == 2 {
                                    println!("initiating task: {}", i + 1);
                                }
                                inactive_tasks.push(i + 1);
                            }
                        }
                    }
                } else {
                    active_task = true;
                    if *verbose == 2{
                        println!("There is an  active task: {:?}", task_elements);
                    }
                    for (i, q) in task_elements.iter().enumerate() {
                        if dras[i].initial != *q
                            && dras[i].dead.iter().all(|y| y != q)
                            && dras[i].acc.iter().all(|y| y.k.iter().all(|z|z != q)) {
                            active_tasks.push(i+1)
                        }
                    }
                }
                let mut task_queue: Vec<usize> = Vec::new();
                if active_task {
                    for task in active_tasks.into_iter() {
                        task_queue.push(task);
                    }
                } else {
                    for task in inactive_tasks.into_iter() {
                        task_queue.push(task);
                    }
                }
                if *verbose == 2 {
                    println!("task queue: {:?}", task_queue);
                }
                for task in task_queue.iter() {
                    let mut p_transition = ProductTransition {
                        sq: ModelCheckingPair { s: state_from.s, q: state_from.q.clone() },
                        a: format!("{}{}", t.a.to_string(), task).to_string(),
                        sq_prime: Vec::new()
                    };
                    for mdp_sprime in t.s_prime.iter() {
                        // label may be empty, if empty, then we should treat it as no label
                        // get the label specific to a task if it meets the appropriate pattern
                        let mut q_prime: Vec<u32> = Vec::new();
                        for lab in mdp.labelling.iter().filter(|l| l.s == mdp_sprime.s) {
                            // task specific labels will be required to have the form ...(k) where the label is specific to
                            // task k. Obviously the DRA will not accept this, and we will require that (k) be replaced by
                            // the enumerated task automaton e.g. (1) for automaton 0.
                            let mut ltemp: String = lab.w.clone();
                            if lab.w.contains("(k)") {
                                let substring = format!("({})", task);
                                ltemp = lab.w.to_string().replace(&*"(k)".to_string(), &*substring);
                                if *verbose == 2 {
                                    println!("replaced {}, with {}", lab.w, ltemp);
                                }
                            }
                            // find if there is a DRA transition. If no transition exists for the labelling is this an error? Let's print some
                            // of the results to find out.

                            for dra_transition in dra.delta.iter().
                                filter(|qq| qq.q == state_from.q && qq.w.contains(&ltemp)) {
                                q_prime = dra_transition.q_prime.clone();
                            }
                            if q_prime.is_empty() {
                                if *verbose == 2 {
                                    println!("No transition was found for (q: {:?}, w: {})", state_from.q, ltemp);
                                    println!("Available transitions are: ");
                                    for dra_transition in dra.delta.iter().
                                        filter(|qq| qq.q == state_from.q) {
                                        println!("{:?}", dra_transition);
                                    }
                                }
                            }
                            self.labelling.push(ProductLabellingPair { sq: ModelCheckingPair { s: mdp_sprime.s, q: q_prime.clone() }, w: vec![ltemp] });
                        }
                        p_transition.sq_prime.push(ModelCheckingPair{ s: mdp_sprime.s, q: q_prime });
                    }
                    self.transitions.push(p_transition);
                }
            }
        }
    }

    /// Adds in the task k labelling update i.e. {init, success, fail}
    ///
    /// ** Note
    /// This function is a new addition and will always be somewhat redundant
    /// because we can do this in the create transitions or create states function can minimise
    /// the required loops over states by one.
    pub fn update_prod_labelling(&mut self, dras: &Vec<DRAMod>, safety_present: &bool) {
        for state in self.states.iter() {
            let mut state_label_found: bool = false;
            let mut state_label: Vec<String> = Vec::new();
            let mut label_index: Option<usize> = None;
            for (i, l) in self.labelling.iter().enumerate().filter(|(_ii,x)| x.sq == *state) {
                state_label_found = true;
                state_label = l.w.to_vec();
                label_index = Some(i);
            }
            let q_bar: &[u32] = if *safety_present {
                state.q.split_last().unwrap().1
            } else {
                &state.q[..]
            };

            for (i, q) in q_bar.iter().enumerate() {
                if dras[i].dead.iter().any(|y| y == q) {
                    // label as fail (i)
                    state_label.push(format!("fail({})", i))

                } else if dras[i].acc.iter().any(|y| y.k.iter().any(|z|z == q)) {
                    // label as success (i)
                    state_label.push(format!("success({})",i))
                }
            }

            if state_label_found {
                self.labelling[label_index.unwrap()].w = state_label;
            } else {
                self.labelling.push(ProductLabellingPair{ sq: ModelCheckingPair { s: state.s, q: state.q.to_vec() }, w: state_label });
            }
        }
    }

    pub fn set_initial(&mut self, mdp: &MDP, dra: &ProductDRA) {
        self.initial = ModelCheckingPair { s: mdp.initial, q: dra.initial.clone() };
    }

    /// Traversing from the initial state, prune unreachable states.
    pub fn prune(&mut self, verbose: &u32) {
        let mut queue: VecDeque<ModelCheckingPair> = VecDeque::new();
        let mut visited: Vec<bool> = vec![false; self.states.len()];
        let mut visited_states: Vec<ModelCheckingPair> = Vec::new();
        let mut visited_transitions: Vec<ProductTransition> = Vec::new();
        let mut visited_labels: Vec<ProductLabellingPair> = Vec::new();
        queue.push_front(ModelCheckingPair{ s: self.initial.s, q: self.initial.q.clone() });
        let index_visited: usize = self.states.iter().
            position(|x| x.s == self.initial.s && x.q == self.initial.q.clone()).unwrap();
        visited[index_visited] = true;
        while !queue.is_empty() {
            if *verbose == 2 {
                println!("queue: {:?}", queue);
            }
            let next_state = queue.pop_front().unwrap();
            visited_states.push(ModelCheckingPair{ s: next_state.s, q: next_state.q.clone() });
            for label in self.labelling.iter().
                filter(|x| x.sq.s == next_state.s && x.sq.q == next_state.q) {
                visited_labels.push(ProductLabellingPair {
                    sq: ModelCheckingPair { s: label.sq.s, q: label.sq.q.clone() },
                    w: label.w.clone()
                });
            }
            for transition in self.transitions.iter().
                filter(|x| x.sq.s == next_state.s && x.sq.q == next_state.q) {
                for transition_to in transition.sq_prime.iter(){
                    let sqprime_index: usize = self.states.iter().
                        position(|xx| xx.s == transition_to.s && xx.q == transition_to.q).unwrap();
                    if !visited[sqprime_index] {
                        queue.push_front(ModelCheckingPair{ s: transition_to.s, q: transition_to.q.clone() });
                        visited[sqprime_index] = true;
                    }
                }
                visited_transitions.push(ProductTransition{
                    sq: ModelCheckingPair { s: transition.sq.s, q: transition.sq.q.clone() },
                    a: transition.a.to_string(),
                    sq_prime: transition.sq_prime.clone()
                });

            }
        }
        self.states = visited_states;
        self.transitions = visited_transitions;
        self.labelling = (HashSet::<_>::from_iter(visited_labels.iter().cloned())).into_iter().collect();

    }

    /// When taking the product of an already generated product MDP with a new DRA, which generates a new Product MDP set of states.
    pub fn add_states(&mut self, dra: &DRA) {
        let mut new_states: Vec<ModelCheckingPair> = Vec::new();
        let cp = self.states.clone().into_iter().cartesian_product(dra.states.clone().into_iter());
        for (old, new) in cp.into_iter() {
            let mut new_q: Vec<u32> = old.q.clone();
            new_q.push(new);
            new_states.push(ModelCheckingPair{ s: old.s, q: new_q })
        }
    }

    pub fn add_initial(&mut self, dra: &DRA) {
        let mut qnew = self.initial.q.clone();
        qnew.push(dra.initial);
        self.initial = ModelCheckingPair{s: self.initial.s, q: qnew};
    }

    pub fn generate_graph(&mut self) -> Graph<String, String> {
        let mut graph: Graph<String, String> = Graph::new();

        for state in self.states.iter() {
            graph.add_node(format!("({},{:?})", state.s, state.q));
        }
        //println!("{:?}", graph.raw_nodes());
        for transition in self.transitions.iter() {
            let origin_index = graph.node_indices().
                find(|x| graph[*x] == format!("({},{:?})", transition.sq.s, transition.sq.q)).unwrap();
            for sqprime in transition.sq_prime.iter() {
                let destination_index = match graph.node_indices().find(|x| graph[*x] == format!("({},{:?})", sqprime.s, sqprime.q)){
                    None => {panic!("state: ({},{:?}) not found!", sqprime.s, sqprime.q)}
                    Some(x) => {x}
                };
                graph.add_edge(origin_index, destination_index, transition.a.clone());
            }
        }
        graph
    }

    /// Function to identify the trivial MEC decomposition of the input graph.
    /// This is essentially any self loop.
    pub fn find_trivial_mecs(&self, mecs: &Vec<Vec<ModelCheckingPair>>) -> HashSet<ModelCheckingPair> {
        // Given a list of MECs we are essentially looking for a state wihtin the MEC which has a
        // self loop and greater than or equal to 2 actions.
        let mut triv_mecs: HashSet<ModelCheckingPair> = HashSet::new();
        for mec in mecs.iter() {
            for state in mec.iter() {
                for transition in self.transitions.iter().
                    filter(|x| x.sq_prime.len() == 1 && x.sq_prime.contains(&x.sq) && x.sq == *state) {
                    //println!("trivial transition: {:?}", transition);
                    triv_mecs.insert(ModelCheckingPair{ s: transition.sq.s, q: transition.sq.q.to_vec() });
                }
            }
        }
        triv_mecs
    }

    /// Function to identify the non-trivial MEC decomposition of the graph
    pub fn find_mecs(&self, g: &Graph<String, String>) -> Vec<Vec<ModelCheckingPair>> {
        let mut g_copy: Graph<String, String> = g.clone();
        //let mut test_counter: u32 = 0;
        // we also need to establish a correspondence between the states of the MDP and the nodes of the graph
        let mut state_actions: HashMap<ModelCheckingPair, HashSet<String>> = HashMap::new();
        for s in self.states.iter() {
            let mut action_vect: HashSet<String> = HashSet::new();
            for transition in self.transitions.iter().filter(|x| x.sq == *s) {
                action_vect.insert(transition.a.to_string());
            }
            state_actions.insert(s.clone(), action_vect);
        }

        let mut mec: Vec<Vec<ModelCheckingPair>> = Vec::new();
        let mut mec_new: Vec<Vec<ModelCheckingPair>> = vec![self.states.clone()];
        let mut removed_hist: Vec<ModelCheckingPair> = Vec::new();

        while mec != mec_new {
            mec = mec_new.drain(0..).collect();
            let mut remove: HashSet<RemoveSCCState> = HashSet::new();
            let scc_ni: Vec<Vec<petgraph::prelude::NodeIndex>> = kosaraju_scc(&g_copy);
            let mut scc_state_ind: Vec<Vec<ModelCheckingPair>> = Vec::new();
            for scc in scc_ni.iter() {
                let mut inner: Vec<ModelCheckingPair> = Vec::new();
                for ni in scc.iter() {
                    let str_state = &g_copy[*ni];
                    match self.convert_string_node_to_state(str_state) {
                        Some(x) => inner.push(x),
                        None => {println!("Error on finding node with state: {}", str_state); return mec_new }
                    }
                }
                scc_state_ind.push(inner);
            }
            for (i, t_k) in scc_state_ind.iter().enumerate() {
                for (j, state) in t_k.iter().enumerate() {
                    let mut action_s: HashSet<String> = state_actions.get(&state).unwrap().clone();
                    //println!("actions: {:?}", action_s);
                    for transition in self.transitions.iter().filter(|x| x.sq == *state) {
                        for sprime in transition.sq_prime.iter() {
                            //println!("s': {:?}", sprime);
                            if !t_k.iter().any(|x| x == sprime) {
                                // is there is a state such that taking action 'a' leaves the SCC with P > 0 then
                                // we remove this action from the SCC and it is not contained in the EC
                                action_s.remove(&*transition.a);
                            }
                        }
                    }
                    if action_s.is_empty() {
                        // remove the state
                        remove.insert(RemoveSCCState {
                            state: ModelCheckingPair { s: state.s, q: state.q.clone() },
                            scc_ind: i,
                            state_ind: j
                        });
                    }
                    state_actions.insert(state.clone(), action_s);
                }
                //println!("T({}), SCC: {:?}, R: {:?}",i, t_k, remove);
            }

            while !remove.is_empty() {
                //println!("remove: {:?} ; remove hist: {:?}", remove, removed_hist);
                let mut remove_more: HashSet<RemoveSCCState> = HashSet::new();
                let mut remove_actions: HashMap<RemoveSCCState, String> = HashMap::new();
                for r in remove.drain() {
                    if !scc_state_ind[r.scc_ind].is_empty() {
                        //println!("looking to remove: {:?}", r);
                        if !removed_hist.iter().any(|x| x == &r.state) {
                            let delete_index = g_copy.node_indices().find(|i| g_copy[*i] == format!("({},{:?})", r.state.s, r.state.q)).unwrap();
                            g_copy.remove_node(delete_index);
                        }

                        // remove the state from the SCCs
                        //println!("Removal History: {:?}", removed_hist);
                        for transition in self.transitions.iter().filter(|x| x.sq_prime.iter().any(|y| *y == r.state) && r.state != x.sq && removed_hist.iter().all(|y| *y != x.sq)){
                            let mut remove_index: Vec<(usize, usize)> = Vec::new();
                            for (k, v) in scc_state_ind.iter().enumerate() {
                                //println!("sccs({}):{:?}, transition state: {:?}", k, v, transition.sq);
                                for (j, _state) in v.iter().filter(|x| **x == transition.sq).enumerate() {
                                    remove_index.push((k,j));
                                }
                            }
                            // there is a problem when the action is a part of a self loop, we are not handling this correctly
                            remove_actions.insert(RemoveSCCState {
                                state: ModelCheckingPair {s: transition.sq.s, q: transition.sq.q.clone()},
                                scc_ind: remove_index[0].0,
                                state_ind: remove_index[0].1,
                            }, transition.a.to_string());
                            //println!("Removal Actions: {:?}", remove_actions);
                        }
                        //println!("scc states: {:?}; total sccs: {:?}", scc_state_ind[r.scc_ind], scc_state_ind);
                        scc_state_ind[r.scc_ind].remove(r.state_ind);
                        removed_hist.push(r.state);
                        //println!("state-actions pairs under review: {:?}", remove_actions);
                    }
                }
                for (state, action) in remove_actions.iter() {
                    let mut actions = state_actions.get(&state.state).unwrap().clone();
                    //println!("actions: {:?}", actions);
                    if actions.is_empty() {
                        //println!("History: {:?}", removed_hist);
                        //println!("Found: {:?} to remove", state.state);
                        remove_more.insert(RemoveSCCState{state: state.state.clone(), scc_ind: state.scc_ind, state_ind: state.state_ind});
                    } else {
                        actions.remove(action);
                        // update the state actions to remove the actions involved
                        state_actions.insert(state.state.clone(), actions);
                    }
                }
                //println!("Remove updated: {:?}", remove_more);
                remove = remove_more.drain().collect();
            }
            // we have to remove the state from the sccs, before we can update what the new MECs are
            //let scc_test: Vec<Vec<petgraph::prelude::NodeIndex>> = kosaraju_scc(&g_copy);
            //let mut count: u32 = 0;
            //for ni_vect in scc_test.iter() {
            //for ni in ni_vect.iter() {
            //let i = ni;
            //println!("{}:{:?}", count, g_copy[*i]);
            //}
            //count += 1;
            //}
            /*if test_counter >= 4 {
                return mec_new
            }*/
            //test_counter += 1;
            mec_new = scc_state_ind.clone();
            //println!("Removal history: {:?}", removed_hist);
            //println!("Test counter: {}", test_counter);
        }
        mec_new
    }

    pub fn convert_string_node_to_state(&self, s: &str) -> Option<ModelCheckingPair> {
        //println!("state: {}", s);
        let re: Regex = Regex::new(r"^\((?P<input1>[0-9]+),(?P<input2>\[[0-9]+(?:,\s[0-9]+)*\])\)").unwrap();
        let input1 = re.captures(s).and_then(|x| {
            x.name("input1").map(|y| y.as_str())
        });
        let input2 = re.captures(s).and_then(|x| {
            x.name("input2").map(|y| y.as_str())
        });
        //println!("input1: {:?}, input2: {:?}", input1, input2);
        let return_state: Option<ModelCheckingPair> = match parse_int(input1.unwrap()) {
            Ok(x) =>
                {
                    match parse_str_vect(input2.unwrap()) {
                        Ok(y) => Some(ModelCheckingPair {s: x, q: y }),
                        _ => None
                    }
                },
            _ => {
                println!("while parsing graph node: {}, received error", s, );
                None
            }
        };
        return_state
    }

    pub fn find_state_index(&self, sq: &ModelCheckingPair) -> usize {
        self.states.iter().position(|x| x.s == sq.s && x.q == sq.q).unwrap()
    }

    /// Create a default value
    pub fn default() -> ProductMDP {
        ProductMDP{
            states: vec![],
            initial: ModelCheckingPair { s: 0, q: vec![] },
            transitions: vec![],
            labelling: vec![]
        }
    }
}

#[derive(Debug)]
pub struct ProductDRA {
    pub states: Vec<Vec<u32>>,
    pub sigma: Vec<String>,
    pub initial: Vec<u32>,
    pub delta: Vec<ProductDRATransitions>,
    pub acc: Vec<AcceptanceSet>,
}

#[derive(Deserialize,Debug)]
// Because this is a deterministic automata (q,w) -> q will always be true, and not (q,w) -> Vec<q>
pub struct ProductDRATransitions {
    q: Vec<u32>,
    w: Vec<String>,
    q_prime: Vec<u32>,
}

#[derive(Debug)]
pub struct TempProductDRATransitions {
    pub q: Vec<u32>,
    pub w: String,
    pub q_prime: Vec<u32>
}

impl ProductDRA {
    pub fn create_states(&mut self, dra: &DRAMod) {
        if self.states.is_empty() {
            // this is the first product DRA
            for q in dra.states.iter() {
                self.states.push(vec![*q]);
            }
        } else {
            let mut q_bar: Vec<Vec<u32>> = Vec::new();
            let it = self.states.clone().into_iter().cartesian_product(dra.states.clone().into_iter());
            for (vect, val) in it.into_iter(){
                let mut new_q: Vec<u32> = vect;
                new_q.push(val);
                q_bar.push(new_q)
            }
            self.states = q_bar;
        }
    }

    pub fn create_transitions(&mut self, dras: &Vec<DRAMod>) {

        //let re: Regex = Regex::new(r"\([0-9]\)").unwrap();
        let mut temp_transitions: Vec<TempProductDRATransitions> = Vec::new();

        for w in self.sigma.iter() {
            for state in self.states.iter() {
                let mut q_prime: Vec<u32> = vec![0; state.len()];
                for (i, q) in state.iter().enumerate() {
                    // if w contains an (k) for k in N then we need to replace the number by a symbolic k
                    // the last DRA could be a safety property, we can check this with the DRA safety variable

                    // In the product DRA we determine which task is active.
                    let new_word = w.to_string();
                    /*if dras[i].safety {
                        new_word = re.replace(w, "(k)").to_string();
                        //println!("w: {}, new word: {}", w, new_word);
                    }*/
                    for transition in dras[i].delta.iter().filter(|x| x.q == *q && x.w.contains(&new_word)){
                        q_prime[i] = transition.q_prime;
                    }
                }
                temp_transitions.push(TempProductDRATransitions {
                    q: state.clone(),
                    w: w.to_string(),
                    q_prime: q_prime
                });
            }
        }

        for q_bar in self.states.iter() {
            for q_bar_prime in self.states.iter() {
                let mut word_vect: Vec<String> = Vec::new();
                for transition in temp_transitions.iter().filter(|x| x.q == *q_bar && x.q_prime == *q_bar_prime) {
                    //println!("transition compression: {:?}", transition)
                    word_vect.push(transition.w.to_string());
                }
                if !word_vect.is_empty() {
                    self.delta.push(ProductDRATransitions {
                        q: q_bar.clone(),
                        w: word_vect,
                        q_prime: q_bar_prime.clone()
                    })
                }
            }
        }
    }

    pub fn task_count(dras: &Vec<DRA>) -> u32 {
        let mut count: u32 = 0;
        for dra in dras.iter() {
            if !dra.safety {
                count += 1;
            }
        }
        count
    }

    pub fn set_initial(&mut self, dras: &Vec<DRAMod>) {
        let mut init: Vec<u32> = vec![0; dras.len()];
        for (i, dra) in dras.iter().enumerate() {
            init[i] = dra.initial;
        }
        self.initial = init;
    }

    pub fn default() -> ProductDRA {
        ProductDRA {
            states: vec![],
            sigma: vec![],
            initial: vec![],
            delta: vec![],
            acc: vec![]
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq)]
struct RemoveSCCState {
    state: ModelCheckingPair,
    scc_ind: usize,
    state_ind: usize,
}

#[derive(Debug,Clone,Eq, PartialEq, Hash)]
pub struct ModelCheckingPair {
    pub s: u32,
    pub q: Vec<u32>
}