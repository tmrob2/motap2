extern crate gurobi;
use gurobi::*;
use std::collections::HashMap;
use super::decomp_team_mdp;

pub fn witness(hullset: &Vec<Vec<f64>>, target: &Vec<f64>, dim: &usize, num_agents: &usize) -> Option<Vec<f64>> {

    //let env = Env::new().unwrap();
    let mut env = gurobi::Env::new("").unwrap();
    env.set(param::OutputFlag, 0).unwrap();
    env.set(param::LogToConsole, 0).unwrap();
    let mut model = Model::new("model2", &env).unwrap();


    let mut v: HashMap<String, gurobi::Var> = HashMap::new();
    //println!("|hullset|: {}", hullset.len());
    for i in 0..hullset.len() {
        v.insert(format!("u{}", i), model.add_var(&*format!("u{}", i), Continuous, 0.0, 0.00, 1.0, &[], &[]).unwrap());
    }
    let dummy = model.add_var("dummy", Continuous, 0.0, 0.0, 0.0, &[], &[]).unwrap();
    model.update().unwrap();

    let mut u_vars  = Vec::new();
    for i in 0..v.len(){
        let u = v.get(&format!("u{}", i)).unwrap();
        u_vars.push(u.clone());
    }

    let mut q_transpose: Vec<Vec<f64>> = Vec::new();
    for i in 0..*dim {
        let mut q = Vec::new();
        for j in 0..hullset.len() {
            q.push(hullset[j][i]);
        }
        q_transpose.push(q);
    }

    //println!("qT: {:?}", q_transpose);

    for i in 0..*dim {
        //let expr = LinExpr::new();
        //let n_expr = expr.add_terms(&q_transpose[i][..], &u_vars[i][..]);
        let q_new = &q_transpose[i];
        //println!("sum(q): {}", q_sum);
        //let count_contrib = q_transpose[i].iter().fold(0u32, |sum, &val| if f64::abs(val) > 0.0 { sum + 1u32} else { sum });
        //println!("data constributions for var {}: {}", i, count_contrib);
        //if f64::abs(q_sum) > 0.0 && count_contrib >= *num_agents as u32 {
        let ui_expr = LinExpr::new();
        //println!("q_new: {:?}", q_new);
        //println!("u vars: {}", &u_vars[..].len());
        let ui_expr1 = ui_expr.add_terms(&q_new[..], &u_vars[..]);
        model.add_constr(&*format!("c{}", i), ui_expr1, Greater, target[i]);
        //}
        //println!("q.v: {}, r:{}", q_sum, target[i]);

        //println!("q_transpose: {:?}", q_transpose[i]);
    }
    let mut u_expr = LinExpr::new();
    let coefs: Vec<f64> = vec![1.0; hullset.len()];
    let final_expr = u_expr.add_terms( &coefs[..], &u_vars[..]);
    model.add_constr("u_final", final_expr, gurobi::Equal, 1.0);

    model.update().unwrap();
    model.set_objective(dummy,gurobi::Maximize).unwrap();

    //println!("Model type: {:?}", model.get(gurobi::attr::IsMIP));

    //model.write("logfile.lp").unwrap();

    model.optimize().unwrap();
    //println!("model status: {:?}", model.status());
    //println!("model obj: {:?}", model.get(gurobi::attr::ObjVal).unwrap());
    let mut vars = Vec::new();
    for i in 0..hullset.len() {
        let var = v.get(&format!("u{}",i)).unwrap();
        vars.push(var.clone());
    }
    match model.get_values(attr::X, &vars[..]) {
        Ok(x) => {Some(x)}
        Err(e) => { println!("unable to retrieve var because: {:?}", e); None}
    }

}

pub fn lp5(h: &Vec<Vec<f64>>, t: &Vec<f64>, dim: &usize) -> Option<Vec<f64>> {
    //h: &Vec<Vec<f64>>, t: &Vec<f64>, dim: &usize
    let mut env = gurobi::Env::new("").unwrap();
    env.set(param::OutputFlag, 0).unwrap();
    env.set(param::LogToConsole, 0).unwrap();
    env.set(param::InfUnbdInfo, 1);
    //env.set(param::FeasibilityTol,10e-9).unwrap();
    env.set(param::NumericFocus,2).unwrap();
    let mut model = Model::new("model1", &env).unwrap();
    let scale: f64 = 1f64;

    // create an empty model
    //let mut model = env.new_model("model1").unwrap();

    // add vars
    let lb: f64 = scale / 10000f64;
    let mut v: HashMap<String, gurobi::Var> = HashMap::new();
    for i in 0..*dim {
        v.insert(format!("w{}", i), model.add_var(
            &*format!("w{}", i), Continuous, 0.0, lb, scale, &[], &[]).unwrap()
        );
    }
    let d = model.add_var(
        "d", Continuous, 0.0, -gurobi::INFINITY, gurobi::INFINITY, &[], &[]
    ).unwrap();

    model.update().unwrap();
    let mut w_vars = Vec::new();
    for i in 0..*dim {
        let w = v.get(&format!("w{}", i)).unwrap();
        w_vars.push(w.clone());
    }

    let mut t_expr = LinExpr::new();
    let t_expr1 = t_expr.add_terms(&t[..], &w_vars[..]);
    let t_expr2 = t_expr1.add_term(1.0, d.clone());
    let t_expr3 = t_expr2.add_constant(-1.0);
    model.add_constr("t0", t_expr3, gurobi::Greater, 0.0);

    for (i, x) in h.iter().enumerate() {
        let mut expr = LinExpr::new();
        let expr1 = expr.add_terms(&x[..], &w_vars[..]);
        let expr2 = expr1.add_term(1.0, d.clone());
        let expr3 = expr2.add_constant(-1.0);
        model.add_constr(&*format!("c{}", i), expr3, gurobi::Less, 0.0);
    }

    let mut w_expr = LinExpr::new();
    let coefs: Vec<f64> = vec![1.0; *dim];
    let final_expr = w_expr.add_terms( &coefs[..], &w_vars[..]);
    model.add_constr("w_final", final_expr, gurobi::Equal, scale);

    model.update().unwrap();

    model.set_objective(&d, gurobi::Maximize).unwrap();

    //println!("Model type: {:?}", model.get(gurobi::attr::IsMIP));

    //model.write("logfile.lp").unwrap();

    model.optimize().unwrap();
    //println!("model status: {:?}", model.status());
    //println!("kappa: {:?}", model.get(gurobi::attr::KappaExact).unwrap());
    //println!("model obj: {:?}", model.get(gurobi::attr::ObjVal).unwrap());
    let mut vars = Vec::new();
    for i in 0..*dim {
        let var = v.get(&format!("w{}",i)).unwrap();
        vars.push(var.clone());
    }
    let val = model.get_values(attr::X, &vars[..]).unwrap();
    let val_scaled: Vec<f64> = val.iter().map(|x| if *x > 0.0 { *x / scale } else { lb / scale } ).collect();
    if model.status().unwrap() == Status::Infeasible {
        None
    } else {
        Some(val_scaled)
    }
}