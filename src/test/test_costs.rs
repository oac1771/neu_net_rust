use crate::neu_net::builder::costs::Quadratic;
use crate::neu_net::builder::costs::CostFunction;
use rulinalg::vector;

#[test]
fn successfull_quadratic_cost_function() {
    let output = vector![-1.0, 0.0, 1.0];
    let label = vector![1.0, 0.0, 0.0];
    let result = Quadratic{}.cost(&label, &output);

    assert_eq!(result, vector![2.0, 0.0, 0.5]);
}

#[test]
fn successfull_quadratic_dcostdact_function() {
    let output = vector![-1.0, 0.0, 1.0];
    let label = vector![1.0, 0.0, 0.0];
    let result = Quadratic{}.dcostdact(&label, &output, 100.0);

    assert_eq!(result, vector![-0.02, 0.0, 0.01]);
}