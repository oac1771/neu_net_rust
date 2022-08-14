use crate::activations::*;

use rulinalg::vector;

#[test]
fn successfull_init_of_neu_net_using_builder() {
    let input = vector![-1000000.0, 0.0, 1000000000.0]; 
    let result = Sigmoid{}.act(&input);

    assert_ne!(input, result);
}