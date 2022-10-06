use crate::neu_net::builder::activations::Sigmoid;
use crate::neu_net::builder::activations::ActivationFunction;

use ndarray::array;

#[test]
fn successfull_sigmoid_activation_function() {
    let input = array![[-10000000000.0, 0.0, 1000000000.0]]; 
    let result = Sigmoid{}.act(&input);

    assert_eq!(result, array![[0.0, 0.5, 1.0]])
}

#[test]
fn successfull_sigmoid_derivative_activation_function() {
    let input = array![[-55.0, 0.0, 55.0]]; 
    let result = Sigmoid{}.dactdz(&input);

    assert_eq!(result, array![[1.2995814250075068e-24, 0.25, 1.2995814250075068e-24]])
}
