use crate::activations::ActivationFunction;

use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
use rulinalg::vector;

// backpropagation

pub struct NeuNet{
    pub layer_nodes: Vec<usize>,
    pub bias: Vec<Vector<f32>>,
    pub weights: Vec<Matrix<f32>>,
    pub layer_types: Vec<Box<dyn ActivationFunction>> 
}

impl NeuNet {

    pub fn evaluate(&self, input: &Vector<f32>) -> Vector<f32> {

        let mut dot_product = vector![];
        let mut act_layer = self.layer_types[0].act(input);

        for index in 0..self.layer_nodes.len()-1 {
            dot_product = self.layer_types[index].act(&(&self.weights[index] * &act_layer + &self.bias[index]));
            act_layer = dot_product.clone();
        }

        return dot_product
    }
}


#[cfg(test)]
#[path = "./test/test_neu_net.rs"]
mod test_neu_net;
