use crate::neu_net::NeuNet;
use crate::builder::activations::{ActivationFunction, Sigmoid};
use crate::builder::costs::Quadratic;

use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
use rand::Rng;

pub struct Builder{}

impl Builder{

    pub fn build(layer_nodes: &Vec<usize>) -> NeuNet {
        
        if layer_nodes.len() <= 1 {
            panic!("Must Provide a Network of More Than One Layer")
        }
        
        let cost_function = Box::new(Quadratic{});
        let mut weights: Vec<Matrix<f64>> = Vec::new();
        let mut bias: Vec<Vector<f64>> = Vec::new();
        let mut layer_types: Vec<Box<dyn ActivationFunction>> = Vec::new();
        let mut rng = rand::thread_rng();
        layer_types.push(Box::new(Sigmoid{}));
        
        for index in 1..layer_nodes.len() {
            weights.push(
                Matrix::from_fn(layer_nodes[index], 
                    layer_nodes[index - 1],
                    |_col, _row| {
                        rng.gen_range(-1.0..1.0)
                    }
                )
            );
            bias.push(
                Vector::from_fn(layer_nodes[index], 
                        |_row| {
                        rng.gen_range(-1.0..1.0)
                    }
                )
            );

            layer_types.push(Box::new(Sigmoid{}))
        }

        let neu_net = NeuNet{
            layer_nodes: layer_nodes.to_vec(),
            weights: weights,
            bias: bias,
            layer_types: layer_types,
            cost_function: cost_function
        };

        return neu_net;
    }

}

#[cfg(test)]
#[path = "../test/test_builder.rs"]
mod test;