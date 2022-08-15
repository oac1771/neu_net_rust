use crate::neu_net::NeuNet;
use crate::builder::activations::{ActivationFunction, Sigmoid};

use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

static MATRIX_SCALE: f32 = 0.5;
static BIAS_SCALE: f32 = 0.5;

pub struct Builder{}

impl Builder{

    pub fn build(layer_nodes: &Vec<usize>) -> NeuNet {
        
        if layer_nodes.len() <= 1 {
            panic!("Must Provide a Network of More Than One Layer")
        }

        let mut weights: Vec<Matrix<f32>> = Vec::new();
        let mut bias: Vec<Vector<f32>> = Vec::new();
        let mut layer_types: Vec<Box<dyn ActivationFunction>> = Vec::new();
        
        for index in 1..layer_nodes.len() {
            weights.push(
                Matrix::from_fn(layer_nodes[index], 
                    layer_nodes[index - 1],
                    |_col, _row| {
                        MATRIX_SCALE * 1.0
                    }
                )
            );
            bias.push(
                Vector::from_fn(layer_nodes[index], 
                        |_row| {
                        BIAS_SCALE * 1.0
                    }
                )
            );

            layer_types.push(Box::new(Sigmoid{}))
        }

        let neu_net = NeuNet{
            layer_nodes: layer_nodes.to_vec(),
            weights: weights,
            bias: bias,
            layer_types: layer_types
        };

        return neu_net;
    }

}

#[cfg(test)]
#[path = "../test/test_builder.rs"]
mod test;