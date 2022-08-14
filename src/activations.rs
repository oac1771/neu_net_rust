use rulinalg::vector::Vector;
use std::f32::consts::E;

pub trait ActivationFunction {
    fn act(&self, input: &Vector<f32>) -> Vector<f32>;
}


pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn act(&self, input: &Vector<f32>) -> Vector<f32> {
        let result = input.into_iter().map(|x| {
            1.0 / (1.0 + E.powf(-x)) 
        }).collect();
        return result
    }
}


#[cfg(test)]
#[path = "./test/test_activations.rs"]
mod test_neu_net;
