use rulinalg::vector::Vector;
use std::f64::consts::E;

pub trait ActivationFunction {
    fn act(&self, input: &Vector<f64>) -> Vector<f64>;
    fn dactdz(&self, input: &Vector<f64>) -> Vector<f64>;
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn act(&self, input: &Vector<f64>) -> Vector<f64> {
        let result = input.into_iter().map(|x| {
            1.0 / (1.0 + E.powf(-x)) 
        }).collect();
        return result
    }

    fn dactdz(&self, input: &Vector<f64>) -> Vector<f64> {
        let result = input.into_iter().map(|x| {
            E.powf(-x) / (1.0 + E.powf(-x)).powf(2.0)
        }).collect();
        return result
    }

}


#[cfg(test)]
#[path = "../test/test_activations.rs"]
mod test;