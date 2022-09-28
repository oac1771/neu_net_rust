use ndarray::Array1;
use std::f64::consts::E;

pub trait ActivationFunction {
    fn act(&self, input: &Array1<f64>) -> Array1<f64>;
    fn dactdz(&self, input: &Array1<f64>) -> Array1<f64>;
}

pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn act(&self, input: &Array1<f64>) -> Array1<f64> {
        let result = input.map(|x| {
            1.0 / (1.0 + E.powf(-x)) 
        });
        return result
    }

    fn dactdz(&self, input: &Array1<f64>) -> Array1<f64> {
        let result = input.map(|x| {
            E.powf(-x) / (1.0 + E.powf(-x)).powf(2.0)
        });
        return result
    }

}


#[cfg(test)]
#[path = "./../test/test_activations.rs"]
mod test;