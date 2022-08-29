use crate::builder::activations::ActivationFunction;
use crate::builder::costs::CostFunction;
use crate::builder::data::Data;

use rulinalg::matrix::{Matrix, BaseMatrix};
use rulinalg::vector::Vector;

// figure out how to get matrix from two vectors
    // could use Matrix::from_fn and build it


pub struct NeuNet{
    pub layer_nodes: Vec<usize>,
    pub bias: Vec<Vector<f64>>,
    pub weights: Vec<Matrix<f64>>,
    pub layer_types: Vec<Box<dyn ActivationFunction>>,
    pub cost_function: Box<dyn CostFunction>
}

pub struct Propagation {
    pub weighted_inputs: Vec<Vector<f64>>,
    pub activations: Vec<Vector<f64>>
}

impl NeuNet {

    pub fn evaluate(&self, input: &Vector<f64>) -> Vector<f64> {

        let propagation = self.eval(input);

        return propagation.activations.last().unwrap().clone()
    }

    pub fn train(&mut self, data: Vec<Data>, training_iterations: i32, learning_rate: f64) {
        
        let training_constant = learning_rate / data.len() as f64;
        let mut propagation: Propagation;
        let mut output_error: Vector<f64>;
        let last_element = self.layer_types.len() - 1;

        for index in 0..training_iterations {
            for data in &data {
                
                propagation = self.eval(&data.data);
                let dcostdact = self.cost_function.dcostdact(&data.label, &propagation.activations[last_element]);
                let dactdz =  self.layer_types[last_element].dactdz(&propagation.weighted_inputs[last_element]);
                output_error = dcostdact.elemul(&dactdz);
                
                self.backpropagation(propagation, output_error, training_constant)
            }
        }
    }

    fn eval(&self, data: &Vector<f64>) -> Propagation {

        let mut weighted_inputs: Vec<Vector<f64>> = vec![];
        let mut activations: Vec<Vector<f64>> = vec![];
        let mut act_layer = self.layer_types[0].act(data);

        for index in 0..self.layer_nodes.len()-1 {
            let weighted_input = &self.weights[index] * act_layer + &self.bias[index];
            let activation = self.layer_types[index].act(&weighted_input);
            
            weighted_inputs.push(weighted_input);
            activations.push(activation.clone());

            act_layer = activation
        }

        return Propagation{weighted_inputs, activations}
    }

    fn backpropagation(&mut self, propagation: Propagation, output_error: Vector<f64>, training_constant: f64) {

        let mut delta_layer = output_error;

        for index in (0..(self.layer_nodes.len()-1)).rev() {
            // let delta_weights = delta_layer * propagation.activations[index];
            let weight = &self.weights[index];
            let layer = delta_layer;
            let foo = weight.transpose() * &layer;
            let dactdz = &self.layer_types[index].dactdz(&propagation.weighted_inputs[index]);
            delta_layer = (foo).elemul(dactdz);

            self.bias[index] = &self.bias[index] - &delta_layer;
        }
    }
}


#[cfg(test)]
#[path = "./test/test_neu_net.rs"]
mod test_neu_net;
