use crate::neu_net::builder::activations::ActivationFunction;
use crate::neu_net::builder::costs::CostFunction;
use crate::neu_net::builder::data::Data;

use ndarray::Array2;
use time_graph::spanned;

pub struct NeuNet{
    pub layer_nodes: Vec<usize>,
    pub bias: Vec<Array2<f64>>,
    pub weights: Vec<Array2<f64>>,
    pub layer_types: Vec<Box<dyn ActivationFunction>>,
    pub cost_function: Box<dyn CostFunction>
}

pub struct Propagation {
    pub weighted_inputs: Vec<Array2<f64>>,
    pub activations: Vec<Array2<f64>>
}

impl NeuNet {

    pub fn evaluate(&self, input: &Array2<f64>) -> Array2<f64> {

        let propagation = self.eval(input);
        return propagation.activations.last().unwrap().clone()
    }

    pub fn train(&mut self, data: &Vec<impl Data>, training_iterations: i32, learning_rate: f64) {
        
        let mut propagation: Propagation;
        let mut output_error: Array2<f64>;
        let last_element = self.layer_types.len() - 1;

        for _index in 0..training_iterations {
            for training_data in data {
                
                propagation = self.eval(&training_data.get_data());
                let dcostdact = self.cost_function.dcostdact(&training_data.get_label(), 
                    &propagation.activations.last().clone().unwrap());
                let dactdz =  self.layer_types[last_element].dactdz(&propagation.weighted_inputs.last().clone().unwrap());
                output_error = dcostdact * &dactdz;
                
                self.backpropagation(&propagation, output_error, learning_rate)
            }
        }
    }

    #[time_graph::instrument]
    fn eval(&self, data: &Array2<f64>) -> Propagation {

        let mut weighted_inputs: Vec<Array2<f64>> = Vec::with_capacity(self.layer_nodes.len()-1);
        let mut activations: Vec<Array2<f64>> = Vec::with_capacity(self.layer_nodes.len());
        let mut act_layer = self.layer_types[0].act(data);

        activations.push(act_layer.clone());

        for index in 0..self.layer_nodes.len()-1 {
            let weighted_input: Array2<f64>;

            let _ = spanned!("dot productg", {
                weighted_input = &self.weights[index].dot(&act_layer) + &self.bias[index];
            });

            let activation = self.layer_types[index].act(&weighted_input);

            let _ = spanned!("pushing to vec", {
                weighted_inputs.push(weighted_input);
                activations.push(activation.clone());
                                
            });

            act_layer = activation;
        }

        return Propagation{weighted_inputs, activations}
    }

    #[time_graph::instrument]
    fn backpropagation(&mut self, propagation: &Propagation, output_error: Array2<f64>, learning_rate: f64) {

        let mut layer_errors = vec![];
        let mut delta_layer = output_error;
        layer_errors.push(delta_layer.clone());

        for index in (0..(self.layer_nodes.len() - 2)).rev() {

            let error_term = (&self.weights[index + 1].t()).dot(&delta_layer);
            let dactdz = self.layer_types[index].dactdz(&propagation.weighted_inputs[index]);
            delta_layer = error_term * &dactdz;

            layer_errors.push(delta_layer.clone());
        }
        self.update_controls(&layer_errors, &propagation.activations, learning_rate)
    }

    #[time_graph::instrument]
    fn update_controls(&mut self, layer_errors: &Vec<Array2<f64>>, activations: &Vec<Array2<f64>>, _learning_rate: f64) {

        for (index, layer_error) in layer_errors.iter().rev().enumerate() {

            let _ = spanned!("matrix calc", {
                let _foo = layer_error.dot(&activations[index].t());
                                
            });
            self.bias[index] = &self.bias[index] - layer_error;
            // self.weights[index] = &self.weights[index] - delta_weight;

        }

    }
}


#[cfg(test)]
#[path = "./test/test_neu_net.rs"]
mod test_neu_net;
