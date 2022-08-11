use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

// todo: add loop for evaluate function
// todo: add error logic to catch if input is wrong size
// todo: create builder logic

pub struct NeuNet{
    layer_nodes: Vec<i32>,
    bias: Vec<Vector<f32>>,
    weights: Vec<Matrix<f32>>
}

impl NeuNet {
    pub fn evaluate(&self, input: Vector<f32>) -> Vector<f32> {

        // for index in 1..self.layer_nodes.len(){

        // }
        let dot_product = &self.weights[0] * &input + &self.bias[0];
        return dot_product
    }
}


#[cfg(test)]
#[path = "./test/test_neu_net.rs"]
mod test_neu_net;
