use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

// todo: create builder logic (check)
    // todo: add error logic for if input len is less than one
// todo: read this https://doc.rust-lang.org/book/ch17-03-oo-design-patterns.html
// todo: add loop for evaluate function
// todo: add error logic to catch if input is wrong size

pub struct NeuNet{
    pub layer_nodes: Vec<usize>,
    pub bias: Vec<Vector<f32>>,
    pub weights: Vec<Matrix<f32>>
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
