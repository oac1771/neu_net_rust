use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
use rulinalg::vector;

// todo: read this to improve error handling https://doc.rust-lang.org/book/ch13-00-functional-features.html

pub struct NeuNet{
    pub layer_nodes: Vec<usize>,
    pub bias: Vec<Vector<f32>>,
    pub weights: Vec<Matrix<f32>>
}

impl NeuNet {

    pub fn evaluate(&self, input: Vector<f32>) -> Vector<f32> {

        if input.clone().into_iter().len() != self.layer_nodes[0] {
            panic!("Please make sure length of input is {:?}", self.layer_nodes[0])
        }

        let result = self.propagate(input);
        return result
    }

    fn propagate(&self, mut input: Vector<f32>) -> Vector<f32> {
        let mut dot_product = vector![];

        for index in 0..self.layer_nodes.len()-1 {
            dot_product = &self.weights[index] * input + &self.bias[index];
            input = dot_product.clone();
        }

        return dot_product
    }

}


#[cfg(test)]
#[path = "./test/test_neu_net.rs"]
mod test_neu_net;
