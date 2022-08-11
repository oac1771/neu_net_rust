use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
use rulinalg::vector;

pub struct NeuNet{
    layer_nodes: Vec<i32>,
    layer_types: Vec <&'static str>,
    layers: Vec<Vector<f32>>,
    bias: Vec<Vector<f32>>,
    weights: Vec<Matrix<f32>>
}

impl NeuNet {
    fn evaluate(&self) -> Vector<f32> {
        return vector![1.3]
    }
}


#[cfg(test)]
#[path = "./test/test_neu_net.rs"]
mod test_neu_net;
