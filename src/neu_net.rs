use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;


pub struct NeuNet<'a>{
    layer_nodes: &'a Vec<i32>,
    layer_types: &'a Vec<&'a str>,
    layers: &'a Vec<Vector<f32>>,
    bias: &'a Vec<Vector<f32>>,
    weights: &'a Vec<Matrix<f32>>
}

#[cfg(test)]
#[path = "./test/test_neu_net.rs"]
mod test_neu_net;
