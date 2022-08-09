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
mod tests {

    use super::*;
    use rulinalg::vector;
    use rulinalg::matrix;

    #[test]
    fn successfull_init_of_neu_net() {
        let layer_nodes = vec![1, 2, 3]; 
        let layer_types = vec!["foo", "bar", "hi"];
        let layers = vec![vector![1.3, 1.2, 10.1]];
        let bias = vec![vector![1.3, 1.2, 10.2]];
        let weights = vec![matrix![1.0, 0.5, 0.5;
                0.5, 1.0, 0.5;
                0.5, 0.5, 1.0]];


        let neu_net = NeuNet{
            layer_nodes: &layer_nodes,
            layer_types: &layer_types,
            layers: &layers,
            bias: &bias,
            weights: &weights
        };

        assert_eq!(neu_net.layer_nodes, &layer_nodes);
        assert_eq!(neu_net.layer_types, &layer_types);
        assert_eq!(neu_net.layers, &layers);
        assert_eq!(neu_net.bias, &bias);
        assert_eq!(neu_net.weights, &weights);

    }
}
