use crate::neu_net::NeuNet;
use crate::builder::Builder;

use rulinalg::vector;
use rulinalg::matrix;

#[test]
fn successfull_init_of_neu_net() {
    let layer_nodes = vec![1, 2, 3]; 
    let bias = vec![vector![1.3, 1.2, 10.2]];
    let weights = vec![matrix![1.0, 0.5, 0.5;
            0.5, 1.0, 0.5;
            0.5, 0.5, 1.0]];

    let neu_net = NeuNet{
        layer_nodes: layer_nodes.to_vec(),
        bias: bias.to_vec(),
        weights: weights.to_vec()
    };

    assert_eq!(neu_net.layer_nodes, layer_nodes);
    assert_eq!(neu_net.bias, bias);
    assert_eq!(neu_net.weights, weights);

}

#[test]
fn succesfull_evaluation_of_input() {
    let layer_nodes = vec![4, 3, 2];
    let neu_net = Builder::build(&layer_nodes);
    let input = vector![2.0; neu_net.layer_nodes[0]];
    let result = neu_net.evaluate(input);

    assert_eq!(result.into_iter().len(), 2);
}

#[test]
#[should_panic(expected = "Please make sure length of input is 4")]
fn catch_error_if_layer_nodes_less_than_one() {
    let layer_nodes = vec![4, 3];
    let input = vector![1.0, 2.2, 3.4];
    let neu_net: NeuNet = Builder::build(&layer_nodes);

    _ = neu_net.evaluate(input)
}