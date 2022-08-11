use crate::neu_net::NeuNet;

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
        layer_nodes: layer_nodes.to_vec(),
        layer_types: layer_types.to_vec(),
        layers: layers.to_vec(),
        bias: bias.to_vec(),
        weights: weights.to_vec()
    };

    assert_eq!(neu_net.layer_nodes, layer_nodes);
    assert_eq!(neu_net.layer_types, layer_types);
    assert_eq!(neu_net.layers, layers);
    assert_eq!(neu_net.bias, bias);
    assert_eq!(neu_net.weights, weights);

}

#[test]
fn succesfull_evaluation_of_input() {
    let neu_net = return_neu_net();
    let result = neu_net.evaluate();

    assert_eq!(result, vector![1.3])
}

fn return_neu_net() -> NeuNet {
    let layer_nodes = vec![1, 2, 3]; 
    let layer_types = vec!["foo", "bar", "hi"];
    let layers = vec![vector![1.3, 1.2, 10.1]];
    let bias = vec![vector![1.3, 1.2, 10.2]];
    let weights = vec![matrix![1.0, 0.5, 0.5;
            0.5, 1.0, 0.5;
            0.5, 0.5, 1.0]];


    let neu_net = NeuNet{
        layer_nodes: layer_nodes,
        layer_types: layer_types,
        layers: layers,
        bias: bias,
        weights: weights
    };

    return neu_net;
}