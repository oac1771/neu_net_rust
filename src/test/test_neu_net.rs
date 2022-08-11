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

// #[test]
// fn succesfull_evaluation_of_input() {
//     let neu_net = Builder::build(vec![4, 3, 2]);
//     let input = vector![2.0; neu_net.layer_nodes[0].clone() as usize];
//     let result = neu_net.evaluate(input);

//     assert_eq!(result.into_iter().len(), 2);
//     // assert_eq!(&result, &vector![1.0])
// }
