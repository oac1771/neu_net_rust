use crate::builder::Builder;

use rulinalg::vector;

#[test]
fn succesfull_evaluation_of_input() {
    let layer_nodes = vec![4, 3, 2];
    let neu_net = Builder::build(&layer_nodes);
    let input = vector![10000.0; neu_net.layer_nodes[0]];
    let result = neu_net.evaluate(&input);

    assert_eq!(result.into_iter().len(), 2);
}
