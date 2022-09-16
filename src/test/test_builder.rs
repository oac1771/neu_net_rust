use crate::neu_net::neu_net::NeuNet;
use crate::builder::builder::Builder;

use rulinalg::matrix::BaseMatrix;


#[test]
fn successfull_init_of_neu_net_using_builder() {
    let layer_nodes = vec![4, 3, 2]; 
    let neu_net: NeuNet = Builder::build(&layer_nodes);

    assert_eq!(&neu_net.layer_nodes, &layer_nodes);

    for index in 0..layer_nodes.len()-1 {
        assert_eq!(neu_net.bias[index].iter().len(), layer_nodes[index + 1]);
        assert_eq!(neu_net.weights[index].rows(), layer_nodes[index + 1]);
        assert_eq!(neu_net.weights[index].cols(), layer_nodes[index]);
    }
}

#[test]
#[should_panic]
fn catch_error_if_layer_nodes_less_than_one() {
    let layer_nodes = vec![4]; 
    let _neu_net: NeuNet = Builder::build(&layer_nodes);
}