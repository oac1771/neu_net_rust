use crate::neu_net::neu_net::NeuNet;
use crate::neu_net::builder::builder::Builder;



#[test]
fn successfull_init_of_neu_net_using_builder() {
    let layer_nodes = vec![4, 3, 2]; 
    let neu_net: NeuNet = Builder::build(&layer_nodes);

    assert_eq!(&neu_net.layer_nodes, &layer_nodes);

    for index in 1..layer_nodes.len() {
        assert_eq!(neu_net.bias[index-1].iter().len(), layer_nodes[index]);
        assert_eq!(neu_net.weights[index-1].shape(), [layer_nodes[index], layer_nodes[index - 1]] );
    }
}

#[test]
#[should_panic]
fn catch_error_if_layer_nodes_less_than_one() {
    let layer_nodes = vec![4]; 
    let _neu_net: NeuNet = Builder::build(&layer_nodes);
}