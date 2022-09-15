// cargo-deps: time="0.1.25", neural_network 
extern crate time;
extern crate neural_network;

use crate::neural_network::builder::builder::Builder;

fn main() {
    let layer_nodes = vec![4,3,1];
    let builder = Builder::build(&layer_nodes);
    println!("{}", time::now().rfc822z());

}