use neural_network::neu_net::builder::builder::Builder;
use utils::mnist::MnistData;
use std::time::Instant;

fn main() {
    let label_path = "data/t10k-labels-idx1-ubyte.gz";
    let image_path = "data/t10k-images-idx3-ubyte.gz";
    let mnist_data = MnistData::load_data(label_path, image_path).unwrap();

    println!("Building Neural Network");
    let layer_nodes = vec![784, 10];
    let learning_rate = 0.9;
    let training_iterations = 1;
    let mut neu_net = Builder::build(&layer_nodes);

    let start = Instant::now();
    time_graph::enable_data_collection(true);
    neu_net.train(&mnist_data, training_iterations, learning_rate);
    let duration = start.elapsed();

    let graph = time_graph::get_full_graph();

    println!("{}", graph.as_dot());
    println!("Training time is: {:?}", duration);


}