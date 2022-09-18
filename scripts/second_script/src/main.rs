// use neural_network::neu_net::builder::builder::Builder;
use utils::mnist::MnistData;
use std::env::current_dir;

// figure out how to read from different directories (full path or relative does not work)

fn main() {
    let path = "../data/t10k-labels-idx1-ubyte.gz";
    let cwd = current_dir();
    let _mnist = MnistData::load_data(path);

    println!("path {}", path);
    println!("cwd {:?}", cwd);
}
