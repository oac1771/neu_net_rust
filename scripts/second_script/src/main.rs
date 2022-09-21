// use neural_network::neu_net::builder::builder::Builder;
use utils::mnist::MnistData;

// have train take in list of MnistData -> might need MnistDate to impl Trait Data (????)

fn main() {
    let label_path = "data/t10k-labels-idx1-ubyte.gz";
    let image_path = "data/t10k-images-idx3-ubyte.gz";
    let _mnist_data = MnistData::load_data(label_path, image_path).unwrap();
}

