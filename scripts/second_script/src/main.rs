// use neural_network::neu_net::builder::builder::Builder;
use utils::mnist::MnistData;

// figure out which image file to use
// add iter_map to change from u8 to f64
// return list of MnistData
    // maybe make datalength set customizable??
// have train take in list of MnistData -> might need MnistDate to impl Trait Data (????)

fn main() {
    let label_path = "data/t10k-labels-idx1-ubyte.gz";
    let image_path = "data/t10k-labels-idx1-ubyte.gz";
    let _mnist = MnistData::load_data(label_path, image_path);
}

