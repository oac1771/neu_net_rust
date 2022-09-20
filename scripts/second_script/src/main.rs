// use neural_network::neu_net::builder::builder::Builder;
use utils::mnist::MnistData;

// process images data
    // https://stackoverflow.com/questions/46867355/is-it-possible-to-split-a-vector-into-groups-of-10-with-iterators
// return list of MnistData
    // maybe make datalength set customizable??
// have train take in list of MnistData -> might need MnistDate to impl Trait Data (????)

fn main() {
    let label_path = "data/t10k-labels-idx1-ubyte.gz";
    let image_path = "data/t10k-images-idx3-ubyte.gz";
    let _mnist = MnistData::load_data(label_path, image_path);
}

