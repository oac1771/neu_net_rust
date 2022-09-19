use std::io::Error;
use std::io::prelude::*;
use rulinalg::vector::Vector;
use flate2::read::GzDecoder;
use std::fs::File;

pub struct MnistData {
    pub data: Vector<f64>,
    pub label: Vector<f64>
}

impl MnistData {

    pub fn load_data(label_path: &str, image_path: &str) -> Result<(), Error> {

        let label_contents = MnistData::load_contents(label_path);
        let image_contents = MnistData::load_contents(image_path);


        Ok(())
    }

    fn load_contents(path: &str) -> Result<(), Error> {
        let file = File::open(path)?;
        let mut gz = GzDecoder::new(file);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents);

        Ok(())

    }
}