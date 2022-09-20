use std::io::Error;
use std::io::prelude::*;
use rulinalg::vector::Vector;
use rulinalg::vector;
use flate2::read::GzDecoder;
use std::fs::File;

const START_LABEL_DATA_INDEX: usize = 8;
const LENGTH_OF_LABEL_OUTPUT_VECTOR: usize = 10;
const START_IMAGE_DATA_INDEX: usize = 16;

pub struct MnistData {
    // pub data: Vector<f64>,
    pub label: Vector<f64>
}

impl MnistData {

    pub fn new() -> MnistData {
        return MnistData{
            label: vector![]
        };
    }
    pub fn load_data(label_path: &str, image_path: &str) -> Result<(), Error> {

        let label_contents = MnistData::load_contents(label_path)?;
        let image_contents = MnistData::load_contents(image_path)?;

        let _ = MnistData::process(&label_contents, &image_contents);
        Ok(())
    }

    fn load_contents(path: &str) -> Result<Vec::<u8>, Error> {

        let file = File::open(path)?;
        let mut gz = GzDecoder::new(file);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;

        return Ok(contents);

    }

    fn process(label_contents: &Vec<u8>, image_contents: &Vec<u8>) {
        let striped_label_contents = &label_contents[START_LABEL_DATA_INDEX..];
        let _striped_image_contents = &image_contents[START_IMAGE_DATA_INDEX..];

        let mut mnist_data: Vec<MnistData> = Vec::with_capacity(striped_label_contents.len());
        let mut label: Vector<f64>;

        for label_value in striped_label_contents {
            label = MnistData::process_label(label_value);
            mnist_data.push(MnistData { label });

        }
    }

    fn process_label(label: &u8) -> Vector<f64> {
        let mut label_vector = Vector::<f64>::zeros(LENGTH_OF_LABEL_OUTPUT_VECTOR);
        label_vector[*label as usize] = 1.0;

        return label_vector;

    }
}