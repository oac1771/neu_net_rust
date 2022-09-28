use std::io::Error;
use std::io::prelude::*;
use ndarray::Array1;
use flate2::read::GzDecoder;
use std::fs::File;
use neural_network::neu_net::builder::data::Data;


const START_LABEL_DATA_INDEX: usize = 8;
const LENGTH_OF_LABEL_OUTPUT_VECTOR: usize = 10;
const START_IMAGE_DATA_INDEX: usize = 16;
const NUMBER_ELEMENTS_IN_IMAGE: usize = 784;
const MAX_IMAGE_ELEMENT_VALUE: f64 = 255.0;

pub struct MnistData {
    pub data: Array1<f64>,
    pub label: Array1<f64>
}

impl Data for MnistData {
    fn get_data(&self) -> &Array1<f64> {
        return &self.data
    }

    fn get_label(&self) -> &Array1<f64> {
        return &self.label
    }
}

impl MnistData {

    pub fn load_data(label_path: &str, image_path: &str) -> Result<Vec<MnistData>, Error> {

        let label_contents = MnistData::load_contents(label_path)?;
        let image_contents = MnistData::load_contents(image_path)?;

        let mnist_data = MnistData::process(&label_contents, &image_contents);
        Ok(mnist_data)
    }

    fn load_contents(path: &str) -> Result<Vec::<u8>, Error> {

        let file = File::open(path)?;
        let mut gz = GzDecoder::new(file);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;

        return Ok(contents);

    }

    fn process(label_contents: &Vec<u8>, image_contents: &Vec<u8>) -> Vec<MnistData> {
        let cleaned_label_contents = &label_contents[START_LABEL_DATA_INDEX..];
        let data_set_length = cleaned_label_contents.len();
        let cleaned_image_contents = &image_contents[START_IMAGE_DATA_INDEX..]
            .to_vec()
            .chunks(NUMBER_ELEMENTS_IN_IMAGE)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<u8>>>();

        let mut mnist_data: Vec<MnistData> = Vec::with_capacity(data_set_length);
        let mut label: Array1<f64>;
        let mut data: Array1<f64>;

        for index in 0..data_set_length {
            label = MnistData::process_label(&cleaned_label_contents[index]);
            data = MnistData::process_image(&cleaned_image_contents[index]);
            mnist_data.push(MnistData{label, data});
        }

        return mnist_data
    }

    fn process_label(label: &u8) -> Array1<f64> {
        let mut label_vector = Array1::<f64>::zeros(LENGTH_OF_LABEL_OUTPUT_VECTOR);
        label_vector[*label as usize] = 1.0;

        return label_vector;

    }

    fn process_image(image: &Vec<u8>) -> Array1<f64> {
        let image_vector = image.into_iter().map(|x| *x as f64 / MAX_IMAGE_ELEMENT_VALUE).collect::<Array1<f64>>();
        return image_vector

    }
}