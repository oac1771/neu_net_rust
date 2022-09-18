use std::io;
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

    pub fn load_data(path: &str) -> Result<(), Error> {

        let file_result = File::open(path);
        let file  = match file_result {
            Ok(file) => file,
            Err(error) => panic!("Could not open file: {:?}", error)
        };

        let d = GzDecoder::new(file);

        for line in io::BufReader::new(d).lines() {
            println!("{}", line.unwrap());
        }
        Ok(())
    }
}