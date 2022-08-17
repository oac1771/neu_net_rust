use rulinalg::vector::Vector;
use rulinalg::vector;

pub struct Data {
    pub training_data: Vector<f32>,
    pub training_labels: Vector<f32>
}

pub struct DataIter{
    pub data: Data,
    pub count: i32,
    pub training_iterations: i32
}
// I actually think this would be a list of iterators, each iterator generates test data object
// https://stackoverflow.com/questions/16421033/lazy-sequence-generation-in-rust
// https://medium.com/journey-to-rust/iterators-in-rust-a73560f796ee

impl Data {

//     pub fn generate_test_data(&self, training_iterations: usize, dataset_length: usize) 
//     -> Vec<Box<dyn Iterator<Item=Vec<Data>>>>{

//         let mut data = vec![];

//         for i in 0..dataset_length {
//             data.push(new())
//         }

//         let result = vec![vec![&data; training_iterations]].iter();

//         return result
//     }

    pub fn new() -> Data {
        return Data{
            training_data: vector![1.0, 0.0, 0.0, 1.0],
            training_labels: vector![1.0, 0.0],
        }
    }
}

// This eventually should return Vec<Data>
impl Iterator for DataIter {
    type Item=Data;
    fn next(&mut self) -> Option<Self::Item>{
        if self.count < self.training_iterations {
            self.count+=1;
            Some(Data::new())
        } else {
            None
        }
        
    }
}

#[cfg(test)]
#[path = "../test/test_data.rs"]
mod test;