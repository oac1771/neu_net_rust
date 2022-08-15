use rulinalg::vector::Vector;

pub struct Data;
    // pub training_data: Vector<f32>,
    // pub training_labels: Vector<f32>

impl Data {

    pub fn generate_test_data(&self, training_iterations: usize, dataset_length: usize) 
    -> impl Iterator<Item = Vec<Vec<Data>>> {
        
    }
}

#[cfg(test)]
#[path = "../test/test_data.rs"]
mod test;