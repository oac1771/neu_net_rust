use crate::neu_net::builder::builder::Builder;
use crate::neu_net::builder::data::Data;

use ndarray::Array2;
use ndarray::array;


pub struct TestData {
    pub data: Array2<f64>,
    pub label: Array2<f64>
}

impl Data for TestData {
    fn get_data(&self) -> &Array2<f64> {
        return &self.data
    }

    fn get_label(&self) -> &Array2<f64> {
        return &self.label
    }
}

fn generate_test_data<'a>(data_set_length: i32, data: &'a Array2<f64>, label: &'a Array2<f64>) -> Vec<impl Data> {
    let mut data_suite = Vec::new();
    
    for _ in 0..data_set_length {
        
        data_suite.push(TestData{
            data: data.clone(),
            label: label.clone()
        })
    }

    return data_suite
}

#[test]
fn succesfull_evaluation_of_input() {
    let layer_nodes = vec![4, 3, 2];
    let neu_net = Builder::build(&layer_nodes);
    let input = Array2::from_shape_fn((neu_net.layer_nodes[0], 1) , |_i|{100.0});
    let result = neu_net.evaluate(&input);

    assert_eq!(result.into_iter().len(), 2);
}

#[test]
fn successfull_train_of_network() {
    let learning_rate = 100.0;
    let training_iterations = 250;
    let data_set_length = 250;

    let data_suite_data = array![[1.0], [1.0], [0.0], [0.0]];
    let data_suite_label = array![[1.0], [0.0]];
    let data_suite_2_data = array![[0.0], [0.0], [1.0], [1.0]];
    let data_suite_2_label = array![[0.0], [1.0]];
 
    let mut data_suite = generate_test_data(data_set_length, &data_suite_data, &data_suite_label);
    let data_suite_2 = generate_test_data(data_set_length, &data_suite_2_data, &data_suite_2_label);
    data_suite.extend(data_suite_2);


    let layer_nodes = vec![4, 3, 2];
    let mut neu_net = Builder::build(&layer_nodes);
    neu_net.train(&data_suite, training_iterations, learning_rate);

    assert!((data_suite_label - neu_net.evaluate(&data_suite_data)).sum() < 0.01);
    assert!((data_suite_2_label - neu_net.evaluate(&data_suite_2_data)).sum() < 0.01);
    // assert_eq!(data_suite_label, neu_net.evaluate(&data_suite_data));
}