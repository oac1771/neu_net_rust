use crate::builder::builder::Builder;
use crate::builder::data::Data;

use rulinalg::vector::Vector;
use rulinalg::vector;

fn generate_test_data(data_set_length: i32, data: &Vector<f64>, label: &Vector<f64>) -> Vec<Data> {
    let mut data_suite = Vec::new();

    for _ in 0..data_set_length {
        data_suite.push(Data{
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
    let input = vector![10000.0; neu_net.layer_nodes[0]];
    let result = neu_net.evaluate(&input);

    assert_eq!(result.into_iter().len(), 2);
}

#[test]
fn successfull_train_of_network() {
    let learning_rate = 0.5;
    let training_iterations = 250;
    let data_set_length = 250;

    let data_suite_data = vector![1.0, 1.0, 0.0, 0.0];
    let data_suite_label = vector![1.0, 0.0];
    let data_suite_2_data = vector![0.0, 0.0, 1.0, 1.0];
    let data_suite_2_label = vector![0.0, 1.0];
 
    let mut data_suite = generate_test_data(data_set_length, &data_suite_data, &data_suite_label);
    let data_suite_2 = generate_test_data(data_set_length, &data_suite_2_data, &data_suite_2_label);
    data_suite.extend(data_suite_2);


    let layer_nodes = vec![4, 3, 2];
    let mut neu_net = Builder::build(&layer_nodes);
    neu_net.train(data_suite, training_iterations, learning_rate);

    // println!("foooo {:?}", neu_net.evaluate(&data_suite_data))

    // assert_eq!(neu_net.evaluate(&data_suite_data), data_suite_label)

}