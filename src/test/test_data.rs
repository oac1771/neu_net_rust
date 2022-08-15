use crate::builder::data::Data;


#[test]
fn successfull_generation_of_test_data() {
    let data = Data{};
    let training_iterations = 100;
    let dataset_length = 50;

    let test_data = data.generate_test_data(training_iterations, dataset_length);
    // .chain() to combine two iterators
    assert_eq!(test_data.count(), training_iterations as usize);


}