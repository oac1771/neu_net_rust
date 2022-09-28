use ndarray::Array1;

pub trait Data {
    fn get_data(&self) -> &Array1<f64>;
    fn get_label(&self) -> &Array1<f64>;
}