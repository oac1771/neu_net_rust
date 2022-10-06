use ndarray::Array2;

pub trait Data {
    fn get_data(&self) -> &Array2<f64>;
    fn get_label(&self) -> &Array2<f64>;
}