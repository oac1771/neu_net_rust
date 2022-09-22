use rulinalg::vector::Vector;

pub trait Data {
    fn get_data(&self) -> &Vector<f64>;
    fn get_label(&self) -> &Vector<f64>;
}