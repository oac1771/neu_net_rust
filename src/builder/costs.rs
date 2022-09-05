use rulinalg::vector::Vector;

pub trait CostFunction {
    fn cost(&self, label: &Vector<f64>, output: &Vector<f64>) -> Vector<f64>;
    fn dcostdact(&self, label: &Vector<f64>, output: &Vector<f64>, data_set_length: f64) -> Vector<f64>;
}

pub struct Quadratic;

impl CostFunction for Quadratic {
    fn cost(&self, label: &Vector<f64>, output: &Vector<f64>) -> Vector<f64> {

        let result = (label - output).into_iter().map(|x| {
            x.powf(2.0) / 2.0
        }).collect();

        return result
    }

    fn dcostdact(&self, label: &Vector<f64>, output: &Vector<f64>, data_set_length: f64) -> Vector<f64> {

        let result = (output - label).into_iter().map(|x| {
            x / data_set_length
        }).collect();
        return result
    }
}


#[cfg(test)]
#[path = "../test/test_costs.rs"]
mod test;