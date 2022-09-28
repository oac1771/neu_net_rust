use ndarray::Array1;

pub trait CostFunction {
    fn cost(&self, label: &Array1<f64>, output: &Array1<f64>) -> Array1<f64>;
    fn dcostdact(&self, label: &Array1<f64>, output: &Array1<f64>) -> Array1<f64>;
}

pub struct Quadratic;

impl CostFunction for Quadratic {
    fn cost(&self, label: &Array1<f64>, output: &Array1<f64>) -> Array1<f64> {

        let result = (label - output).map(|x| {
            x.powf(2.0) / 2.0
        });

        return result
    }

    fn dcostdact(&self, label: &Array1<f64>, output: &Array1<f64>) -> Array1<f64> {

        let result = output - label;
        return result
    }
}


#[cfg(test)]
#[path = "./../test/test_costs.rs"]
mod test;