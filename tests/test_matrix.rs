use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrix;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let b = Matrix::new(2, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);

        let row = b.row(0).raw_slice();
        let expected = vec![1.0, 2.0, 3.0];
        
        assert_eq!(row, expected);
    }
}