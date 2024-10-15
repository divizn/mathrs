use pyo3::prelude::*;

/// Sums a list of numbers.
#[pyfunction]
fn sum_list(list: Vec<usize>) -> PyResult<usize> {
    Ok(list.iter().sum())
}

/// Doubles numbers in a list and returns the doubled list
#[pyfunction]
fn double_list(list: Vec<usize>) -> PyResult<Vec<usize>> {
    Ok(list.iter().map(|x| x * 2).collect())
}

/// Doubles a number
#[pyfunction]
fn double(n: usize) -> usize {
    n * 2
}

/// Finds the square root of a number
#[pyfunction]
fn sqrt(n: f64) -> f64 {
    n.sqrt()
}


/// A Python math module implemented in Rust.
#[pymodule]
fn mathrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)?;
    m.add_function(wrap_pyfunction!(sum_list, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(double_list, m)?)?;
    Ok(())
}
