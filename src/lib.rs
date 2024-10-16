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

/// Finds the Sine value of a number
#[pyfunction]
fn sin(n: f64) -> f64 {
    n.sin()
}

/// Finds the Cosine value of a number
#[pyfunction]
fn cos(n: f64) -> f64 {
    n.cos()
}

/// Finds the Tangent value of a number
#[pyfunction]
fn tan(n: f64) -> f64 {
    n.tan()
}

// TODO: implement properly
/// Applies a ReLU activation function to a number
#[pyfunction]
fn relu(n: f64) -> f64 { 
    if n > 0.0 {
        n
    } else {
        0.0
    }
}

/// Applies a Sigmoid activation function to a number
#[pyfunction]
fn sigmoid(n: f64) -> f64 {
    1.0 / (1.0 + (-n).exp())
}

/// A Python math module implemented in Rust.
#[pymodule]
fn mathrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)?;
    m.add_function(wrap_pyfunction!(sum_list, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(double_list, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    Ok(())
}
