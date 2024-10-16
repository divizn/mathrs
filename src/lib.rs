use pyo3::prelude::*;

const EPSILON: f64 = 1e-10; // Tolerance for floating-point comparisons


/// Sums a list of numbers.
/// 
/// # Arguments
/// * `list` - Input list of numbers.
/// 
/// # Example
/// ```python
/// mathrs.sum_list([1, 2, 3])  # Returns 6
/// ```
#[pyfunction]
fn sum_list(list: Vec<usize>) -> PyResult<usize> {
    Ok(list.iter().sum())
}

/// Doubles numbers in a list and returns the doubled list
/// 
/// # Arguments
/// * `list` - Input list of numbers.
/// 
/// # Example
/// ```python
/// mathrs.double_list([1, 2, 3])  # Returns [2, 4, 6]
/// ```
#[pyfunction]
fn double_list(list: Vec<usize>) -> PyResult<Vec<usize>> {
    Ok(list.iter().map(|x| x * 2).collect())
}

/// Doubles a number
/// 
/// # Arguments
/// * `n` - Input number.
/// 
/// # Example
/// ```python
/// mathrs.double(4)  # Returns 8
/// ```
#[pyfunction]
fn double(n: usize) -> usize {
    n * 2
}

/// Finds the square root of a number
/// 
/// # Arguments
/// * `n` - Input number.
/// 
/// # Example
/// ```python
/// mathrs.sqrt(4)  # Returns 2.0
/// ```
#[pyfunction]
fn sqrt(n: f64) -> f64 {
    n.sqrt()
}

/// Finds the sine of a number, with an optional `degrees` argument (default is radians).
/// 
/// Due to floating-point precision, results for values like `sin(math.pi)` may not be exactly 0 
/// (e.g., `1.2246467991473532e-16`). This small difference is expected and can be treated as zero 
/// for practical purposes.
/// 
/// # Arguments
/// * `n` - Input number (radians by default).
/// * `degrees` - Set to `true` if `n` is in degrees (optional, default `false`).
///
/// # Example
/// ```python
/// mathrs.sin(math.pi)  # Returns a value close to 0
/// mathrs.sin(180, degrees=True)  # Also returns a value close to 0
/// ```

#[pyfunction]
#[pyo3(signature = (n, degrees=false))]
fn sin(n: f64, degrees: bool) -> f64 {
    let result = if degrees {
        n.to_radians().sin()
    } else {
        n.sin()
    };

    if result.abs() < EPSILON {
        0.0 // Treat small values as 0
    } else {
        result
    }
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
