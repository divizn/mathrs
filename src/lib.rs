use pyo3::prelude::*;

const EPSILON: f64 = 1e-10; // Tolerance for floating-point comparisons
const HALF_PI: f64 = std::f64::consts::FRAC_PI_2; // π/2 (90 degrees)
const THREE_HALF_PI: f64 = 3.0 * HALF_PI; // 3π/2 (270 degrees)


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
fn sum_list(list: Vec<isize>) -> isize {
    list.iter().sum()
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
fn double_list(list: Vec<isize>) -> Vec<isize> {
    list.iter().map(|x| x * 2).collect()
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
fn double(n: isize) -> isize {
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
/// mathrs.sin(math.pi)  # Returns 0.0
/// mathrs.sin(180, degrees=True)  # Also returns 0.0
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
/// 
/// # Arguments
/// * `n` - Input number (radians by default).
/// * `degrees` - Set to `true` if `n` is in degrees (optional, default `false`).
/// 
/// # Example
/// ```python
/// mathrs.cos(math.pi)  # Returns -1.0
/// mathrs.cos(180, degrees=True)  # Also returns -1.0
/// mathrs.cos(0)  # Returns 1.0
/// mathrs.cos(0, degrees=True)  # Also returns 1.0
/// ```
#[pyfunction]
#[pyo3(signature = (n, degrees=false))]
fn cos(n: f64, degrees: bool) -> f64 {
    let radians = if degrees {
        n.to_radians()
    } else {
        n
    };

    // Handle special cases for the cosine function
    let result = radians.cos();

    // Handle cases where result should be close to zero
    if result.abs() < EPSILON {
        return 0.0; // Return zero for values very close to zero
    } else {
        return result; // Return the computed result
    }
}

/// Finds the Tangent value of a number, with an optional `degrees` argument (default is radians).
/// 
/// Due to floating-point precision, results for values like `tan(math.pi/2)` may not be exactly infinity, however, we handle this case and treat it as infinity
/// (e.g., `1.633123935319537e16`). This large difference is expected and is treated as infinity for practical purposes.
/// Floating-point precision can also cause small values to be returned instead of zero - we use a tolerance to handle this (the same for others).
/// 
/// # Arguments
/// * `n` - Input number (radians by default).
/// * `degrees` - Set to `true` if `n` is in degrees (optional, default `false`).
/// 
/// # Example
/// ```python
/// mathrs.tan(math.pi/2)  # Returns inf
/// mathrs.tan(90, degrees=True)  # Also returns a inf
/// mathrs.tan(0)  # Returns 0.0
/// mathrs.tan(0, degrees=True)  # Also returns 0.0
/// mathrs.tan(270, degrees=True)  # Returns negative inf
/// ```
#[pyfunction]
#[pyo3(signature = (n, degrees=false))]
fn tan(n: f64, degrees: bool) -> f64 {
    let radians = if degrees {
        n.to_radians()
    } else {
        n
    };

    // Check for angles near odd multiples of π/2 (90°, 270°, etc.) where tangent is infinite
    if (radians - HALF_PI).abs() < EPSILON {
        return f64::INFINITY; // Approaching π/2 (90°)
    } else if (radians - THREE_HALF_PI).abs() < EPSILON {
        return f64::NEG_INFINITY; // Approaching 3π/2 (270°)
    }

    let result = radians.tan();

    if result.abs() < EPSILON {
        return 0.0; // Handle cases where result should be close to zero
    } else {
        return result; // Return the computed result
    }
}



/// Computes the Rectified Linear Unit (ReLU) of a number.
/// 
/// # Arguments
/// * `n` - Input number.
/// 
/// # Example
/// ```python
/// mathrs.relu(-1)  # Returns 0.0
/// mathrs.relu(1)  # Returns 1.0
/// ```
#[pyfunction]
#[pyo3(signature = (n))]
fn relu(n: f64) -> f64 { 
    if n > 0.0 {
        n
    } else {
        0.0
    }
}

/// Computes the Sigmoid activation function of a number.
/// Maps any real value to the range [0, 1].
/// 
/// # Arguments
/// * `n` - Input number.
/// 
/// # Example
/// ```python
/// mathrs.sigmoid(0)  # Returns 0.5
/// mathrs.sigmoid(1)  # Returns ~0.731
/// mathrs.sigmoid(-1)  # Returns ~0.268
/// mathrs.sigmoid(10)  # Returns ~0.999
/// ```
#[pyfunction]
fn sigmoid(n: f64) -> f64 {
    1.0 / (1.0 + (-n).exp())
}

/// Computes the Softmax activation function of a list of numbers.
/// Maps a list of real values to the range [0, 1] such that the sum of the values is 1.
/// 
/// # Arguments
/// * `list` - Input list of numbers.
/// 
/// # Example
/// ```python
/// mathrs.softmax([1, 2, 3])  # Returns [~0.09, ~0.24, ~0.67]
/// ```
#[pyfunction]
#[pyo3(signature = (list))]
fn softmax(list: Vec<f64>) -> Vec<f64> {
    let sum: f64 = list.iter().map(|x| x.exp()).sum();
    list.iter().map(|x| x.exp() / sum).collect()
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
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_double() {
        assert_eq!(double(2), 4);
        assert_eq!(double(-2), -4);
        assert_eq!(double(0), 0);
    }
    
    #[test]
    fn test_sum_list() {
        assert_eq!(sum_list(vec![1, 2, 3]), 6);
        assert_eq!(sum_list(vec![0, 0, 0]), 0);
        assert_eq!(sum_list(vec![-2, -4, -2]), -8);
    }
    
    #[test]
    fn test_double_list() {
        assert_eq!(double_list(vec![1, 2, 3]), vec![2, 4, 6]);
        assert_eq!(double_list(vec![0, 0, 0]), vec![0, 0, 0]);
        assert_eq!(double_list(vec![-2, -4, -2]), vec![-4, -8, -4]);
    }
    
    #[test]
    fn test_sqrt() {
        assert_eq!(sqrt(4.0), 2.0);
        assert_eq!(sqrt(0.0), 0.0);
    }

    
    #[test]
    fn test_sin() {
        assert_eq!(sin(0.0, false), 0.0);
        assert_eq!(sin(0.0, true), 0.0);
        assert_eq!(sin(HALF_PI, false), 1.0);
        assert_eq!(sin(90.0, true), 1.0);
        assert_eq!(sin(THREE_HALF_PI, false), -1.0);
        assert_eq!(sin(270.0, true), -1.0);
    }

    #[test]
    fn test_cos() {
        assert_eq!(cos(0.0, false), 1.0);
        assert_eq!(cos(0.0, true), 1.0);
        assert_eq!(cos(HALF_PI, false), 0.0);
        assert_eq!(cos(90.0, true), 0.0);
        assert_eq!(cos(THREE_HALF_PI, false), 0.0);
        assert_eq!(cos(270.0, true), 0.0);
    }

    #[test]
    fn test_tan() {
        assert_eq!(tan(0.0, false), 0.0);
        assert_eq!(tan(0.0, true), 0.0);
        assert_eq!(tan(HALF_PI, false), f64::INFINITY);
        assert_eq!(tan(90.0, true), f64::INFINITY);
        assert_eq!(tan(THREE_HALF_PI, false), f64::NEG_INFINITY);
        assert_eq!(tan(270.0, true), f64::NEG_INFINITY);
    }
    
    #[test]
    fn test_relu() {
        assert_eq!(relu(3.5), 3.5);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(-1.2), 0.0);
    }
    
    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!((sigmoid(1.0) - 0.731058).abs() < 1e-6);
        assert!((sigmoid(-1.0) - 0.268941).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let scores = vec![1.0, 2.0, 3.0];
        let expected = vec![0.09003057, 0.24472847, 0.66524096]; // Normalized probabilities
        let result = softmax(scores);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }
}
