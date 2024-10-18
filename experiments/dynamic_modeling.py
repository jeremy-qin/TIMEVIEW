from scipy.interpolate import BSpline
import numpy as np

def fit_cubic_spline_for_feature(t, feature_values, knots, eval_at='midpoints'):

    k = 3  

    expected_num_coefficients = len(knots) - k - 1

    bspline_basis = BSpline(knots, np.eye(expected_num_coefficients), k)

    if feature_values.ndim == 1:
        feature_values = feature_values[:, np.newaxis]

    bspline_matrix = bspline_basis(t)

    coefficients, _, _, _ = np.linalg.lstsq(bspline_matrix, feature_values, rcond=None)

    spline = BSpline(knots, coefficients.ravel(), k)

    if eval_at == 'knots':
        eval_points = knots 
    elif eval_at == 'midpoints':
        eval_points = (knots[:-1] + knots[1:]) / 2  

    first_derivative = spline.derivative(nu=1)
    second_derivative = spline.derivative(nu=2)

    first_deriv_values = first_derivative(eval_points)
    second_deriv_values = second_derivative(eval_points)

    return spline, first_deriv_values, second_deriv_values


def fit_cubic_splines_for_all_samples(ts_dynamic, feature_values, knots, eval_at='midpoints'):

    splines = []
    first_derivatives = []
    second_derivatives = []

    n_samples = feature_values.shape[0]  
    
    for i in range(n_samples):
        t = ts_dynamic[i] 
        y = feature_values[i] 

        try:
            spline, first_deriv_values, second_deriv_values = fit_cubic_spline_for_feature(t, y, knots, eval_at=eval_at)
        except ValueError as e:
            print(f"Error calculating derivatives for sample {i}: {e}")
            first_deriv_values = np.zeros(len(knots) - 1)
            second_deriv_values = np.zeros(len(knots) - 1)

        splines.append(spline)
        first_derivatives.append(first_deriv_values)
        second_derivatives.append(second_deriv_values)
    
    return splines, first_derivatives, second_derivatives

def encode_motif_numerical(f_prime, f_double_prime):

    if f_prime > 0 and f_double_prime == 0:
        return 1  # Positive slope
    elif f_prime < 0 and f_double_prime == 0:
        return -1  # Negative slope
    elif f_prime == 0 and f_double_prime == 0:
        return 0  # Zero slope
    elif f_prime > 0 and f_double_prime > 0:
        return 2  # Increasing, convex
    elif f_prime > 0 and f_double_prime < 0:
        return 3  # Increasing, concave
    elif f_prime < 0 and f_double_prime > 0:
        return -2  # Decreasing, convex
    elif f_prime < 0 and f_double_prime < 0:
        return -3  # Decreasing, concave
    else:
        raise ValueError("Unrecognized combination of derivatives.")

def encode_dynamic_feature_as_single_vector(f_prime_intervals, f_double_prime_intervals):

    n_intervals = len(f_prime_intervals)
    feature_matrix = np.zeros((n_intervals, 3))

    for i in range(n_intervals):
        motif_numerical = encode_motif_numerical(f_prime_intervals[i], f_double_prime_intervals[i])
        magnitude_f_prime = np.abs(f_prime_intervals[i])
        magnitude_f_double_prime = np.abs(f_double_prime_intervals[i])
        feature_matrix[i] = [motif_numerical, magnitude_f_prime, magnitude_f_double_prime]

    return feature_matrix

def get_regular_knots(n_time_steps, n_intervals):
    return np.linspace(0, n_time_steps, n_intervals + 4)

def calculate_transition_properties(spline, knots, n_intervals):

    properties = []

    for i in range(n_intervals - 1):
        property_value = spline(knots[i])
        properties.append(property_value)

    last_property_value = spline(knots[-1])
    properties.append(last_property_value)

    return properties