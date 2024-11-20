import numpy as np

import classic_models
import settings
import foo_utils

class Measurement:
    def __init__(self, ts : int, coord : list[float], source : int):
        self.ts = ts 
        self.coord = coord 
        self.source = source


class MeasurementSet:
    def __init__(self, initial_size=None):
        if initial_size is None:
            self.measurements = []
        else:
            self.measurements = [None for _ in range(initial_size)]

    def include_measurement(self, measurement : Measurement):
        self.measurements.append(measurement)

    def filter_by_source(self, source):
        result = MeasurementSet(len(self.measurements))

        for m in self.measurements:
            if m.source == source:
                result.include_measurement(m)

        return result 

    def get_raw_data(self):
        data = [None for _ in range(len(self.measurements))]

        for m in self.measurements:
            if m is None:
                continue
            if data[m.ts] is None:
                data[m.ts] = []
            data[m.ts].append(m.coord)
        
        for i in range(len(data)):
            data[i] = np.array(data[i])
        
        return data 

    def get_uncluttered_data(self):
        return self.filter_by_source(1).get_raw_data()
    
    def get_clutter_history(self):
        return self.filter_by_source(0).get_raw_data()


def generate_ar_parameters(params : settings.DataGenerationParameters, num_integrators=0):
    process_coefficients = np.zeros((params.simulation_time, params.ar_generation_params.process_order))

    is_real_pole = np.zeros(params.ar_generation_params.process_order)
    polynomial_roots = np.zeros(params.ar_generation_params.process_order, dtype=complex)
    if num_integrators > 0:
        polynomial_roots[-num_integrators:] += 1
        is_real_pole[-num_integrators:] = True
    root_ind = 0
    while root_ind < params.ar_generation_params.process_order - num_integrators:
        add_real_pole = (params.ar_generation_params.process_order - root_ind == 1) or (np.random.uniform() < 0.25 and False)

        damping = params.ar_generation_params.min_damping + np.random.gamma(shape=params.ar_generation_params.damping_gamma_shape, scale=params.ar_generation_params.damping_gamma_scale)
        damping = min(damping, params.ar_generation_params.max_damping)

        # Note that the phase could be PI as well for negative real roots, but these are being ignored for convenience
        phase = 0 if add_real_pole else np.random.uniform(low=0, high=params.ar_generation_params.phase_uniform_max_pi_factor * np.pi)
        new_pole = np.exp(-damping + 1j * phase)

        if add_real_pole:
            is_real_pole[root_ind] = 1
            polynomial_roots[root_ind] = new_pole
            root_ind += 1
        else:
            polynomial_roots[root_ind] = new_pole 
            polynomial_roots[root_ind + 1] = np.conj(new_pole)
            root_ind += 2 

    polynomial_coefficients = np.real(np.polynomial.polynomial.polyfromroots(polynomial_roots))
    process_coefficients[0,:] = -polynomial_coefficients[0:-1][::-1]

    time_varying_scale = params.ar_generation_params.walk_scale if params.ar_generation_params.time_varying else 0

    for current_timestep in range(1, params.simulation_time):
        root_ind = 0

        while root_ind < params.ar_generation_params.process_order - num_integrators:
            log_root = np.log(polynomial_roots[root_ind])
            damping = -np.real(log_root)
            phase = np.imag(log_root)
            damping = np.random.normal(loc=damping, scale=time_varying_scale)

            damping = max(damping, params.ar_generation_params.min_damping)
            damping = min(damping, params.ar_generation_params.max_damping)

            if is_real_pole[root_ind]:
                new_pole = np.exp(-damping + 1j * phase)
                polynomial_roots[root_ind] = new_pole 
                root_ind += 1
            else:
                phase = np.random.normal(loc=phase, scale=time_varying_scale * np.pi)

                phase = min(phase, params.ar_generation_params.phase_uniform_max_pi_factor * np.pi)
                phase = max(phase, -params.ar_generation_params.phase_uniform_max_pi_factor * np.pi)

                new_pole = np.exp(-damping + 1j * phase)

                polynomial_roots[root_ind] = new_pole 
                polynomial_roots[root_ind + 1] = np.conj(new_pole)
                root_ind += 2

        polynomial_coefficients = np.real(np.polynomial.polynomial.polyfromroots(polynomial_roots))
        process_coefficients[current_timestep,:] = -polynomial_coefficients[0:-1][::-1]

    return process_coefficients 


def generate_smooth_ar_parameters(params : settings.DataGenerationParameters):
    process_coefficients = np.zeros((params.simulation_time, params.ar_generation_params.process_order))

    polynomial_roots = np.zeros(params.ar_generation_params.process_order, dtype=complex)
    root_arguments = np.zeros(params.ar_generation_params.process_order)
    max_radian_range = 2 * np.pi * params.ar_generation_params.phase_uniform_max_pi_factor
    damping_factor = np.exp(-params.ar_generation_params.max_damping)
    variation_scale = max_radian_range / params.ar_generation_params.process_order
    variation_frequency = 2 * np.pi / 100

    for current_timestep in range(params.simulation_time):
        if params.ar_generation_params.process_order % 2 == 1:
            root_arguments = np.arange(-(params.ar_generation_params.process_order // 2), params.ar_generation_params.process_order // 2 + 1)
        else:
            root_arguments[params.ar_generation_params.process_order // 2:] = np.arange(1, params.ar_generation_params.process_order // 2 + 1)
            root_arguments[:params.ar_generation_params.process_order // 2] = -np.arange(1, params.ar_generation_params.process_order // 2 + 1)
        root_arguments = root_arguments.astype(np.float64)
        
        root_arguments += np.sin(current_timestep * variation_frequency) * (variation_scale / 2) * np.sign(root_arguments).astype(np.float64)

        polynomial_roots = damping_factor * np.exp(root_arguments * 1j * max_radian_range / params.ar_generation_params.process_order)

        polynomial_coefficients = np.real(np.polynomial.polynomial.polyfromroots(polynomial_roots))
        process_coefficients[current_timestep,:] = -polynomial_coefficients[0:-1][::-1]

    return process_coefficients 


def generate_smooth_ojsp_ar_parameters(params : settings.DataGenerationParameters):
    process_coefficients = np.zeros((params.simulation_time, params.ar_generation_params.process_order))

    polynomial_roots = np.zeros(params.ar_generation_params.process_order, dtype=complex)
    root_arguments = np.zeros(params.ar_generation_params.process_order)
    max_radian_range = 2 * np.pi * params.ar_generation_params.phase_uniform_max_pi_factor
    damping_factor = np.exp(-params.ar_generation_params.max_damping)
    variation_scale = 0.8
    variation_frequency = 1 * np.pi / 100

    for current_timestep in range(params.simulation_time):
        if params.ar_generation_params.process_order % 2 == 1:
            root_arguments = np.arange(-(params.ar_generation_params.process_order // 2), params.ar_generation_params.process_order // 2 + 1)
        else:
            root_arguments[params.ar_generation_params.process_order // 2:] = np.arange(1, params.ar_generation_params.process_order // 2 + 1)
            root_arguments[:params.ar_generation_params.process_order // 2] = -np.arange(1, params.ar_generation_params.process_order // 2 + 1)
        root_arguments = root_arguments.astype(np.float64)
        
        root_arguments -= np.cos(current_timestep * variation_frequency) * (variation_scale) * np.sign(root_arguments).astype(np.float64)

        root_arguments *= max_radian_range / params.ar_generation_params.process_order

        polynomial_roots = np.where(root_arguments == 0, 1, damping_factor) * np.exp(root_arguments * 1j)

        polynomial_coefficients = np.real(np.polynomial.polynomial.polyfromroots(polynomial_roots))
        process_coefficients[current_timestep,:] = -polynomial_coefficients[0:-1][::-1]

    return process_coefficients 




def generate_ar_process_instance(process_coefficients : np.ndarray, params : settings.DataGenerationParameters):
    data = np.zeros((params.simulation_time, params.observation_dimensions))

    for current_timestep in range(1, params.simulation_time):
        innovation = np.random.normal(loc=0, scale=np.sqrt(params.ar_generation_params.innovation_variance), size=params.observation_dimensions)
        if current_timestep <= params.ar_generation_params.process_order:
            data[current_timestep, :] = innovation
        else:
            data[current_timestep, :] = data[current_timestep - params.ar_generation_params.process_order : current_timestep,:].T @ process_coefficients[current_timestep,:][::-1].T + innovation
    return data 


def add_observation_noise(noiseless_data : np.ndarray, params : settings.DataGenerationParameters):
    return noiseless_data + np.random.normal(loc=0, scale=np.sqrt(params.measurement_params.noise_variance), size=noiseless_data.shape)


def generate_ojsp_dataset(data_generation_params : settings.DataGenerationParameters, num_integrators=0):
    # Note that the leading coefficient affects the most recent sample
    if data_generation_params.ar_generation_params.oscillating:
        true_process_coefficients = generate_smooth_ojsp_ar_parameters(data_generation_params)
    else:
        true_process_coefficients = generate_ar_parameters(data_generation_params, num_integrators)

    ground_truth_position = generate_ar_process_instance(true_process_coefficients, data_generation_params)
    uncluttered_data = add_observation_noise(ground_truth_position, data_generation_params)

    min_boundary, max_boundary = foo_utils.get_bounding_box(uncluttered_data, 0.1)
    clutter_history = [generate_poisson_samples(data_generation_params.measurement_params.clutter_rate, [min_boundary[0], max_boundary[0]], [min_boundary[1], max_boundary[1]]) for _ in range(data_generation_params.simulation_time)]

    true_measurement_observed = np.random.uniform(size=data_generation_params.simulation_time) < data_generation_params.measurement_params.observation_probability

    data = MeasurementSet()
    for ts in range(data_generation_params.simulation_time):
        if true_measurement_observed[ts]:
            data.include_measurement(Measurement(ts, list(uncluttered_data[ts]), 1))

    for ts, clutter in enumerate(clutter_history):
        for m in list(clutter):
            data.include_measurement(Measurement(ts, list(m), 0))

    return data, ground_truth_position, true_process_coefficients



def generate_real_cv_dataset(data_generation_params : settings.DataGenerationParameters):
    state = np.array([0, 0, 0, 0], dtype="float")
    state[0] = np.random.uniform(low=-100, high=0)
    state[2] = np.random.uniform(low=-100, high=0)

    state[1] = np.random.uniform(low=0, high=5)
    state[3] = np.random.uniform(low=0, high=5)

    Ts = 1
    cov = classic_models.WhiteNoiseAcceleration.Q(Ts, data_generation_params.ar_generation_params.innovation_variance)
    A = classic_models.WhiteNoiseAcceleration.A(Ts)

    ground_truth_position = np.zeros((data_generation_params.simulation_time, data_generation_params.observation_dimensions))
    for ts in range(data_generation_params.simulation_time):
        state = A @ state + np.random.multivariate_normal(mean=np.zeros(state.size), cov=cov)
        ground_truth_position[ts] = state[[0,2]]

    uncluttered_data = add_observation_noise(ground_truth_position, data_generation_params)

    min_boundary, max_boundary = foo_utils.get_bounding_box(uncluttered_data, 0.1)
    clutter_history = [generate_poisson_samples(data_generation_params.measurement_params.clutter_rate, [min_boundary[0], max_boundary[0]], [min_boundary[1], max_boundary[1]]) for _ in range(data_generation_params.simulation_time)]

    true_measurement_observed = np.random.uniform(size=data_generation_params.simulation_time) < data_generation_params.measurement_params.observation_probability

    data = MeasurementSet()
    for ts in range(data_generation_params.simulation_time):
        if true_measurement_observed[ts]:
            data.include_measurement(Measurement(ts, list(uncluttered_data[ts]), 1))

    for ts, clutter in enumerate(clutter_history):
        for m in list(clutter):
            data.include_measurement(Measurement(ts, list(m), 0))

    return data, ground_truth_position, None 


def generate_poisson_samples(base_rate, x_range : list[int], y_range : list[int]):
    if len(x_range) != 2 or x_range[0] >= x_range[1]:
        raise Exception("The x range must contain exactly two values [x_min, x_max] with x_min < x_max")
    if len(y_range) != 2 or y_range[0] >= y_range[1]: 
        raise Exception("The y range must contain exactly two values [y_min, y_max] with y_min < y_max")
    
    x_measure = x_range[1] - x_range[0]
    y_measure = y_range[1] - y_range[0]
    total_measure = x_measure * y_measure 

    total_num_samples = np.random.poisson(total_measure * base_rate)
    samples = np.random.uniform(low=[x_range[0], y_range[0]], high=[x_range[1], y_range[1]], size=(total_num_samples, 2))

    return samples