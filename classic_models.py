import numpy as np
import colorama
from tqdm import tqdm
import scipy.stats

import foo_utils
from pf_utils import get_mean_log_evidence
import settings

colorama.init(autoreset=True)

class KalmanFilter:

    def __init__(self, A : np.ndarray, Q : np.ndarray, H : np.ndarray, R : np.ndarray, m : np.ndarray, P : np.ndarray):
        self.A = A 
        self.Q = Q 
        self.H = H 
        self.R = R 
        self.m = m 
        self.P = P 

        self.order = None
    
    def predict_state(self):
        return self.A @ self.m, self.A @ self.P @ self.A.T + self.Q 

    def update(self, measurement : np.ndarray):
        m_pred, P_pred = self.predict_state()

        innovation = measurement - self.H @ m_pred
        S = self.H @ P_pred @ self.H.T + self.R 
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.m =  m_pred + K @ innovation 
        self.P =  P_pred - K @ S @ K.T
    
    def predict_measurement(self):
        m_pred, P_pred = self.predict_state()
        return self.H @ m_pred, self.H @ P_pred @ self.H.T + self.R 
    
    def translate_state_to_obs(self):
        return self.H @ self.m, self.H @ self.P @ self.H.T

    def copy(self):
        return KalmanFilter(self.A.copy(), self.Q.copy(), self.H.copy(), self.R.copy(), self.m.copy(), self.P.copy())

    def __str__(self):
        return f"""A: {self.A}
Q: {self.Q}
H: {self.H}
R: {self.R}

m: {self.m}
P: {self.P}"""
    

class InteractingMultipleModel:

    def __init__(self, models : list[KalmanFilter]):
        self.models = [m.copy() for m in models]
        self.num_models = len(self.models)

        self.order = 3 # True for the double Singer IMM

        self.mu_terms = np.ones((self.num_models)) / self.num_models
        self.model_change_probability = np.ones((self.num_models, self.num_models)) / self.num_models


    def update(self, measurement : np.ndarray):
        mu_mat = np.zeros((self.num_models, self.num_models))

        for i in range(self.num_models):
            for j in range(self.num_models):
                mu_mat[j,i] = self.model_change_probability[j,i] * self.mu_terms[i]

        c_vec = np.sum(mu_mat, axis=1)
        mu_mat /= c_vec 

        means = np.zeros((self.num_models, self.models[0].m.shape[0], 1))

        for j in range(self.num_models):
            for i in range(self.num_models):
                means[j] += mu_mat[j,i] * self.models[i].m 
        
        covs = np.zeros((self.num_models, self.models[0].m.shape[0], self.models[0].m.shape[0]))
        for j in range(self.num_models):
            for i in range(self.num_models):
                dev = self.models[i].m - means[i]
                covs[j,:,:] += mu_mat[j,i] * (self.models[i].P + dev @ dev.T)

        log_lambda_term = np.zeros((self.num_models))
        for j in range(self.num_models):
            max_cov = np.max(covs[j])
            rounded_covs = np.round(covs[j] / max_cov, 3) * max_cov #* np.round(max_cov, 3)

            self.models[j].m = means[j]
            self.models[j].P = rounded_covs

            pred_meas_mean, pred_meas_cov = self.models[i].predict_measurement()
            log_lambda_term[j] = scipy.stats.multivariate_normal.logpdf(measurement.flatten(), mean=pred_meas_mean.flatten(), cov=pred_meas_cov)

            self.models[j].update(measurement)

        log_new_mu_terms = np.zeros((self.num_models))
        for j in range(self.num_models):
            log_new_mu_terms[j] = log_lambda_term[j] + np.log(c_vec[j])

        self.mu_terms = np.exp(log_new_mu_terms - np.logaddexp.reduce(log_new_mu_terms))
        self.mu_terms = np.round(self.mu_terms, 3)
        self.mu_terms = self.mu_terms / np.sum(self.mu_terms)


    def get_position_estimate(self):
        mean = np.zeros((self.models[0].H.shape[0], 1))
        cov = np.zeros((mean.shape[0], mean.shape[0]))
        for j in range(self.num_models): 
            out_m = self.models[j].H @ self.models[j].m
            mean += self.mu_terms[j] * out_m 
            
        for j in range(self.num_models): 
            out_m = self.models[j].H @ self.models[j].m
            out_P = self.models[j].H @ self.models[j].P @ self.models[j].H.T + self.models[j].R 
            
            dev = out_m - mean 
            cov += self.mu_terms[j] * (out_P + dev @ dev.T)
        
        return mean, cov


    def get_position_prediction(self):
        mean = np.zeros((self.models[0].H.shape[0], 1))
        cov = np.zeros((mean.shape[0], mean.shape[0]))

        for j in range(self.num_models): 
            out_m = self.models[j].H @ self.models[j].A @ self.models[j].m
            mean += self.mu_terms[j] * out_m 
            
        for j in range(self.num_models): 
            A = self.models[j].A 
            H = self.models[j].H 
            
            out_m = H @ A @ self.models[j].m
            out_P = H @ A @ self.models[j].P @ A.T @ H.T + self.models[j].R 

            dev = out_m - mean 
            cov += self.mu_terms[j] * (out_P + dev @ dev.T)
        
        return mean, cov


class ConstantVelocity(KalmanFilter):
    
    @staticmethod
    def A(T):
        return np.array([[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]])

    @staticmethod
    def Q(T, process_variance):
        q1 = (T ** 3) / 3
        q2 = (T ** 2) / 2
        q3 = T 
        return np.array([[q1, q2, 0, 0], [q2, q3, 0, 0], [0, 0, q1, q2], [0, 0, q2, q3]]) * process_variance
    
    @staticmethod
    def H():
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    
    @staticmethod
    def R(noise_variance):
        return np.identity(2) * noise_variance 

    def __init__(self, m : np.ndarray, P : np.ndarray, T : float, process_variance : float, noise_variance : float):
        super().__init__(ConstantVelocity.A(T), 
                         ConstantVelocity.Q(T, process_variance), 
                         ConstantVelocity.H(), 
                         ConstantVelocity.R(noise_variance), 
                         m, P)
        
        self.order = 2

        if process_variance > 1E-2:
            print(f"{colorama.Fore.LIGHTYELLOW_EX}Note: the constant velocity model requires the innovation variance to be small, currently {process_variance}")
    

class WhiteNoiseAcceleration(KalmanFilter):
    
    @staticmethod
    def A(T):
        return np.array([[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]])

    @staticmethod
    def Q(T, process_variance):
        q1 = (T ** 3) / 3
        q2 = (T ** 2) / 2
        q3 = T 
        return np.array([[q1, q2, 0, 0], [q2, q3, 0, 0], [0, 0, q1, q2], [0, 0, q2, q3]]) * process_variance
    
    @staticmethod
    def H():
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    
    @staticmethod
    def R(noise_variance):
        return np.identity(2) * noise_variance 

    def __init__(self, m : np.ndarray, P : np.ndarray, T : float, process_variance : float, noise_variance : float):
        super().__init__(WhiteNoiseAcceleration.A(T), 
                         WhiteNoiseAcceleration.Q(T, process_variance), 
                         WhiteNoiseAcceleration.H(), 
                         WhiteNoiseAcceleration.R(noise_variance), 
                         m, P)
        
        self.order = 2

        if process_variance < 1E-2:
            print(f"{colorama.Fore.LIGHTYELLOW_EX}Note: the white noise acceleration model requires the innovation variance to be larger")
    


class SingerAcceleration(KalmanFilter):
    
    @staticmethod
    def A(T, alpha):
        exp_term = np.exp(-alpha * T)
        coord_terms = np.array([
            [1, T, (alpha * T - 1 + exp_term) / (alpha ** 2)], 
            [0, 1, (1 - exp_term) / alpha], 
            [0, 0, exp_term]
        ])

        A_mat = np.zeros((6, 6))
        A_mat[0:3,0:3] = coord_terms 
        A_mat[3:6,3:6] = coord_terms
        return A_mat 

    @staticmethod
    def Q(T, alpha, process_variance):
        # See Singer paper, section 4: https://ieeexplore.ieee.org/document/4103555
        exp_term = np.exp(-alpha * T)
        q11 = (1 - exp_term ** 2 + 2 * alpha * T + (2 / 3) * (alpha * T) ** 3 - 2 * (alpha * T) ** 2 - 4 * alpha * T * exp_term) / (2 * alpha ** 5)
        q12 = (exp_term ** 2 + 1 - 2 * exp_term + 2 * alpha * T * exp_term - 2 * alpha * T + (alpha * T) ** 2) / (2 * alpha ** 4)
        q13 = (1 - exp_term ** 2 - 2 * alpha * T * exp_term) / (2 * alpha ** 3)
        q22 = (4 * exp_term - 3 - exp_term ** 2 + 2 * alpha * T) / (2 * alpha ** 3)
        q23 = (exp_term ** 2 + 1 - 2 * exp_term) / (2 * alpha ** 2)
        q33 = (1 - exp_term ** 2) / (2 * alpha)

        Q_arr = np.zeros((6, 6))
        Q_arr[0:3,0:3] = np.array([[q11, q12, q13], [q12, q22, q23], [q13, q23, q33]])
        Q_arr[3:6,3:6] = np.array([[q11, q12, q13], [q12, q22, q23], [q13, q23, q33]])

        return Q_arr * process_variance * 2 * alpha 
    
    @staticmethod
    def H():
        return np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    
    @staticmethod
    def R(noise_variance):
        return np.identity(2) * noise_variance 

    def __init__(self, m : np.ndarray, P : np.ndarray, T : float, alpha : float, process_variance : float, noise_variance : float):
        super().__init__(SingerAcceleration.A(T, alpha), 
                         SingerAcceleration.Q(T, alpha, process_variance), 
                         SingerAcceleration.H(), 
                         SingerAcceleration.R(noise_variance), 
                         m, P)
        
        self.order = 3



class CovarianceMethodARTracker(KalmanFilter):
    
    @staticmethod
    def A(dim, ar_order):
        mat = np.zeros((dim * ar_order, dim * ar_order))

        for d in range(dim):
            for j in range(1, ar_order):
                mat[d * ar_order + j, d * ar_order + j - 1] = 1
        
        return mat 

    @staticmethod
    def Q(dim, ar_order, process_variance):
        mat = np.zeros((dim * ar_order, dim * ar_order))

        for d in range(dim):
            mat[d * ar_order, d * ar_order] = process_variance

        return mat 
    
    @staticmethod
    def H(dim, ar_order):
        mat = np.zeros((dim, ar_order * dim))
        for d in range(dim):
            mat[d, d * ar_order] = 1
        return mat 
    
    @staticmethod
    def R(dim, noise_variance):
        mat = np.zeros((dim, dim))
        for d in range(dim):
            mat[d, d] = noise_variance
        return mat 
    
    @staticmethod
    def get_sample_covariance(samples):
        N = samples.shape[0]
        res = np.zeros((N, N))

        return res 

    def __init__(self, m : np.ndarray, P : np.ndarray, dim : int, ar_order : int, process_variance : float, noise_variance : float):
        super().__init__(CovarianceMethodARTracker.A(dim, ar_order), 
                         CovarianceMethodARTracker.Q(dim, ar_order, process_variance), 
                         CovarianceMethodARTracker.H(dim, ar_order), 
                         CovarianceMethodARTracker.R(dim, noise_variance), 
                         m, P)
        
        self.ar_order = ar_order 
        self.dim = dim 
        self.order = ar_order 

        self.cov_mat = np.zeros((dim, ar_order, ar_order))
        self.normal_eq_target = np.zeros((dim, ar_order, 1))
        self.measurement_memory = np.zeros((dim, ar_order + 1))

        self.max_cov_mat_terms = np.ones(dim)

        for i in range(ar_order):
            self.shift_in_measurement(m[i::ar_order])
    

    def cov_term_update(self, this_dim : int, k : int, l : int):
        return self.measurement_memory[this_dim, -k - 1] * self.measurement_memory[this_dim, -l - 1]


    def update_cov_and_target(self):
        for this_dim in range(self.dim):
            for k in range(1, self.ar_order + 1):
                for l in range(1, self.ar_order + 1):
                    self.cov_mat[this_dim, k - 1, l - 1] += self.cov_term_update(this_dim, k, l) / self.max_cov_mat_terms[this_dim]
                self.normal_eq_target[this_dim, k - 1] += self.cov_term_update(this_dim, k, 0) / self.max_cov_mat_terms[this_dim]
            max_term = np.max(self.cov_mat[this_dim,:,:])
            self.cov_mat[this_dim,:,:] /= max_term
            self.max_cov_mat_terms[this_dim] *= max_term

    def shift_in_measurement(self, measurement : np.ndarray):
        self.measurement_memory[:, :self.ar_order] = self.measurement_memory[:, 1:]
        self.measurement_memory[:, -1] = measurement.flatten()


    def update_A_mat(self, new_ar : np.ndarray):
        for d in range(self.dim):
            self.A[d * self.ar_order, d * self.ar_order:(d + 1) * self.ar_order] = new_ar[d,:].flatten()


    def update(self, measurement : np.ndarray):
        self.shift_in_measurement(measurement)
        self.update_cov_and_target() 
        eps = 1E-6 # Make sure the matrix is invertible
        new_ar = foo_utils.inv(self.cov_mat + eps * np.identity(self.ar_order)[None,:,:]) @ self.normal_eq_target

        print(new_ar)
        
        self.update_A_mat(new_ar)

        super(CovarianceMethodARTracker, self).update(measurement)


class WindowedCovarianceMethodARTracker(KalmanFilter):
    
    @staticmethod
    def A(dim, ar_order):
        mat = np.zeros((dim * ar_order, dim * ar_order))

        for d in range(dim):
            for j in range(1, ar_order):
                mat[d * ar_order + j, d * ar_order + j - 1] = 1
        
        return mat 

    @staticmethod
    def Q(dim, ar_order, process_variance):
        mat = np.zeros((dim * ar_order, dim * ar_order))

        for d in range(dim):
            mat[d * ar_order, d * ar_order] = process_variance

        return mat 
    
    @staticmethod
    def H(dim, ar_order):
        mat = np.zeros((dim, ar_order * dim))
        for d in range(dim):
            mat[d, d * ar_order] = 1
        return mat 
    
    @staticmethod
    def R(dim, noise_variance):
        mat = np.zeros((dim, dim))
        for d in range(dim):
            mat[d, d] = noise_variance
        return mat 
    
    @staticmethod
    def get_sample_covariance(samples):
        N = samples.shape[0]
        res = np.zeros((N, N))

        return res 


    def __init__(self, m : np.ndarray, P : np.ndarray, dim : int, ar_order : int, process_variance : float, noise_variance : float, window_size : int):
        super().__init__(CovarianceMethodARTracker.A(dim, ar_order), 
                         CovarianceMethodARTracker.Q(dim, ar_order, process_variance), 
                         CovarianceMethodARTracker.H(dim, ar_order), 
                         CovarianceMethodARTracker.R(dim, noise_variance), 
                         m, P)
        
        self.ar_order = ar_order 
        self.dim = dim 
        self.order = ar_order 

        self.window_size = window_size

        self.cov_mat = np.zeros((dim, ar_order, ar_order))
        self.cov_mat[:,0,:] = 1
        self.normal_eq_target = np.zeros((dim, ar_order, 1))
        self.normal_eq_target[:,0] = 1
        self.measurement_memory = np.zeros((dim, window_size))
        self.memory_filled = 0

        self.max_cov_mat_terms = np.ones(dim)


        for i in range(ar_order):
            self.shift_in_measurement(m[i::ar_order])


    def update_cov_and_target(self):
        for this_dim in range(self.dim):
            dim_meas = self.measurement_memory[this_dim,:][:,None]
            #print("SHAPE", dim_meas.shape)
            mean = np.mean(dim_meas)
            adj_meas = dim_meas - mean
            cov = np.zeros((self.window_size, self.window_size))

            for k in range(self.window_size):
                for l in range(self.window_size):
                    for n in range(max(k, l), self.window_size):
                        cov[k,l] += adj_meas[n - l] * adj_meas[n - k]

            self.cov_mat[this_dim,1:,:] = cov[1:self.cov_mat.shape[1], 1:self.cov_mat.shape[2] + 1]
            self.normal_eq_target[this_dim,1:] = cov[1:self.normal_eq_target.shape[1],0:1]


    def shift_in_measurement(self, measurement : np.ndarray):
        self.measurement_memory[:, :self.window_size - 1] = self.measurement_memory[:, 1:]
        self.measurement_memory[:, -1] = measurement.flatten()

        self.memory_filled = min(self.memory_filled + 1, self.window_size)


    def update_A_mat(self, new_ar : np.ndarray):
        for d in range(self.dim):
            self.A[d * self.ar_order, d * self.ar_order:(d + 1) * self.ar_order] = new_ar[d,:].flatten()


    def update(self, measurement : np.ndarray):
        self.shift_in_measurement(measurement)
        self.update_cov_and_target() 
        eps = 1E-6 # Make sure the matrix is invertible
        new_ar = foo_utils.inv(self.cov_mat + eps * np.identity(self.ar_order)[None,:,:]) @ self.normal_eq_target
        
        self.update_A_mat(new_ar)

        super(WindowedCovarianceMethodARTracker, self).update(measurement)




def kf_with_nn_association(model : KalmanFilter, cluttered_data : np.ndarray, params : settings.ClassicalStructuralTrackerParameters, verbose=True):
    model_history = [None for _ in range(params.simulation_time)]
    log_observation_predictions = np.zeros(params.simulation_time)

    for ts in tqdm(range(model.order, params.simulation_time), disable=not verbose):
        measurements = cluttered_data[ts]

        meas_pred_mean, meas_pred_cov = model.predict_measurement()
        meas_ind = np.argmin(mahalanobis(measurements[:,:,None], meas_pred_mean, meas_pred_cov[None,:,:]))

        meas = measurements[meas_ind, :, None]
        obs_pred_mean, obs_pred_cov = model.predict_measurement()

        det = np.linalg.det(obs_pred_cov)
        assert not np.isnan(det), "Covariance determinant cannot be NaN"
        assert np.allclose(obs_pred_cov, obs_pred_cov.T), f"Covariance must be symmetric. Provided: {obs_pred_cov}"

        if det > 1E-3:
            log_observation_predictions[ts] = scipy.stats.multivariate_normal.logpdf(meas.flatten(), obs_pred_mean.flatten(), obs_pred_cov, allow_singular=True)
        else:
            log_observation_predictions[ts] = -np.inf 

        model.update(meas)

        model_history[ts] = model.copy()
    
    return model_history, log_observation_predictions 


def imm_with_nn_association(model : InteractingMultipleModel, cluttered_data : np.ndarray, params : settings.ClassicalStructuralTrackerParameters, verbose=True):
    mean_history = np.zeros((params.simulation_time, params.observation_dimensions))
    cov_history = np.zeros((params.simulation_time, params.observation_dimensions, params.observation_dimensions))
    
    pred_mean_history = np.zeros((params.simulation_time, params.observation_dimensions))
    pred_cov_history = np.zeros((params.simulation_time, params.observation_dimensions, params.observation_dimensions))
    
    log_observation_predictions = np.zeros(params.simulation_time)
    for ts in tqdm(range(model.order, params.simulation_time), disable=not verbose):
        measurements = cluttered_data[ts]

        meas_pred_mean, meas_pred_cov = model.get_position_prediction()
        meas_ind = np.argmin(mahalanobis(measurements[:,:,None], meas_pred_mean, meas_pred_cov[None,:,:]))

        meas = measurements[meas_ind, :, None]
        obs_pred_mean, obs_pred_cov = model.get_position_prediction()
        pred_mean_history[ts] = obs_pred_mean.flatten()
        pred_cov_history[ts] = obs_pred_cov 

        det = np.linalg.det(obs_pred_cov)
        assert not np.isnan(det), "Covariance determinant cannot be NaN"
        assert np.allclose(obs_pred_cov, obs_pred_cov.T, atol=1E-3), f"Covariance must be symmetric. Provided:\n{obs_pred_cov}"

        if det > 1E-3:
            log_observation_predictions[ts] = scipy.stats.multivariate_normal.logpdf(meas.flatten(), obs_pred_mean.flatten(), obs_pred_cov, allow_singular=True)
        else:
            log_observation_predictions[ts] = -np.inf 

        model.update(meas)
        m, P = model.get_position_estimate()
        mean_history[ts], cov_history[ts] = m.flatten(), P

    
    return mean_history, cov_history, pred_mean_history, pred_cov_history, log_observation_predictions 




def setup_cv(ground_truth : np.ndarray, params : settings.ClassicalStructuralTrackerParameters):
    T = 1

    if params.z_oracle_prior:
        initial_state = np.zeros((4, 1))
        initial_state[0] = ground_truth[1, 0]
        initial_state[1] = ground_truth[1, 0] - ground_truth[0, 0]
        initial_state[2] = ground_truth[1, 1]
        initial_state[3] = ground_truth[1, 1] - ground_truth[0, 1]

        initial_cov = np.identity(4)
        initial_cov[0,0] = 1E-2
        initial_cov[2,2] = 1E-2
        initial_cov[1,1] = 1
        initial_cov[3,3] = 1
    else:
        raise NotImplementedError

    return ConstantVelocity(initial_state, initial_cov, T, params.state_dynamic_variance, params.measurement_params.noise_variance)


def setup_wna(ground_truth : np.ndarray, params : settings.ClassicalStructuralTrackerParameters):
    T = 1

    if params.z_oracle_prior:
        initial_state = np.zeros((4, 1))
        initial_state[0] = ground_truth[1, 0]
        initial_state[1] = ground_truth[1, 0] - ground_truth[0, 0]
        initial_state[2] = ground_truth[1, 1]
        initial_state[3] = ground_truth[1, 1] - ground_truth[0, 1]

        initial_cov = np.identity(4)
        initial_cov[0,0] = 1E-2
        initial_cov[2,2] = 1E-2
        initial_cov[1,1] = 1
        initial_cov[3,3] = 1
    else:
        raise NotImplementedError

    return WhiteNoiseAcceleration(initial_state, initial_cov, T, params.state_dynamic_variance, params.measurement_params.noise_variance)


def setup_singer(ground_truth : np.ndarray, params : settings.ClassicalStructuralTrackerParameters):
    T = 1

    if params.z_oracle_prior:
        initial_state = np.zeros((6, 1))
        initial_state[0] = ground_truth[2, 0]
        initial_state[1] = ground_truth[2, 0] - ground_truth[1, 0]
        # TODO: Double check this initialisation
        initial_state[2] = 0.5 * ( (ground_truth[2, 0] - ground_truth[1, 0]) - (ground_truth[1, 0] - ground_truth[0, 0]))
        initial_state[3] = ground_truth[2, 1]
        initial_state[4] = ground_truth[2, 1] - ground_truth[1, 1]
        initial_state[5] = 0.5 * ( (ground_truth[2, 1] - ground_truth[1, 1]) - (ground_truth[1, 1] - ground_truth[0, 1]))


        initial_cov = np.identity(6)
        initial_cov[0,0] = 1E-2
        initial_cov[3,3] = 1E-2

        initial_cov[1,1] = 1E-1
        initial_cov[4,4] = 1E-1
        
        initial_cov[2,2] = 1
        initial_cov[5,5] = 1
    else:
        raise NotImplementedError

    return SingerAcceleration(initial_state, initial_cov, T, params.singer_rate, params.state_dynamic_variance, params.measurement_params.noise_variance)


def setup_windowed_covariance_method_ar(ground_truth : np.ndarray, params : settings.CovarianceMethodARParameters):
    if params.z_oracle_prior:
        initial_state = np.zeros((params.model_order * params.observation_dimensions, 1))
        initial_cov = (1E-3) * np.identity(params.model_order * params.observation_dimensions)
        for i in range(params.model_order):
            initial_state[i::params.model_order,0:1] = ground_truth[i,:,None]
    else:
        raise NotImplementedError
    
    return WindowedCovarianceMethodARTracker(initial_state, initial_cov, params.observation_dimensions, params.model_order, params.state_dynamic_variance, params.measurement_params.noise_variance, 10)


def setup_double_singer_imm(ground_truth : np.ndarray, params : settings.IMMTrackerParameters):
    models = []
    for m_par in params.model_params:
        models.append(setup_singer(ground_truth, m_par))
    
    return InteractingMultipleModel(models)


def unpack_model_history(model_history : list[KalmanFilter], params : settings.TrackerParameters):
    state_mean_history = np.zeros((params.simulation_time, model_history[-1].A.shape[0]))
    state_cov_history = np.zeros((params.simulation_time, model_history[-1].Q.shape[0], model_history[-1].Q.shape[0]))

    state_to_obs_mean_history = np.zeros((params.simulation_time, model_history[-1].H.shape[0]))
    state_to_obs_cov_history = np.zeros((params.simulation_time, model_history[-1].R.shape[0], model_history[-1].R.shape[0]))

    pred_mean_history = np.zeros((params.simulation_time, 2))
    pred_cov_history = np.zeros((params.simulation_time, 2, 2))

    for t in range(params.simulation_time):
        if model_history[t] is None:
            continue
        state_mean_history[t,:] = model_history[t].m.flatten()
        state_cov_history[t,:,:] = model_history[t].P

        obs_m, obs_P = model_history[t].translate_state_to_obs()
        state_to_obs_mean_history[t,:] = obs_m.flatten()
        state_to_obs_cov_history[t,:,:] = obs_P

        pred_m, pred_P = model_history[t].predict_measurement()
        pred_mean_history[t] = pred_m.flatten()
        pred_cov_history[t] = pred_P 
    
    return state_mean_history, state_cov_history, state_to_obs_mean_history, state_to_obs_cov_history, pred_mean_history, pred_cov_history



def get_mse_gauss(ground_truth : np.ndarray, est_m : np.ndarray, est_P : np.ndarray):
    assert ground_truth.shape == est_m.shape, f"Shapes of ground truth and estimated mean do not match: {ground_truth.shape} and {est_m.shape}"
    assert est_m.shape[0] == est_P.shape[0], f"Shapes of estimated mean and covariance are not compatible: {est_m.shape} and {est_P.shape}"
    assert est_P.shape[0] == est_P.shape[1], f"Provided covariance matrix is not square: {est_P.shape}"
    
    diff = ground_truth - est_m 
    return np.sum(diff ** 2) + np.trace(est_P)


def get_rmse_gauss(ground_truth : np.ndarray, est_m : np.ndarray, est_P : np.ndarray):
    return np.sqrt(get_mse_gauss(ground_truth, est_m, est_P))


def get_mean_cumulative_rmse_gauss(ground_truth_history : np.ndarray, est_m_history : np.ndarray, est_P_history : np.ndarray, burn_in : int):
    sim_time = ground_truth_history.shape[0]
    rmse_history = np.zeros(sim_time - burn_in) - np.inf
    for t in range(burn_in, sim_time):
        rmse_history[t - burn_in] = get_rmse_gauss(ground_truth_history[t], est_m_history[t], est_P_history[t])
    return np.mean(rmse_history) 


def get_ground_truth_predictive_likelihood_gauss(ground_truth_history : np.ndarray, pred_mean_history : np.ndarray, pred_cov_history : np.ndarray, burn_in : int):
    res = 0
    sim_time = ground_truth_history.shape[0]

    for t in range(burn_in, sim_time - 1):
        res += scipy.stats.multivariate_normal.logpdf(ground_truth_history[t + 1], pred_mean_history[t], pred_cov_history[t])
    
    return res / (sim_time - 1 - burn_in)


def get_model_performance(model : KalmanFilter, cluttered_data : np.ndarray, ground_truth : np.ndarray, burn_in : int, params : settings.ClassicalStructuralTrackerParameters):
    _, log_likelihood_history = kf_with_nn_association(model, cluttered_data, params, False)
    return get_mean_log_evidence(log_likelihood_history, burn_in)


def get_imm_model_performance(model : InteractingMultipleModel, cluttered_data : np.ndarray, ground_truth : np.ndarray, burn_in : int, params : settings.IMMTrackerParameters):
    _, _, _, _, log_likelihood_history = imm_with_nn_association(model, cluttered_data, params, False)
    return get_mean_log_evidence(log_likelihood_history, burn_in)


def mahalanobis(data, mean, cov):
    diff = data - mean 
    return (np.sqrt(foo_utils.transpose_last2(diff) @ foo_utils.inv(cov) @ diff)).flatten()



class Jin2017Tracker(KalmanFilter):
    """
    Implementation of the tracker proposed in "Jin, B., Guo, J., He, D. et al. Adaptive Kalman filtering based on optimal autoregressive predictive model. GPS Solut 21, 307–317 (2017). https://doi.org/10.1007/s10291-016-0561-x".

    This implementation assumes the coordinates of the position (usually x and y) are loosely coupled so the tracker handles these independently.

    The notation used here is intentionally as close as possible to that used in the algorithm described by Jin.
    """

    @staticmethod 
    def base_A(M : int):
        mat = np.zeros((M, M))
        for i in range(1, M):
            mat[i, i - 1] = 1
        
        return mat 
    

    @staticmethod
    def base_Q(M : int):
        mat = np.zeros((M, M))
        mat[0,0] = 1
        return mat 
    

    @staticmethod 
    def base_H(M : int):
        mat = np.zeros((1, M))
        mat[0,0] = 1

        return mat 
    

    @staticmethod
    def base_R():
        mat = np.array([[1]])
        return mat 


    def __init__(self, dim : int, M : int, N : int, W : int, innovation_variance : float, measurement_noise_variance : float, initial_mean : np.ndarray):
        """
        Initialise the tracker proposed by (Jin, 2017).

        Args:
            dim: the number of observation dimensions.
            M: the process order of the AR models estimated by the tracker.
            N: the degree of the polynomials this tracker approximates the trajectory as.
            W: the window size matching the number of deviation terms used to calculated the S matrix.
            innovation_variance: the initial estimate of the driving process variance.
            measurement_noise_variance: the a priori known measurement noise variance.
            initial_mean: the M x 1 prior on the trajectory history.
        """
        if not (M >= N + 2):
            # (Jin, 2017) specifies M >= N + 2 in their algorithm, but they violate this constraint themselves in some comparisons which could also be
            # useful for this paper.
            print(f"{colorama.Fore.RED}The algorithm described by (Jin, 2017) requires M >= N + 2. Given: (M, N) = ({M}, {N}).")

        self.dim_trackers = [KalmanFilter(
            Jin2017Tracker.base_A(M), 
            Jin2017Tracker.base_Q(M) * innovation_variance, 
            Jin2017Tracker.base_H(M),
            Jin2017Tracker.base_R() * measurement_noise_variance,
            initial_mean[:,dim_ind], 
            np.identity(M) * measurement_noise_variance
        ) for dim_ind in range(dim)]

        self.W = W # Window size
        self.squared_deviation_history = np.zeros((dim, self.W))
        self.k = 0

        self.dim = dim
        self.M = M # Model order
        self.N = N # Polynomial order

        self.S_ests = np.zeros((dim, 1, 1))


    def jin_ar_est(self, dim_ind : int):
        P = self.dim_trackers[dim_ind].P
        # The Vandermonde matrix defined in (Jin, 2017)
        A = np.vander(np.arange(1, self.M + 1), N=self.N + 1, increasing=True).T 

        b = np.zeros((self.N + 1, 1))
        b[0] = 1

        # Calculate the optimal AR estimates
        inv_P = foo_utils.inv(P)
        u_star = inv_P @ A.T @ foo_utils.inv(A @ inv_P @ A.T) @ b # The AR coefficients

        return u_star
    
    
    def update(self, measurement : np.ndarray):
        """
        Update the tracker position estimate (mean and covariance) given the provided measurement.

        Args:
            measurement: the dim x 1 observation of the target.
        """
        self.k += 1

        for dim_ind in range(self.dim):
            # (Jin, 2017): Step 1
            # Get the optimal ar coefficient estimates as an Mx1 vector
            u_star = self.jin_ar_est(dim_ind)

            # Update the transition matrix. In the KF implementation here, the state transition matrix is A, in the place of F in (Jin, 2017).
            self.dim_trackers[dim_ind].A[0,:] = u_star[:,0]


            # (Jin, 2017): Step 2
            # Get the predicted state and covariance
            x_pred, P_pred = self.dim_trackers[dim_ind].predict_state()

            # Defined here for convenience
            this_H = self.dim_trackers[dim_ind].H

            # Keep track of the deviation of the measurement from the predicted position
            d = measurement[dim_ind] - this_H @ x_pred

            history_filled = min(self.W, self.k)
            # Shift memory of squared deviation history
            self.squared_deviation_history[dim_ind, 1:history_filled] = self.squared_deviation_history[dim_ind, 0:history_filled - 1]
            # Insert the new squared deviation
            self.squared_deviation_history[dim_ind, 0] = d ** 2

            
            # (Jin, 2017): Step 3
            # R is not being updated in this algorithm, it is assumed known
            this_R = self.dim_trackers[dim_ind].R
            K = P_pred @ this_H.T @ foo_utils.inv(this_H @ P_pred @ this_H.T + this_R)

            self.dim_trackers[dim_ind].m = x_pred + K @ d
            self.dim_trackers[dim_ind].P = (np.identity(self.M) - K @ this_H) @ P_pred

            
            # (Jin, 2017): Step 4
            # Update the estimate of S
            if self.k >= self.W:
                self.S_ests[dim_ind,:,:] = np.mean(self.squared_deviation_history[dim_ind,:])[None,None]
            else:
                self.S_ests[dim_ind,:,:] = self.S_ests[dim_ind,:,:] * (history_filled - 1) / history_filled + np.mean(self.squared_deviation_history[dim_ind,:history_filled])[None,None]
            # Update the Q estimate
            self.dim_trackers[dim_ind].Q = K @ self.S_ests[dim_ind,:,:] @ K.T


    def get_position_state(self):
        m = np.zeros((self.dim, 1))
        P = np.identity(self.dim)

        for dim_ind in range(self.dim):
            m[dim_ind] = self.dim_trackers[dim_ind].m[0]
            P[dim_ind,dim_ind] = self.dim_trackers[dim_ind].P[0,0]
        
        return m, P 
    

    def get_predicted_position_state(self):
        m = np.zeros((self.dim, 1))
        P = np.zeros((self.dim, self.dim))

        for dim_ind in range(self.dim):
            x_pred, P_pred = self.dim_trackers[dim_ind].predict_state()
            m[dim_ind] = x_pred[0]
            P[dim_ind,dim_ind] = P_pred[0,0]
        
        return m, P 
    

def run_Jin2017_tracker_with_nn_association(cluttered_data : np.ndarray, ground_truth_position : np.ndarray, data_generation_params : settings.DataGenerationParameters, jin2017_params : settings.Jin2017TrackerParameters, verbose=True):

    if jin2017_params.z_oracle_prior:
        trajectory_prior = ground_truth_position[:jin2017_params.model_order,:][::-1]
    else:
        raise NotImplementedError

    jin_model = Jin2017Tracker(jin2017_params.observation_dimensions, jin2017_params.model_order, jin2017_params.polynomial_order, jin2017_params.window, jin2017_params.innovation_variance, 
                               data_generation_params.measurement_params.noise_variance, trajectory_prior)

    mean_history = np.zeros((data_generation_params.simulation_time, data_generation_params.observation_dimensions))
    cov_history = np.zeros((data_generation_params.simulation_time, data_generation_params.observation_dimensions, data_generation_params.observation_dimensions))

    pred_mean_history = np.zeros((data_generation_params.simulation_time, data_generation_params.observation_dimensions))
    pred_cov_history = np.zeros((data_generation_params.simulation_time, data_generation_params.observation_dimensions, data_generation_params.observation_dimensions))

    log_observation_predictions = np.zeros(data_generation_params.simulation_time)

    for ts in tqdm(range(trajectory_prior.shape[0], data_generation_params.simulation_time), disable=not verbose):
        measurements = cluttered_data[ts]

        meas_pred_mean, meas_pred_cov = jin_model.get_predicted_position_state()
        pred_mean_history[ts, :] = meas_pred_mean[:,0]
        pred_cov_history[ts, :, :] = meas_pred_cov
        meas_ind = np.argmin(mahalanobis(measurements[:,:,None], meas_pred_mean, meas_pred_cov[None,:,:]))

        meas = measurements[meas_ind, :, None]
        log_observation_predictions[ts] = scipy.stats.multivariate_normal.logpdf(meas.flatten(), meas_pred_mean.flatten(), meas_pred_cov)
        #print(log_observation_predictions[ts])

        jin_model.update(meas)

        m, P = jin_model.get_position_state()
        mean_history[ts, :] = m[:,0]
        cov_history[ts, :, :] = P

    
    return mean_history, cov_history, pred_mean_history, pred_cov_history, log_observation_predictions


def get_jin2017_model_performance(cluttered_data : list[np.ndarray], ground_truth_position : np.ndarray, data_generation_params : settings.DataGenerationParameters, jin2017_params : settings.Jin2017TrackerParameters, burn_in : int):
    _, _, _, _, log_likelihood_history = run_Jin2017_tracker_with_nn_association(cluttered_data, ground_truth_position, data_generation_params, jin2017_params, False)
    return get_mean_log_evidence(log_likelihood_history, burn_in)