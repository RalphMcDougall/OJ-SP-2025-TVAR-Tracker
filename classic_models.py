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
    
    def predict_state(self):
        return self.A @ self.m, self.A @ self.P @ self.A.T + self.Q 

    def update(self, measurement):
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



def kf_with_nn_association(model : KalmanFilter, cluttered_data : np.ndarray, params : settings.ClassicalStructuralTrackerParameters, verbose=True):
    model_history = [None for _ in range(params.simulation_time)]
    log_observation_predictions = np.zeros(params.simulation_time)

    for ts in tqdm(range(2, params.simulation_time), disable=not verbose):
        measurements = cluttered_data[ts]

        meas_pred_mean, meas_pred_cov = model.predict_measurement()
        meas_ind = np.argmin(mahalanobis(measurements[:,:,None], meas_pred_mean, meas_pred_cov[None,:,:]))

        meas = measurements[meas_ind, :, None]
        obs_pred_mean, obs_pred_cov = model.predict_measurement()
        log_observation_predictions[ts] = scipy.stats.multivariate_normal.logpdf(meas.flatten(), obs_pred_mean.flatten(), obs_pred_cov)

        model.update(meas)

        model_history[ts] = model.copy()
    
    return model_history, log_observation_predictions 



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


def get_model_performance(model : KalmanFilter, cluttered_data : np.ndarray, ground_truth : np.ndarray, burn_in, params : settings.ClassicalStructuralTrackerParameters):
    model_history, log_likelihood_history = kf_with_nn_association(model, cluttered_data, params, False)
    return get_mean_log_evidence(log_likelihood_history, burn_in)



def mahalanobis(data, mean, cov):
    diff = data - mean 
    return (np.sqrt(foo_utils.transpose_last2(diff) @ foo_utils.inv(cov) @ diff)).flatten()