import numpy as np
from tqdm import tqdm
import scipy.special

import settings 
import foo_utils



class ParticleCloud:
    def __init__(self, Z, ma, Pa, alpha, beta, assoc, log_weights):
        self.Z = Z
        self.ma = ma
        self.Pa = Pa 
        self.alpha = alpha 
        self.beta = beta  
        self.assoc = assoc 
        self.log_weights = log_weights

        self.num_particles = self._get_num_particles()
    

    def _get_num_particles(self):
        def get_new_cand(cand : int | None, field : np.ndarray | None):
            if field is not None:
                field_length = field.shape[0]
                assert cand is None or cand == field_length, f"Not all properties have the same number of particles : {cand} vs {field_length}"
                return field_length
            return cand

        num_particles_cand = None
        num_particles_cand = get_new_cand(num_particles_cand, self.Z)
        num_particles_cand = get_new_cand(num_particles_cand, self.ma)
        num_particles_cand = get_new_cand(num_particles_cand, self.Pa)
        num_particles_cand = get_new_cand(num_particles_cand, self.alpha)
        num_particles_cand = get_new_cand(num_particles_cand, self.beta)
        num_particles_cand = get_new_cand(num_particles_cand, self.assoc)
        num_particles_cand = get_new_cand(num_particles_cand, self.log_weights)
        return num_particles_cand 


    def get_choice(self, indices : np.ndarray):
        return ParticleCloud(self.Z[indices], self.ma[indices], self.Pa[indices], self.alpha[indices], self.beta[indices], self.assoc[indices], self.log_weights[indices])
    

    def tile(self, num_copies : int, copies_adjacent : bool):
        tiled_Z = foo_utils.tile_field(self.Z, num_copies, copies_adjacent)
        tiled_ma = foo_utils.tile_field(self.ma, num_copies, copies_adjacent)
        tiled_Pa = foo_utils.tile_field(self.Pa, num_copies, copies_adjacent)
        tiled_alpha = foo_utils.tile_field(self.alpha, num_copies, copies_adjacent)
        tiled_beta = foo_utils.tile_field(self.beta, num_copies, copies_adjacent)
        tiled_assoc = foo_utils.tile_field(self.assoc, num_copies, copies_adjacent)
        tiled_log_weights = foo_utils.tile_field(self.log_weights, num_copies, copies_adjacent)

        return ParticleCloud(tiled_Z, tiled_ma, tiled_Pa, tiled_alpha, tiled_beta, tiled_assoc, tiled_log_weights)
    

    def copy(self):
        copy_field = lambda field: None if field is None else field.copy()
        
        return ParticleCloud(copy_field(self.Z), copy_field(self.ma), copy_field(self.Pa), 
                                copy_field(self.alpha), copy_field(self.beta), copy_field(self.assoc), 
                                copy_field(self.log_weights))





def construct_parameter_prior_mean(model_order : int):
    params = np.zeros(model_order)
    
    # Let the prior be a polynomial with poles at the prior pole
    prior_pole = 0.9

    poly = np.poly([prior_pole for _ in range(model_order)])
    params[:] = -poly[1:]

    return params


def construct_initial_particles(params : settings.TVARTrackerParameters, ground_truth_data : np.ndarray, measurements : list[np.ndarray], ground_truth_ar : np.ndarray):
    if params.z_oracle_prior:
        Z_init = np.array([ground_truth_data[:params.model_order + 1:][::-1,:].T for _ in range(params.num_particles)])
    else:
        Z_init = np.zeros((params.num_particles, params.observation_dimensions, params.model_order + 1))
        for t in range(params.model_order + 1):
            meas_inds = np.random.randint(len(measurements[t]), size=(params.num_particles))
            Z_init[:,:,params.model_order - t] = measurements[t][meas_inds,:]
    
    if params.a_oracle_prior:
        ma_init = np.array([ground_truth_ar[params.model_order, :][:,None] for _ in range(params.num_particles)])
        Pa_init = np.array([np.identity(params.model_order) for _ in range(params.num_particles)]) * 1E-6
    else:
        ma_init = np.array([np.zeros((params.model_order, 1)) for _ in range(params.num_particles)])
        ma_init[:,:,:] = construct_parameter_prior_mean(params.model_order)[None,:,None]
        Pa_init = np.array([np.identity(params.model_order) for _ in range(params.num_particles)]) * np.max([params.state_innovation_to_process_innovation, 1E-2])
    if params.sigma_oracle_prior:
        alpha_init = np.ones((params.num_particles, 1)) * 1E12
        beta_init  = np.ones((params.num_particles, 1)) * 1E12 * params.process_instance_params.innovation_variance
    else:
        # A relatively uninformative scale prior
        alpha_init = np.ones((params.num_particles, 1)) * 6
        beta_init  = np.ones((params.num_particles, 1)) * 5
    assoc_init = np.zeros((params.num_particles, 1))

    log_weights_init = np.zeros(params.num_particles) - np.log(params.num_particles)

    return ParticleCloud(Z_init, ma_init, Pa_init, alpha_init, beta_init, assoc_init, log_weights_init)


def evaluate_log_predictive_likelihood(old_particles : ParticleCloud, observations, params : settings.TVARTrackerParameters):
    # NOTE: This is not taking into account the full observation set. Need to include the V^{-N + 1} factors as well...
    num_observations = observations.shape[0]
    if num_observations == 0:
        return np.log(params.measurement_params.observation_probability) if params.measurement_params.observation_probability > 0 else -np.inf

    all_possible_assoc = foo_utils.tile_field(np.arange(start=0, stop=num_observations + 1), params.num_particles, True)
    tiled_particles = old_particles.tile(num_observations + 1, False)
    tiled_alpha = tiled_particles.alpha 
    tiled_beta = tiled_particles.beta

    upd_mz, upd_Pz = get_zt_given_ztm1_gaussian_in_zt(tiled_particles, params)
    obs_mz, obs_Pz = get_yt_given_zt_gaussian_in_zt(all_possible_assoc, observations, params)
    prod_mz, prod_Pz, const_offset, const_Pz = foo_utils.gaussian_product(upd_mz, upd_Pz, obs_mz, obs_Pz)
    # Now, the z distribution gets integrated out, so only the offset term matters!

    ys = get_modified_yt(all_possible_assoc, observations, params)
    m_from_offset = ys - const_offset
    df, mu, shape = foo_utils.get_st_params(m_from_offset, const_Pz, tiled_alpha, tiled_beta)

    log_st_evaluation = foo_utils.MultivariateStudentT(mu, shape, df).log_pdf(ys).flatten()
    P_G = 1 # An approximation
    log_prob_arr = np.array([ np.log(1 - P_G * params.measurement_params.observation_probability) if P_G * params.measurement_params.observation_probability < 1 else -np.inf, np.log(params.measurement_params.observation_probability)])
    log_lambda_eval = log_prob_arr[np.where(all_possible_assoc > 0, 1, 0)]

    old_log_probs = log_st_evaluation + log_lambda_eval
    log_probs = old_log_probs.reshape((params.num_particles, num_observations + 1), order="C")
    assert np.allclose(old_log_probs[:num_observations + 1], log_probs[0,:num_observations + 1]), f"Log probabilities reshaping was not done correctly:\n OLD: {old_log_probs[:num_observations + 1]}\n\n NEW: {log_probs[0,:num_observations + 1]}"

    log_probs = np.logaddexp.reduce(log_probs, axis=1)

    assert log_probs.shape == old_particles.log_weights.shape, f"Log density does not have the same shape as log weights: {log_probs.shape} vs {old_particles.log_weights.shape}"

    return np.logaddexp.reduce(log_probs + old_particles.log_weights)


def get_modified_yt(assoc, observations, params : settings.TVARTrackerParameters):
    default_mz = np.zeros((1, params.observation_dimensions))
    modified_observations = np.concatenate([default_mz, observations], axis=0)
    return modified_observations[assoc.flatten().astype(int), :, None]


def get_yt_given_zt_gaussian_in_zt(new_assoc, observations, params : settings.TVARTrackerParameters):
    num_observations = observations.shape[0]

    mz = get_modified_yt(new_assoc, observations, params)

    default_Pz_tilde = np.identity(params.observation_dimensions)[None,:,:] * 1E12

    idents = np.array([np.identity(params.observation_dimensions) * params.process_instance_params.noise_to_innovation_ratio for _ in range(num_observations)])
    covariances = np.concatenate([default_Pz_tilde, idents], axis=0) if num_observations > 0 else default_Pz_tilde
    Pz_tilde = covariances[new_assoc.flatten().astype(int), :, :]

    return mz, Pz_tilde 


def get_zt_given_ztm1_gaussian_in_zt(old_particles : ParticleCloud, params : settings.TVARTrackerParameters):
    old_all_z = old_particles.Z 
    old_ma = old_particles.ma 
    old_Pa_tilde = old_particles.Pa 
    
    Z_mat = old_all_z[:, :, :params.model_order]
    
    mz = Z_mat @ old_ma
    Pz_tilde = np.identity(params.observation_dimensions)[None,:,:] + Z_mat @ old_Pa_tilde @ foo_utils.transpose_last2(Z_mat)
    return mz, Pz_tilde 


def ig_update(new_particles : ParticleCloud, old_particles : ParticleCloud, older_particles : ParticleCloud, observations, old_observations, params : settings.TVARTrackerParameters):
    old_alpha = old_particles.alpha 
    old_beta = old_particles.beta 

    if old_observations is None or older_particles is None:
        return old_alpha, old_beta

    all_z = old_particles.Z 
    old_assoc = old_particles.assoc 
    z_0 = all_z[:, :, 0:1]

    upd_mz, upd_Pz = get_zt_given_ztm1_gaussian_in_zt(older_particles, params)
    temp_alpha, temp_beta = foo_utils.inv_gamma_update(z_0 - upd_mz, upd_Pz, old_alpha, old_beta)

    obs_mz, obs_Pz = get_yt_given_zt_gaussian_in_zt(old_assoc, old_observations, params)
    new_alpha, new_beta = foo_utils.inv_gamma_update(z_0 - obs_mz, obs_Pz, temp_alpha, temp_beta)


    return new_alpha, new_beta


def get_zt_proposal_dist(old_particles : ParticleCloud, observations, new_assoc, params : settings.TVARTrackerParameters):
    old_alpha = old_particles.alpha 
    old_beta = old_particles.beta

    upd_mz, upd_Pz = get_zt_given_ztm1_gaussian_in_zt(old_particles, params)
    obs_mz, obs_Pz = get_yt_given_zt_gaussian_in_zt(new_assoc, observations, params)

    prod_mz, prod_Pz, const_offset, const_Pz = foo_utils.gaussian_product(upd_mz, upd_Pz, obs_mz, obs_Pz)

    d = upd_mz.shape[1]
    C_term = (1 / 2) * (foo_utils.transpose_last2(const_offset) @ foo_utils.inv(const_Pz) @ const_offset)[:,:,0] 

    mu = prod_mz 
    shape = ((old_beta + C_term) / (old_alpha + 0.5 * d))[:,:,None] * prod_Pz
    df = 2 * old_alpha + d 

    #df, mu, shape = foo_utils.get_st_params(prod_mz, prod_Pz, old_alpha, old_beta)

    return foo_utils.MultivariateStudentT(mu, shape, df)


def get_assoc_proposal_dist(old_particles : ParticleCloud, observations, params : settings.TVARTrackerParameters):
    num_observations = observations.shape[0]

    all_possible_assoc = np.tile(np.arange(start=0, stop=num_observations + 1), params.num_particles)
    tiled_particles = old_particles.tile(num_observations + 1, False)
    tiled_alpha = tiled_particles.alpha 
    tiled_beta = tiled_particles.beta 

    upd_mz, upd_Pz = get_zt_given_ztm1_gaussian_in_zt(tiled_particles, params)
    obs_mz, obs_Pz = get_yt_given_zt_gaussian_in_zt(all_possible_assoc, observations, params)

    prod_mz, prod_Pz, const_offset, const_Pz = foo_utils.gaussian_product(upd_mz, upd_Pz, obs_mz, obs_Pz)
    # Now, the z distribution gets integrated out, so only the offset term matters!

    ys = get_modified_yt(all_possible_assoc, observations, params)
    m_from_offset = ys - const_offset
    df, mu, shape = foo_utils.get_st_params(m_from_offset, const_Pz, tiled_alpha, tiled_beta)

    log_st_evaluation = foo_utils.MultivariateStudentT(mu, shape, df).log_pdf(ys).flatten()
    P_G = 1 # An approximation
    log_prob_arr = np.array([ np.log(1 - P_G * params.measurement_params.observation_probability) if P_G * params.measurement_params.observation_probability < 1 else -np.inf, np.log(params.measurement_params.observation_probability)])
    log_lambda_eval = log_prob_arr[np.where(all_possible_assoc > 0, 1, 0)]

    old_log_probs = log_st_evaluation + log_lambda_eval
    log_probs = old_log_probs.reshape((params.num_particles, num_observations + 1), order="C")
    assert np.allclose(old_log_probs[:num_observations + 1], log_probs[0,:num_observations + 1]), f"Log probabilities reshaping was not done correctly:\n OLD: {old_log_probs[:num_observations + 1]}\n\n NEW: {log_probs[0,:num_observations + 1]}"

    log_probs -= np.logaddexp.reduce(log_probs, axis=1)[:,None]
    return foo_utils.CategoricalDistribution(np.exp(log_probs))


def sample_proposal(old_particles : ParticleCloud, older_particles : ParticleCloud, observations, old_observations, params : settings.TVARTrackerParameters):
    old_all_z = old_particles.Z 

    new_assoc = get_assoc_proposal_dist(old_particles, observations, params).sample()[:,None]
    new_z0 = get_zt_proposal_dist(old_particles, observations, new_assoc, params).sample()

    new_all_z = np.zeros(old_all_z.shape)
    new_all_z[:, :, 1:] = old_all_z[:, :, :params.model_order]
    new_all_z[:, :, 0:1] = new_z0 

    new_ma, new_Pa = modified_a_dist_update(old_particles, params)

    temp_particles = ParticleCloud(new_all_z, new_ma, new_Pa, None, None, new_assoc, None)
    new_alpha, new_beta = ig_update(temp_particles, old_particles, older_particles, observations, old_observations, params)

    return ParticleCloud(new_all_z, new_ma, new_Pa, new_alpha, new_beta, new_assoc, None)


def get_log_weight_update(old_particles : ParticleCloud, observations, params : settings.TVARTrackerParameters):
    num_observations = observations.shape[0]
    if num_observations == 0:
        return np.zeros(old_particles.log_weights.shape)

    all_possible_assoc = np.tile(np.arange(start=0, stop=num_observations + 1), params.num_particles)
    tiled_particles = old_particles.tile(num_observations + 1, False)
    tiled_alpha = tiled_particles.alpha 
    tiled_beta = tiled_particles.beta 

    upd_mz, upd_Pz = get_zt_given_ztm1_gaussian_in_zt(tiled_particles, params)
    obs_mz, obs_Pz = get_yt_given_zt_gaussian_in_zt(all_possible_assoc, observations, params)

    prod_mz, prod_Pz, const_offset, const_Pz = foo_utils.gaussian_product(upd_mz, upd_Pz, obs_mz, obs_Pz)
    # Now, the z distribution gets integrated out, so only the offset term matters!

    ys = get_modified_yt(all_possible_assoc, observations, params)
    m_from_offset = ys - const_offset
    df, mu, shape = foo_utils.get_st_params(m_from_offset, const_Pz, tiled_alpha, tiled_beta)

    log_st_evaluation = foo_utils.MultivariateStudentT(mu, shape, df).log_pdf(ys).flatten()
    P_G = 1 # An approximation
    log_prob_arr = np.array([ np.log(1 - P_G * params.measurement_params.observation_probability) if P_G * params.measurement_params.observation_probability < 1 else -np.inf, np.log(params.measurement_params.observation_probability)])
    log_lambda_eval = log_prob_arr[np.where(all_possible_assoc > 0, 1, 0)]

    old_log_probs = log_st_evaluation + log_lambda_eval
    log_probs = old_log_probs.reshape((params.num_particles, num_observations + 1), order="C")
    assert np.allclose(old_log_probs[:num_observations + 1], log_probs[0,:num_observations + 1]), "Log probabilities reshaping was not done correctly"


    log_weight_updates = np.logaddexp.reduce(log_probs, axis=1)

    return log_weight_updates


def modified_a_dist_update(old_particles : ParticleCloud, params : settings.TVARTrackerParameters):
    old_all_z = old_particles.Z 
    old_ma = old_particles.ma 
    old_Pa_tilde = old_particles.Pa 
    old_assoc = old_particles.assoc 
    
    # Note: This is just a KF update

    H = old_all_z[:,:,1:]
    z_0 = old_all_z[:,:,0:1]

    A = np.identity(params.model_order)[None,:,:]
    Q = np.identity(params.model_order)[None,:,:] * params.state_innovation_to_process_innovation
    R = np.identity(params.observation_dimensions)[None,:,:]

    m_minus = A @ old_ma
    P_minus = A @ old_Pa_tilde @ foo_utils.transpose_last2(A) + Q   
    
    v = z_0 - H @ m_minus 
    S = H @ P_minus @ foo_utils.transpose_last2(H) + R 
    K = P_minus @ foo_utils.transpose_last2(H) @ foo_utils.inv(S)
    
    m_post = m_minus + K @ v
    P_post = P_minus - K @ S @ foo_utils.transpose_last2(K)

    # Note: association = 0 should correspond to no posterior update
    new_ma = np.where(old_assoc[:,:,None] > 0, m_post, m_minus) 
    new_Pa_tilde = np.where(old_assoc[:,:,None] > 0, P_post, P_minus)
    
    assert new_ma.shape == old_ma.shape, "Kalman update changes shape of mean"
    assert new_Pa_tilde.shape == old_Pa_tilde.shape, "Kalman update changes shape of covariance"

    return new_ma, new_Pa_tilde



def run_pf_tracking(data : list[np.ndarray], params : settings.TVARTrackerParameters, ground_truth_data=None, ground_truth_ar=None, verbose=True):
    particles = construct_initial_particles(params, ground_truth_data, data[:params.model_order + 1], ground_truth_ar)

    particle_history = [None] * params.simulation_time
    predictive_log_likelihood_history = np.zeros((params.simulation_time))

    if verbose:
        print("Running...")
    particle_history[params.model_order] = particles.copy()

    old_measurements = None 
    older_particles = None

    for current_timestep in tqdm(range(params.model_order + 1, params.simulation_time), disable=not verbose):
        old_particles = particles.copy()
        measurements = data[current_timestep]

        predictive_log_likelihood_history[current_timestep] = evaluate_log_predictive_likelihood(particles, measurements, params)

        particles = sample_proposal(old_particles, older_particles, measurements, old_measurements, params)

        new_log_weights = old_particles.log_weights + get_log_weight_update(old_particles, measurements, params)

        particles.log_weights = new_log_weights - foo_utils.array_logaddexp(new_log_weights)
        log_eff_num_particles = foo_utils.log_effective_num_particles(particles.log_weights)

        # Resampling
        if log_eff_num_particles < np.log(params.min_particles):
            weight_est = np.exp(particles.log_weights)
            weight_est /= np.sum(weight_est)
            particle_choice_indices = np.random.choice(np.arange(params.num_particles), size=params.num_particles, p=weight_est)

            particles = particles.get_choice(particle_choice_indices)
            particles.log_weights = np.ones(params.num_particles) * (-np.log(params.num_particles))

        # Recording
        particle_history[current_timestep] = particles.copy()

        old_measurements = measurements
        if old_particles is not None:
            older_particles = old_particles.copy()

    return particle_history, predictive_log_likelihood_history



def unpack_particle_history(particle_history : list[ParticleCloud], params : settings.TVARTrackerParameters):
    effective_num_particles_history = np.zeros(params.simulation_time)
    a_mean_history = np.zeros((params.simulation_time, params.model_order))
    a_cov_history = np.zeros((params.simulation_time, params.model_order, params.model_order))

    log_weight_history = np.zeros((params.simulation_time, params.num_particles))

    zn_mean_history = np.zeros((params.simulation_time, params.observation_dimensions))
    zn_covariance_history = np.zeros((params.simulation_time, params.observation_dimensions))
    zn_max_history  = np.zeros((params.simulation_time, params.observation_dimensions))
    zn_min_history  = np.zeros((params.simulation_time, params.observation_dimensions))
    zn_path_history = np.zeros((params.simulation_time, params.num_particles, params.observation_dimensions, params.model_order + 1))
    particle_process_coefficient_history = np.zeros((params.simulation_time, params.num_particles, params.model_order))
    particle_coefficient_cov_history = np.zeros((params.simulation_time, params.num_particles, params.model_order, params.model_order))

    alpha_mean_history = np.zeros((params.simulation_time))
    beta_mean_history = np.zeros((params.simulation_time))

    for current_timestep in range(params.simulation_time):
        particles = particle_history[current_timestep]
        if particles is None:
            continue
        log_weights = particles.log_weights

        log_eff_num_particles = foo_utils.log_effective_num_particles(log_weights)
        effective_num_particles_history[current_timestep] = np.exp(log_eff_num_particles)

        all_z, ma, normalised_Pa, alpha, beta = particles.Z, particles.ma, particles.Pa, particles.alpha, particles.beta
        weights = np.exp(log_weights)

        Pa = normalised_Pa * ((particles.beta / (particles.alpha - 1))[:,:,None])

        a_mean_history[current_timestep, :] = np.sum(ma * weights[:, None, None], axis=0).flatten()
        mean_diff = ma[:, :, 0] - a_mean_history[current_timestep, :]
        mean_diff = mean_diff[:, :, None]

        a_cov_history[current_timestep, :, :] = np.sum(Pa * weights[:, None, None] + mean_diff @ foo_utils.transpose_last2(mean_diff) * weights[:, None, None], axis=0)

        zn_mean_history[current_timestep, :] = np.sum(all_z[:, :, 0] * weights[:, None], axis=0)
        zn_covariance_history[current_timestep, :] = np.diag(np.cov(all_z[:, :, 0], ddof=0, aweights=weights, rowvar=False))
        zn_max_history[current_timestep, :] = np.max(all_z[:, :, 0], axis=0)
        zn_min_history[current_timestep, :] = np.min(all_z[:, :, 0], axis=0)
        zn_path_history[current_timestep, :, :, :] = all_z

        particle_process_coefficient_history[current_timestep, :, :] = ma[:, :, 0]
        particle_coefficient_cov_history[current_timestep, :, :, :] = Pa[:, :, :]

        log_weight_history[current_timestep,:] = log_weights

        alpha_mean_history[current_timestep] = np.mean(alpha)
        beta_mean_history[current_timestep] = np.mean(beta)

    return a_mean_history, a_cov_history, zn_mean_history, zn_covariance_history, zn_max_history, zn_min_history, \
            effective_num_particles_history, alpha_mean_history, beta_mean_history, zn_path_history, \
            particle_process_coefficient_history, particle_coefficient_cov_history, log_weight_history


def get_mean_log_evidence(prediction_log_likelihood_history, burn_in=5):
    # TODO: Note this isn't entirely correct. No predictions were made for some of the observations.
    return np.sum(prediction_log_likelihood_history[burn_in:]) / (prediction_log_likelihood_history.size - burn_in)


def get_particle_mse(ground_truth : np.ndarray, particles : ParticleCloud):
    diffs = ground_truth[None,:] - particles.Z[:,:,0]
    assert diffs.shape == (particles.num_particles, ground_truth.size), f"Diffs shape does not match expected shape: {diffs.shape} vs ({particles.num_particles}, {ground_truth.size})"
    
    log_diff_sums = np.log(np.sum(diffs ** 2, axis=1))
    assert log_diff_sums.shape == particles.log_weights[:].shape, f"Log diff sums and particle log weights must have the same shape: {log_diff_sums.shape} vs {particles.log_weights[:].shape}"
    
    prod_terms = log_diff_sums + particles.log_weights[:]
    assert prod_terms.size == particles.num_particles, f"Prod terms size does not match num particles, {prod_terms.size} vs {particles.num_particles}"
    
    log_err = np.logaddexp.reduce(prod_terms)
    return np.exp(log_err) 


def get_particle_rmse(ground_truth : np.ndarray, particles : ParticleCloud):
    return np.sqrt(get_particle_mse(ground_truth, particles))


def get_mean_cumulative_rmse(ground_truth_history : np.ndarray, particle_history : list[ParticleCloud], burn_in : int):
    sim_time = ground_truth_history.shape[0]
    vals = np.zeros(sim_time - burn_in) - np.inf 
    for t in range(burn_in, sim_time):
        vals[t - burn_in] = get_particle_rmse(ground_truth_history[t], particle_history[t])
    return np.mean(vals)


def get_rmse_history(ground_truth_history : np.ndarray, particle_history : list[ParticleCloud], burn_in : int):
    sim_time = ground_truth_history.shape[0]
    vals = np.zeros(sim_time - burn_in) - np.inf 
    for t in range(burn_in, sim_time):
        vals[t - burn_in] = get_particle_rmse(ground_truth_history[t], particle_history[t])
    return vals 


def get_ground_truth_predictive_likelihood(ground_truth_history : np.ndarray, particle_history : list[ParticleCloud], burn_in : int, params : settings.TVARTrackerParameters):
    res = 0
    sim_time = ground_truth_history.shape[0]

    for t in range(burn_in, sim_time - 1):
        res += evaluate_log_predictive_likelihood(particle_history[t], ground_truth_history[None,t + 1, :], params)
    
    return res / (sim_time - 1 - burn_in)


def get_model_performance(cluttered_data : list[np.ndarray], ground_truth : np.ndarray, ground_truth_ar : np.ndarray, burn_in : int, params : settings.TVARTrackerParameters):
    particle_history, log_likelihood_history = run_pf_tracking(cluttered_data, params, ground_truth, ground_truth_ar, False)
    return get_mean_log_evidence(log_likelihood_history, burn_in)


def get_particle_predict_rmse(ground_truth_position : np.ndarray, particles : ParticleCloud, params : settings.TVARTrackerParameters):    
    new_z0_dist = get_zt_proposal_dist(particles, np.zeros((0, params.observation_dimensions)), np.array([0] * particles.num_particles), params)

    # TODO: Do this better with logaddexp methods
    mean_rmse = np.sqrt(new_z0_dist.mse(ground_truth_position) @ np.exp(particles.log_weights))

    return mean_rmse 


def get_pf_predictive_rmse(ground_truth_position : np.ndarray, particle_history : list[ParticleCloud], burn_in : int, params : settings.TVARTrackerParameters):
    rmse_vals = np.zeros(len(particle_history) - 1 - burn_in) - np.inf

    for ts in range(burn_in, len(particle_history) - 1):
        rmse_vals[ts - burn_in] = get_particle_predict_rmse(ground_truth_position[ts + 1,:], particle_history[ts], params)

    return np.mean(rmse_vals)