import colorama
import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats
from tqdm import tqdm

plt.close("all")
colorama.init(autoreset=True)


import comparison_models
import data_generation_utils
import settings
import results
import pf_utils
import file_management
import foo_utils

import optimise


TEST_NAME_PREFIX = "batch_ojsp_tv"

testset : file_management.TestsetManager = file_management.FileManager.read(f"datasets/{TEST_NAME_PREFIX}")

############################################################################
# TRACKER SETUP
############################################################################

# TVAR TRACKER PARAMS
TRACKER_SEED = 0
tvar_tracker_params = settings.TVARTrackerParameters(TRACKER_SEED)
tvar_tracker_params.model_order = None

tvar_tracker_params.process_instance_params.innovation_variance = 1
tvar_tracker_params.process_instance_params.innovation_to_noise_ratio = None

tvar_tracker_params.state_innovation_to_process_innovation = 1.1E-2

tvar_tracker_params.num_particles = 1000
tvar_tracker_params.min_particle_factor = 0.1

tvar_tracker_params.a_oracle_prior = False     
tvar_tracker_params.z_oracle_prior = True 
tvar_tracker_params.sigma_oracle_prior = False 

# CV TRACKER PARAMS
cv_tracker_params = settings.ClassicalStructuralTrackerParameters()

cv_tracker_params.z_oracle_prior = True 

cv_tracker_params.state_dynamic_variance = 1E-2
cv_tracker_params.singer_rate = 0

# WNA TRACKER PARAMS
wna_tracker_params = settings.ClassicalStructuralTrackerParameters()

wna_tracker_params.z_oracle_prior = True 

wna_tracker_params.state_dynamic_variance = 5.591
wna_tracker_params.singer_rate = 0


# SINGER TRACKER PARAMS
singer_tracker_params = settings.ClassicalStructuralTrackerParameters()

singer_tracker_params.z_oracle_prior = True 

singer_tracker_params.state_dynamic_variance = 10.0
singer_tracker_params.singer_rate = 2.996


# DOUBLE-SINGER IMM TRACKER PARAMS
imm_tracker_params = settings.IMMTrackerParameters()

imm_tracker_params.z_oracle_prior = True 

imm_tracker_params.model_params = [settings.ClassicalStructuralTrackerParameters() for _ in range(2)]
imm_tracker_params.model_params[0].state_dynamic_variance = 10.0
imm_tracker_params.model_params[0].singer_rate = 2.996
imm_tracker_params.model_params[0].z_oracle_prior = imm_tracker_params.z_oracle_prior 

imm_tracker_params.model_params[1].state_dynamic_variance = 10.0 
imm_tracker_params.model_params[1].singer_rate = 1.134
imm_tracker_params.model_params[1].z_oracle_prior = imm_tracker_params.z_oracle_prior


# COVARIANCE METHOD AR PARAMS
wcm_ar_tracker_params = settings.CovarianceMethodARParameters()
wcm_ar_tracker_params.z_oracle_prior = True   

# Parameter optimised on training set
wcm_ar_tracker_params.state_dynamic_variance = 9.770
# Model order chosen a priori
wcm_ar_tracker_params.model_order = 3


# JIN2017 PARAMS 
jin2017_tracker_params = settings.Jin2017TrackerParameters()
jin2017_tracker_params.z_oracle_prior = True 

# The parameters produced by the optimisation
jin2017_tracker_params.model_order = 4  
jin2017_tracker_params.polynomial_order = 2 
jin2017_tracker_params.window = 10  
jin2017_tracker_params.innovation_variance = 10




burn_in = 10


############################################################################
# RUN TRACKING
############################################################################

NUM_MODELS = 7

TVAR_IND = 0
CV_IND = 1
WNA_IND = 2
SINGER_IND = 3
DS_IMM_IND = 4
WCM_AR_IND = 5
JIN_IND = 6



training_dataset = testset.training_sets[0]
cluttered_data = training_dataset.measurement_set.get_raw_data()

training_data_generation_params = training_dataset.generation_params
training_ground_truth_position = training_dataset.ground_truth


OPTIMISE = False  
if OPTIMISE:
    # TVAR Training
    tvar_tracker_params = settings.inject_data_generation_params(tvar_tracker_params, training_data_generation_params)
    
    tvar_tracker_params.model_order = training_data_generation_params.ar_generation_params.process_order
    tvar_tracker_params.process_instance_params.innovation_to_noise_ratio = tvar_tracker_params.process_instance_params.innovation_variance / training_data_generation_params.measurement_params.noise_variance

    min_boundary, max_boundary = foo_utils.get_bounding_box(training_ground_truth_position, 0.1)
    tvar_tracker_params.measurement_params.set_global_clutter_volume((max_boundary[0] - min_boundary[0]) * (max_boundary[1] - min_boundary[1]))

    tvar_tracker_params = optimise.optimise_tvar_tracker(cluttered_data, training_ground_truth_position, training_dataset.process_coefficients, tvar_tracker_params)

    
    # CV Training
    cv_tracker_params = settings.inject_data_generation_params(cv_tracker_params, training_data_generation_params)
    cv_tracker_params = optimise.optimise_cv_tracker(cluttered_data, training_ground_truth_position, tvar_tracker_params.model_order, cv_tracker_params)

    # WNA Training
    wna_tracker_params = settings.inject_data_generation_params(wna_tracker_params, training_data_generation_params)
    wna_tracker_params = optimise.optimise_wna_tracker(cluttered_data, training_ground_truth_position, tvar_tracker_params.model_order, wna_tracker_params)


    # Singer Training
    singer_tracker_params = settings.inject_data_generation_params(singer_tracker_params, training_data_generation_params)
    singer_tracker_params = optimise.optimise_singer_tracker(cluttered_data, training_ground_truth_position, tvar_tracker_params.model_order, singer_tracker_params)

    # Double-Singer IMM Training
    imm_tracker_params = settings.inject_imm_data_generation_params(imm_tracker_params, training_data_generation_params)
    imm_tracker_params = optimise.optimise_double_singer_imm(cluttered_data, training_ground_truth_position, burn_in, imm_tracker_params)

    # CM-AR Training
    wcm_ar_tracker_params = settings.inject_data_generation_params(wcm_ar_tracker_params, training_data_generation_params)
    wcm_ar_tracker_params = optimise.optimise_wcm_ar_tracker(cluttered_data, training_ground_truth_position, burn_in, wcm_ar_tracker_params)

    # Jin2017 Training
    jin2017_tracker_params = settings.inject_data_generation_params(jin2017_tracker_params, training_data_generation_params)
    jin2017_tracker_params = optimise.optimise_jin2017_tracker(cluttered_data, training_ground_truth_position, burn_in, jin2017_tracker_params, training_data_generation_params)


full_particle_history = [] 
full_cv_model_history = [] 
full_cv_log_observation_prediction = []
full_wna_model_history = [] 
full_wna_log_observation_prediction = []
full_singer_model_history = [] 
full_singer_log_observation_prediction = []
full_wcm_ar_model_history = [] 
full_wcm_ar_log_observation_prediction = []

print("Testing...")


# Reset seeding after running optimisation
tvar_tracker_params.set_current()
jin_mean_history = np.zeros((len(testset.datasets), testset.datasets[0].generation_params.simulation_time, testset.datasets[0].generation_params.observation_dimensions))
jin_cov_history = np.zeros((len(testset.datasets), testset.datasets[0].generation_params.simulation_time, testset.datasets[0].generation_params.observation_dimensions, testset.datasets[0].generation_params.observation_dimensions))
jin_pred_mean_history = np.zeros((len(testset.datasets), testset.datasets[0].generation_params.simulation_time, testset.datasets[0].generation_params.observation_dimensions))
jin_pred_cov_history = np.zeros((len(testset.datasets), testset.datasets[0].generation_params.simulation_time, testset.datasets[0].generation_params.observation_dimensions, testset.datasets[0].generation_params.observation_dimensions))
ds_imm_mean_history = np.zeros((len(testset.datasets), testset.datasets[0].generation_params.simulation_time, testset.datasets[0].generation_params.observation_dimensions))
ds_imm_cov_history = np.zeros((len(testset.datasets), testset.datasets[0].generation_params.simulation_time, testset.datasets[0].generation_params.observation_dimensions, testset.datasets[0].generation_params.observation_dimensions))
ds_imm_pred_mean_history = np.zeros((len(testset.datasets), testset.datasets[0].generation_params.simulation_time, testset.datasets[0].generation_params.observation_dimensions))
ds_imm_pred_cov_history = np.zeros((len(testset.datasets), testset.datasets[0].generation_params.simulation_time, testset.datasets[0].generation_params.observation_dimensions, testset.datasets[0].generation_params.observation_dimensions))

for dataset_ind, dm in enumerate(testset.datasets):
    print(f"{colorama.Fore.GREEN}Testcase {colorama.Fore.LIGHTYELLOW_EX}{dataset_ind + 1} / {len(testset.datasets)}")

    data : data_generation_utils.MeasurementSet = dm.measurement_set
    ground_truth_position : np.ndarray = dm.ground_truth
    true_process_coefficients : np.ndarray = dm.process_coefficients
    data_generation_params = dm.generation_params

    cluttered_data = data.get_raw_data()

    # TVAR Run
    tvar_tracker_params = settings.inject_data_generation_params(tvar_tracker_params, data_generation_params)

    tvar_tracker_params.model_order = data_generation_params.ar_generation_params.process_order 

    tvar_tracker_params.process_instance_params.innovation_to_noise_ratio = tvar_tracker_params.process_instance_params.innovation_variance / data_generation_params.measurement_params.noise_variance
    tvar_tracker_params.process_instance_params.innovation_to_noise_ratio = data_generation_params.ar_generation_params.innovation_variance / data_generation_params.measurement_params.noise_variance

    min_boundary, max_boundary = foo_utils.get_bounding_box(ground_truth_position, 0.1)
    tvar_tracker_params.measurement_params.set_global_clutter_volume((max_boundary[0] - min_boundary[0]) * (max_boundary[1] - min_boundary[1]))

    tvar_tracker_params.set_current()

    particle_history, _ = pf_utils.run_pf_tracking(cluttered_data, tvar_tracker_params, ground_truth_position, true_process_coefficients)
    full_particle_history.append(particle_history)

    # CV Run
    cv_tracker_params = settings.inject_data_generation_params(cv_tracker_params, data_generation_params)
    cv_model = comparison_models.setup_cv(ground_truth_position, cv_tracker_params)
    cv_model_history, cv_log_observation_prediction = comparison_models.kf_with_nn_association(cv_model, cluttered_data, cv_tracker_params)
    full_cv_model_history.append(cv_model_history)
    full_cv_log_observation_prediction.append(cv_log_observation_prediction)

    # WNA Run
    wna_tracker_params = settings.inject_data_generation_params(wna_tracker_params, data_generation_params)
    wna_model = comparison_models.setup_wna(ground_truth_position, wna_tracker_params)
    wna_model_history, wna_log_observation_prediction = comparison_models.kf_with_nn_association(wna_model, cluttered_data, wna_tracker_params)
    full_wna_model_history.append(wna_model_history)
    full_wna_log_observation_prediction.append(wna_log_observation_prediction)

    # Singer Run
    singer_tracker_params = settings.inject_data_generation_params(singer_tracker_params, data_generation_params)
    singer_model = comparison_models.setup_singer(ground_truth_position, singer_tracker_params)
    singer_model_history, singer_log_observation_prediction = comparison_models.kf_with_nn_association(singer_model, cluttered_data, singer_tracker_params)
    full_singer_model_history.append(singer_model_history)
    full_singer_log_observation_prediction.append(singer_log_observation_prediction)

    # DS-IMM Run
    imm_tracker_params = settings.inject_imm_data_generation_params(imm_tracker_params, data_generation_params)
    imm_model = comparison_models.setup_double_singer_imm(ground_truth_position, imm_tracker_params)

    ds_imm_mean_history[dataset_ind], ds_imm_cov_history[dataset_ind], ds_imm_pred_mean_history[dataset_ind], \
        ds_imm_pred_cov_history[dataset_ind], _ = comparison_models.imm_with_nn_association(imm_model, cluttered_data, imm_tracker_params)    

    # CM-AR Run
    wcm_ar_tracker_params = settings.inject_data_generation_params(wcm_ar_tracker_params, data_generation_params)
    wcm_ar_model = comparison_models.setup_windowed_covariance_method_ar(ground_truth_position, wcm_ar_tracker_params)
    wcm_ar_model_history, wcm_ar_log_observation_prediction = comparison_models.kf_with_nn_association(wcm_ar_model, cluttered_data, wcm_ar_tracker_params)
    full_wcm_ar_model_history.append(wcm_ar_model_history)
    full_wcm_ar_log_observation_prediction.append(wcm_ar_log_observation_prediction)

    # Jin Run
    jin2017_tracker_params = settings.inject_data_generation_params(jin2017_tracker_params, data_generation_params)

    jin_mean_history[dataset_ind], jin_cov_history[dataset_ind], jin_pred_mean_history[dataset_ind], \
        jin_pred_cov_history[dataset_ind], _ = comparison_models.run_Jin2017_tracker_with_nn_association(cluttered_data, ground_truth_position, data_generation_params, jin2017_tracker_params)






rmse_results = np.zeros((NUM_MODELS, len(testset.datasets)))
predictive_rmse_results = np.zeros((NUM_MODELS, len(testset.datasets)))



for dataset_ind, dm in enumerate(testset.datasets):
    print(f"Processing dataset {colorama.Fore.LIGHTYELLOW_EX}{dataset_ind + 1}")
    ############################################################################
    # REPORTING DATA
    ############################################################################
    data : data_generation_utils.MeasurementSet = dm.measurement_set
    ground_truth_position : np.ndarray = dm.ground_truth
    true_process_coefficients : np.ndarray = dm.process_coefficients
    data_generation_params = dm.generation_params

    
    # TVAR
    tvar_tracker_params = settings.inject_data_generation_params(tvar_tracker_params, data_generation_params)
    
    tvar_tracker_params.process_instance_params.innovation_to_noise_ratio = tvar_tracker_params.process_instance_params.innovation_variance / data_generation_params.measurement_params.noise_variance
    tvar_tracker_params.process_instance_params.innovation_to_noise_ratio = data_generation_params.ar_generation_params.innovation_variance / data_generation_params.measurement_params.noise_variance

    min_boundary, max_boundary = foo_utils.get_bounding_box(ground_truth_position, 0.1)
    tvar_tracker_params.measurement_params.set_global_clutter_volume((max_boundary[0] - min_boundary[0]) * (max_boundary[1] - min_boundary[1]))

    tvar_tracker_params.set_current()

    cv_tracker_params = settings.inject_data_generation_params(cv_tracker_params, data_generation_params)
    wna_tracker_params = settings.inject_data_generation_params(wna_tracker_params, data_generation_params)
    singer_tracker_params = settings.inject_data_generation_params(singer_tracker_params, data_generation_params)
    wcm_ar_tracker_params = settings.inject_data_generation_params(wcm_ar_tracker_params, data_generation_params)
    jin2017_tracker_params = settings.inject_data_generation_params(jin2017_tracker_params, data_generation_params)

    # TVAR
    particle_history = full_particle_history[dataset_ind]

    tvar_error = pf_utils.get_mean_cumulative_rmse(ground_truth_position, particle_history, burn_in) / np.sqrt(tvar_tracker_params.measurement_params.noise_variance)
    tvar_predictive_error = pf_utils.get_pf_predictive_rmse(ground_truth_position, particle_history, burn_in, tvar_tracker_params) / np.sqrt(tvar_tracker_params.measurement_params.noise_variance)
    print(f"TVAR error: {colorama.Fore.LIGHTYELLOW_EX}{tvar_error}")
    print(f"TVAR pred. error: {colorama.Fore.LIGHTYELLOW_EX}{tvar_predictive_error}")
    print("")

    a_mean_history, a_cov_history, zn_mean_history, zn_cov_history, \
        zn_max_history, zn_min_history, effective_num_particles_history, \
            alpha_mean_history, beta_mean_history, zn_path_history, \
                particle_process_coefficient_history, particle_coefficient_cov_history, log_weight_history = pf_utils.unpack_particle_history(particle_history, tvar_tracker_params)


    if dataset_ind == 0:
        results.plot_ar_coefficient_estimates(a_mean_history, a_cov_history, dm.process_coefficients, f"{TEST_NAME_PREFIX}_process_coefficient_tracking")
        if not tvar_tracker_params.sigma_oracle_prior:
            results.plot_sigma_estimates(alpha_mean_history, beta_mean_history, data_generation_params.ar_generation_params.innovation_variance, f"{TEST_NAME_PREFIX}_sigma_tracking")
        for t in [20, 50, 80]:
            results.plot_pole_dist(particle_process_coefficient_history[t], 
                                        particle_coefficient_cov_history[t], 
                                        log_weight_history[t], dm.process_coefficients[t], f"{TEST_NAME_PREFIX}_d1_pole_dist_{t}", 
                                        [[0.1, 1.2], [-1.1, 1.1]])

    rmse_results[TVAR_IND,dataset_ind] = tvar_error
    predictive_rmse_results[TVAR_IND,dataset_ind] = tvar_predictive_error

    # CV
    cv_model_history = full_cv_model_history[dataset_ind]
    cv_log_observation_prediction = full_cv_log_observation_prediction[dataset_ind]

    cv_state_mean_history, cv_state_cov_history, cv_state_to_obs_mean_history, \
        cv_state_to_obs_cov_history, cv_pred_mean_history, cv_pred_cov_history = comparison_models.unpack_model_history(cv_model_history, cv_tracker_params)
    cv_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position, cv_state_to_obs_mean_history, cv_state_to_obs_cov_history, burn_in) / np.sqrt(cv_tracker_params.measurement_params.noise_variance)
    cv_predictive_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position[1:,:], cv_pred_mean_history[:-1,:], cv_pred_cov_history, burn_in) / np.sqrt(cv_tracker_params.measurement_params.noise_variance)
    print(f"CV error: {colorama.Fore.LIGHTYELLOW_EX}{cv_error}")
    print(f"CV pred. error: {colorama.Fore.LIGHTYELLOW_EX}{cv_predictive_error}")
    print("")
    rmse_results[CV_IND,dataset_ind] = cv_error
    predictive_rmse_results[CV_IND,dataset_ind] = cv_predictive_error


    # WNA
    wna_model_history = full_wna_model_history[dataset_ind]
    wna_log_observation_prediction = full_wna_log_observation_prediction[dataset_ind]

    wna_state_mean_history, wna_state_cov_history, wna_state_to_obs_mean_history, \
        wna_state_to_obs_cov_history, wna_pred_mean_history, wna_pred_cov_history = comparison_models.unpack_model_history(wna_model_history, wna_tracker_params)
    wna_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position, wna_state_to_obs_mean_history, wna_state_to_obs_cov_history, burn_in) / np.sqrt(wna_tracker_params.measurement_params.noise_variance)
    wna_predictive_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position[1:,:], wna_pred_mean_history[:-1,:], wna_pred_cov_history, burn_in) / np.sqrt(wna_tracker_params.measurement_params.noise_variance)
    print(f"WNA error: {colorama.Fore.LIGHTYELLOW_EX}{wna_error}")
    print(f"WNA pred. error: {colorama.Fore.LIGHTYELLOW_EX}{wna_predictive_error}")
    print("")
    rmse_results[WNA_IND,dataset_ind] = wna_error
    predictive_rmse_results[WNA_IND,dataset_ind] = wna_predictive_error
        

    # Singer
    singer_model_history = full_singer_model_history[dataset_ind]
    singer_log_observation_prediction = full_singer_log_observation_prediction[dataset_ind]

    singer_state_mean_history, singer_state_cov_history, singer_state_to_obs_mean_history, \
        singer_state_to_obs_cov_history, singer_pred_mean_history, singer_pred_cov_history = comparison_models.unpack_model_history(singer_model_history, singer_tracker_params)
    singer_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position, singer_state_to_obs_mean_history, singer_state_to_obs_cov_history, burn_in) / np.sqrt(singer_tracker_params.measurement_params.noise_variance)
    singer_predictive_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position[1:,:], singer_pred_mean_history[:-1,:], singer_pred_cov_history, burn_in) / np.sqrt(singer_tracker_params.measurement_params.noise_variance)
    print(f"Singer error: {colorama.Fore.LIGHTYELLOW_EX}{singer_error}")
    print(f"Singer pred. error: {colorama.Fore.LIGHTYELLOW_EX}{singer_predictive_error}")
    print("")
    rmse_results[SINGER_IND,dataset_ind] = singer_error
    predictive_rmse_results[SINGER_IND,dataset_ind] = singer_predictive_error 


    # DS-IMM
    ds_imm_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position, ds_imm_mean_history[dataset_ind], ds_imm_cov_history[dataset_ind], burn_in) / np.sqrt(wna_tracker_params.measurement_params.noise_variance)
    ds_imm_predictive_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position[1:,:], ds_imm_pred_mean_history[dataset_ind,:-1,:], ds_imm_pred_cov_history[dataset_ind], burn_in) / np.sqrt(wna_tracker_params.measurement_params.noise_variance)
    print(f"IMM error: {colorama.Fore.LIGHTYELLOW_EX}{ds_imm_error}")
    print(f"IMM pred. error: {colorama.Fore.LIGHTYELLOW_EX}{ds_imm_predictive_error}")
    print("")
    rmse_results[DS_IMM_IND,dataset_ind] = ds_imm_error
    predictive_rmse_results[DS_IMM_IND,dataset_ind] = ds_imm_predictive_error
    
    
    # CM-AR
    wcm_ar_model_history = full_wcm_ar_model_history[dataset_ind]
    wcm_ar_log_observation_prediction = full_wcm_ar_log_observation_prediction[dataset_ind]

    wcm_ar_state_mean_history, wcm_ar_state_cov_history, wcm_ar_state_to_obs_mean_history, \
        wcm_ar_state_to_obs_cov_history, wcm_ar_pred_mean_history, wcm_ar_pred_cov_history = comparison_models.unpack_model_history(wcm_ar_model_history, wcm_ar_tracker_params)
    wcm_ar_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position, wcm_ar_state_to_obs_mean_history, wcm_ar_state_to_obs_cov_history, burn_in) / np.sqrt(wcm_ar_tracker_params.measurement_params.noise_variance)
    wcm_ar_predictive_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position[1:,:], wcm_ar_pred_mean_history[:-1,:], wcm_ar_pred_cov_history, burn_in) / np.sqrt(wcm_ar_tracker_params.measurement_params.noise_variance)
    print(f"CM-AR error: {colorama.Fore.LIGHTYELLOW_EX}{wcm_ar_error}")
    print(f"CM-AR pred. error: {colorama.Fore.LIGHTYELLOW_EX}{wcm_ar_predictive_error}")
    print("")
    rmse_results[WCM_AR_IND,dataset_ind] = wcm_ar_error
    predictive_rmse_results[WCM_AR_IND,dataset_ind] = wcm_ar_predictive_error


    # Jin
    jin_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position, jin_mean_history[dataset_ind], jin_cov_history[dataset_ind], burn_in) / np.sqrt(wna_tracker_params.measurement_params.noise_variance)
    jin_predictive_error = comparison_models.get_mean_cumulative_rmse_gauss(ground_truth_position[1:,:], jin_pred_mean_history[dataset_ind,:-1,:], jin_pred_cov_history[dataset_ind], burn_in) / np.sqrt(wna_tracker_params.measurement_params.noise_variance)
    print(f"JIN error: {colorama.Fore.LIGHTYELLOW_EX}{jin_error}")
    print(f"JIN pred. error: {colorama.Fore.LIGHTYELLOW_EX}{jin_predictive_error}")
    print("")
    rmse_results[JIN_IND,dataset_ind] = jin_error
    predictive_rmse_results[JIN_IND,dataset_ind] = jin_predictive_error

    
        


print("TVAR result ranges")
print(f"RMSE: {colorama.Fore.LIGHTGREEN_EX}{np.min(rmse_results[TVAR_IND,:])} -> {colorama.Fore.LIGHTRED_EX}{np.max(rmse_results[TVAR_IND,:])}")
print("")


print("CV result ranges")
print(f"RMSE: {colorama.Fore.LIGHTGREEN_EX}{np.min(rmse_results[CV_IND,:])} -> {colorama.Fore.LIGHTRED_EX}{np.max(rmse_results[CV_IND,:])}")
print("")


print("WNA result ranges")
print(f"RMSE: {colorama.Fore.LIGHTGREEN_EX}{np.min(rmse_results[WNA_IND,:])} -> {colorama.Fore.LIGHTRED_EX}{np.max(rmse_results[WNA_IND,:])}")
print("")


print("Singer result ranges")
print(f"RMSE: {colorama.Fore.LIGHTGREEN_EX}{np.min(rmse_results[SINGER_IND,:])} -> {colorama.Fore.LIGHTRED_EX}{np.max(rmse_results[SINGER_IND,:])}")
print("")

print("DS-IMM result ranges")
print(f"RMSE: {colorama.Fore.LIGHTGREEN_EX}{np.min(rmse_results[DS_IMM_IND,:])} -> {colorama.Fore.LIGHTRED_EX}{np.max(rmse_results[DS_IMM_IND,:])}")
print("")

print("WCM-AR result ranges")
print(f"RMSE: {colorama.Fore.LIGHTGREEN_EX}{np.min(rmse_results[WCM_AR_IND,:])} -> {colorama.Fore.LIGHTRED_EX}{np.max(rmse_results[WCM_AR_IND,:])}")
print("")

print("JIN result ranges")
print(f"RMSE: {colorama.Fore.LIGHTGREEN_EX}{np.min(rmse_results[JIN_IND,:])} -> {colorama.Fore.LIGHTRED_EX}{np.max(rmse_results[JIN_IND,:])}")
print("")


print("RMSE means:", np.mean(rmse_results, axis=1))
print("RMSE Rank averages:", np.mean(foo_utils.get_rankings(rmse_results, False), axis=1))
print("Pred RMSE means:", np.mean(predictive_rmse_results, axis=1))
print("Pred RMSE Rank averages:", np.mean(foo_utils.get_rankings(predictive_rmse_results, False), axis=1))


result_table = np.zeros((NUM_MODELS, 4))
result_table[:,0] = np.mean(rmse_results, axis=1)
result_table[:,1] = np.mean(foo_utils.get_rankings(rmse_results, False), axis=1)
result_table[:,2] = np.mean(predictive_rmse_results, axis=1)
result_table[:,3] = np.mean(foo_utils.get_rankings(predictive_rmse_results, False), axis=1)

results.export_result_table(result_table, TEST_NAME_PREFIX + "_rmse")

input(f"{colorama.Fore.GREEN}Done.")