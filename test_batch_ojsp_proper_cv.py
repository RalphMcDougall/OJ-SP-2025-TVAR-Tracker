import colorama
import matplotlib.pyplot as plt 
import numpy as np

plt.close("all")
colorama.init(autoreset=True)


import classic_models
import data_generation_utils
import settings
import results
import pf_utils
import file_management
import foo_utils

TEST_NAME_PREFIX = "batch_ojsp_proper_cv"

testset : file_management.TestsetManager = file_management.FileManager.read(f"datasets/{TEST_NAME_PREFIX}")

############################################################################
# TRACKER SETUP
############################################################################

# TVAR TRACKER PARAMS
TRACKER_SEED = 0
tvar_tracker_params = settings.TVARTrackerParameters(TRACKER_SEED)
tvar_tracker_params.model_order = 2
tvar_tracker_params.simulation_time = None 
tvar_tracker_params.observation_dimensions = None

tvar_tracker_params.measurement_params = None

tvar_tracker_params.process_instance_params.innovation_variance = testset.datasets[0].generation_params.ar_generation_params.innovation_variance
tvar_tracker_params.process_instance_params.innovation_to_noise_ratio = None

tvar_tracker_params.state_innovation_to_process_innovation = 1E-5

tvar_tracker_params.num_particles = 1000
tvar_tracker_params.min_particle_factor = 0.1

tvar_tracker_params.a_oracle_prior = False     
tvar_tracker_params.z_oracle_prior = True 
tvar_tracker_params.sigma_oracle_prior = False 


# WNA TRACKER PARAMS
wna_tracker_params = settings.ClassicalStructuralTrackerParameters()
wna_tracker_params.simulation_time = None
wna_tracker_params.observation_dimensions = None

wna_tracker_params.z_oracle_prior = True 

wna_tracker_params.state_dynamic_variance = 1E0
wna_tracker_params.singer_rate = None

wna_tracker_params.measurement_params = None


############################################################################
# RUN TRACKING
############################################################################

TVAR_IND = 0
WNA_IND = 1


training_dataset = testset.training_sets[0]
cluttered_data = training_dataset.measurement_set.get_raw_data()

training_data_generation_params = training_dataset.generation_params
training_ground_truth_position = training_dataset.ground_truth


full_particle_history = [] 
full_predictive_log_likelihood_history = [] 
full_wna_model_history = [] 
full_wna_log_observation_prediction = []


print("Testing...")


# Reset seeding after running optimisation
tvar_tracker_params.set_current()


for dataset_ind, dm in enumerate(testset.datasets):
    print(f"{colorama.Fore.GREEN}Testcase {colorama.Fore.LIGHTYELLOW_EX}{dataset_ind + 1} / {len(testset.datasets)}")

    data : data_generation_utils.MeasurementSet = dm.measurement_set
    ground_truth_position : np.ndarray = dm.ground_truth
    true_process_coefficients : np.ndarray = dm.process_coefficients
    data_generation_params = dm.generation_params

    cluttered_data = data.get_raw_data()


    #tvar_tracker_params.model_order = data_generation_params.ar_generation_params.process_order
    tvar_tracker_params.simulation_time = data_generation_params.simulation_time
    tvar_tracker_params.observation_dimensions = data_generation_params.observation_dimensions

    tvar_tracker_params.measurement_params = data_generation_params.measurement_params.copy()

    tvar_tracker_params.process_instance_params.innovation_to_noise_ratio = data_generation_params.ar_generation_params.innovation_variance / data_generation_params.measurement_params.noise_variance

    min_boundary, max_boundary = foo_utils.get_bounding_box(ground_truth_position, 0.1)
    tvar_tracker_params.measurement_params.set_global_clutter_volume((max_boundary[0] - min_boundary[0]) * (max_boundary[1] - min_boundary[1]))

    tvar_tracker_params.set_current()

    particle_history, predictive_log_likelihood_history = pf_utils.run_pf_tracking(cluttered_data, tvar_tracker_params, ground_truth_position, true_process_coefficients)
    full_particle_history.append(particle_history)
    full_predictive_log_likelihood_history.append(predictive_log_likelihood_history)
    

    wna_tracker_params.simulation_time = data_generation_params.simulation_time
    wna_tracker_params.observation_dimensions = data_generation_params.observation_dimensions

    wna_tracker_params.measurement_params = data_generation_params.measurement_params.copy()

    wna_model = classic_models.setup_wna(ground_truth_position, wna_tracker_params)
    wna_model_history, wna_log_observation_prediction = classic_models.kf_with_nn_association(wna_model, cluttered_data, wna_tracker_params)
    full_wna_model_history.append(wna_model_history)
    full_wna_log_observation_prediction.append(wna_log_observation_prediction)



log_lik_results = np.zeros((2, len(testset.datasets)))
rmse_results = np.zeros((2, len(testset.datasets)))
predictive_rmse_results = np.zeros((2, len(testset.datasets)))
gt_lik_results = np.zeros((2, len(testset.datasets)))

for dataset_ind, dm in enumerate(testset.datasets):
    print(f"Processing dataset {colorama.Fore.LIGHTYELLOW_EX}{dataset_ind + 1}")
    ############################################################################
    # REPORTING DATA
    ############################################################################
    data : data_generation_utils.MeasurementSet = dm.measurement_set
    ground_truth_position : np.ndarray = dm.ground_truth
    true_process_coefficients : np.ndarray = dm.process_coefficients
    data_generation_params = dm.generation_params

    #tvar_tracker_params.model_order = data_generation_params.ar_generation_params.process_order
    tvar_tracker_params.simulation_time = data_generation_params.simulation_time
    tvar_tracker_params.observation_dimensions = data_generation_params.observation_dimensions

    tvar_tracker_params.measurement_params = data_generation_params.measurement_params.copy()

    tvar_tracker_params.process_instance_params.innovation_to_noise_ratio = data_generation_params.ar_generation_params.innovation_variance / data_generation_params.measurement_params.noise_variance

    min_boundary, max_boundary = foo_utils.get_bounding_box(ground_truth_position, 0.1)
    tvar_tracker_params.measurement_params.set_global_clutter_volume((max_boundary[0] - min_boundary[0]) * (max_boundary[1] - min_boundary[1]))

    tvar_tracker_params.set_current()


    wna_tracker_params.simulation_time = data_generation_params.simulation_time
    wna_tracker_params.observation_dimensions = data_generation_params.observation_dimensions

    wna_tracker_params.measurement_params = data_generation_params.measurement_params.copy()


    burn_in = 10

    particle_history = full_particle_history[dataset_ind]
    predictive_log_likelihood_history = full_predictive_log_likelihood_history[dataset_ind]

    tvar_log_likelihood = pf_utils.get_mean_log_evidence(predictive_log_likelihood_history)
    print(f"TVAR mean log-likelihood: {colorama.Fore.GREEN}{tvar_log_likelihood}")
    tvar_error = pf_utils.get_mean_cumulative_rmse(ground_truth_position, particle_history, burn_in) / np.sqrt(tvar_tracker_params.measurement_params.noise_variance)
    tvar_predictive_error = pf_utils.get_pf_predictive_rmse(ground_truth_position, particle_history, burn_in, tvar_tracker_params) / np.sqrt(tvar_tracker_params.measurement_params.noise_variance)
    print(f"TVAR error: {colorama.Fore.LIGHTYELLOW_EX}{tvar_error}")
    print(f"TVAR pred. error: {colorama.Fore.LIGHTYELLOW_EX}{tvar_predictive_error}")
    tvar_pred_lik = pf_utils.get_ground_truth_predictive_likelihood(ground_truth_position, particle_history, burn_in, tvar_tracker_params)
    print("")

    a_mean_history, a_cov_history, zn_mean_history, zn_cov_history, \
        zn_max_history, zn_min_history, effective_num_particles_history, \
            alpha_mean_history, beta_mean_history, zn_path_history, \
                particle_process_coefficient_history, particle_coefficient_cov_history, log_weight_history = pf_utils.unpack_particle_history(particle_history, tvar_tracker_params)


    if dataset_ind == 0:
        results.plot_sigma_estimates(alpha_mean_history, beta_mean_history, None, f"{TEST_NAME_PREFIX}_sigma_tracking")
        proc_coefs = np.zeros(a_mean_history.shape)
        proc_coefs[:,0] = 2
        proc_coefs[:,1] = -1
        results.plot_ar_coefficient_estimates_with_cp(a_mean_history, a_cov_history, proc_coefs, f"{TEST_NAME_PREFIX}_process_coefficient_tracking")
        

    log_lik_results[TVAR_IND,dataset_ind] = pf_utils.get_mean_log_evidence(predictive_log_likelihood_history)
    rmse_results[TVAR_IND,dataset_ind] = tvar_error
    predictive_rmse_results[TVAR_IND,dataset_ind] = tvar_predictive_error
    gt_lik_results[TVAR_IND,dataset_ind] = pf_utils.get_ground_truth_predictive_likelihood(ground_truth_position, particle_history, burn_in, tvar_tracker_params)


    wna_model_history = full_wna_model_history[dataset_ind]
    wna_log_observation_prediction = full_wna_log_observation_prediction[dataset_ind]

    wna_log_likelihood = pf_utils.get_mean_log_evidence(wna_log_observation_prediction)
    print(f"WNA mean log-likelihood: {colorama.Fore.GREEN}{wna_log_likelihood}")
    wna_state_mean_history, wna_state_cov_history, wna_state_to_obs_mean_history, \
        wna_state_to_obs_cov_history, wna_pred_mean_history, wna_pred_cov_history = classic_models.unpack_model_history(wna_model_history, wna_tracker_params)
    wna_error = classic_models.get_mean_cumulative_rmse_gauss(ground_truth_position, wna_state_to_obs_mean_history, wna_state_to_obs_cov_history, burn_in) / np.sqrt(wna_tracker_params.measurement_params.noise_variance)
    wna_predictive_error = classic_models.get_mean_cumulative_rmse_gauss(ground_truth_position[1:,:], wna_pred_mean_history[:-1,:], wna_pred_cov_history, burn_in) / np.sqrt(wna_tracker_params.measurement_params.noise_variance)
    print(f"WNA error: {colorama.Fore.LIGHTYELLOW_EX}{wna_error}")
    print(f"WNA pred. error: {colorama.Fore.LIGHTYELLOW_EX}{wna_predictive_error}")
    wna_pred_lik = classic_models.get_ground_truth_predictive_likelihood_gauss(ground_truth_position, wna_pred_mean_history, wna_pred_cov_history, max(2, burn_in))
    print("")
    log_lik_results[WNA_IND,dataset_ind] = wna_log_likelihood
    rmse_results[WNA_IND,dataset_ind] = wna_error
    predictive_rmse_results[WNA_IND,dataset_ind] = wna_predictive_error
    gt_lik_results[WNA_IND,dataset_ind] = wna_pred_lik 
        


print("TVAR result ranges")
print(f"LL: {colorama.Fore.LIGHTRED_EX}{np.min(log_lik_results[TVAR_IND,:])} -> {colorama.Fore.LIGHTGREEN_EX}{np.max(log_lik_results[TVAR_IND,:])}")
print(f"RMSE: {colorama.Fore.LIGHTGREEN_EX}{np.min(rmse_results[TVAR_IND,:])} -> {colorama.Fore.LIGHTRED_EX}{np.max(rmse_results[TVAR_IND,:])}")
print("")


print("WNA result ranges")
print(f"LL: {colorama.Fore.LIGHTRED_EX}{np.min(log_lik_results[WNA_IND,:])} -> {colorama.Fore.LIGHTGREEN_EX}{np.max(log_lik_results[WNA_IND,:])}")
print(f"RMSE: {colorama.Fore.LIGHTGREEN_EX}{np.min(rmse_results[WNA_IND,:])} -> {colorama.Fore.LIGHTRED_EX}{np.max(rmse_results[WNA_IND,:])}")
print("")


print("RMSE Rank averages:", np.mean(foo_utils.get_rankings(rmse_results, False), axis=1))
print("RMSE means:", np.mean(rmse_results, axis=1))
print("Pred RMSE Rank averages:", np.mean(foo_utils.get_rankings(predictive_rmse_results, False), axis=1))
print("Pred RMSE means:", np.mean(predictive_rmse_results, axis=1))


result_table = np.zeros((2, 2))
result_table[:,0] = np.mean(rmse_results, axis=1)
result_table[:,1] = np.mean(predictive_rmse_results, axis=1)

results.export_result_table(result_table, TEST_NAME_PREFIX + "_rmse")

input("")