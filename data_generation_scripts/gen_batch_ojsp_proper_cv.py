import numpy as np
import colorama

import settings
import data_generation_utils
import file_management

import results

############################################################################
# DATA GENERATION
############################################################################

GENERATION_SEED = 1725833889
print("SEED", GENERATION_SEED)
training_generation_params = settings.DataGenerationParameters(GENERATION_SEED)
training_generation_params.set_current()

training_generation_params.simulation_time = 250
training_generation_params.observation_dimensions = 2

training_generation_params.ar_generation_params.process_order = None 
training_generation_params.ar_generation_params.time_varying = None 
training_generation_params.ar_generation_params.innovation_variance = 1E0
training_generation_params.ar_generation_params.phase_uniform_max_pi_factor = None 
training_generation_params.ar_generation_params.walk_scale = None 
training_generation_params.ar_generation_params.min_damping = None 
training_generation_params.ar_generation_params.max_damping = None 

training_generation_params.ar_generation_params.damping_gamma_shape = None 
training_generation_params.ar_generation_params.damping_gamma_scale = None

training_generation_params.ar_generation_params.oscillating = None

training_generation_params.measurement_params.noise_variance = 1E-6
training_generation_params.measurement_params.clutter_rate = 0
training_generation_params.measurement_params.observation_probability = 1


### SETUP THE DATA GENERATION PARAMS, MODIFYING SOME FIELDS FROM THE TRAINING SET

data_generation_params = training_generation_params.copy()
data_generation_params.measurement_params.noise_variance = 5E-2
data_generation_params.measurement_params.clutter_rate = 1E-6


DATASET_NAME = "batch_ojsp_proper_cv"

NUM_TRAINING_SETS = 1
NUM_TESTS = 20
testset = file_management.TestsetManager()

for training_ind in range(NUM_TRAINING_SETS):
    data, ground_truth_position, true_process_coefficients = data_generation_utils.generate_real_cv_dataset(training_generation_params)  

    dm = file_management.DatasetManager()
    dm.measurement_set = data
    dm.ground_truth = ground_truth_position
    dm.generation_params = training_generation_params
    dm.process_coefficients = true_process_coefficients
    
    if training_ind == 0:
        results.plot_trajectory_plot(None, data, ground_truth_position, training_generation_params, f"{DATASET_NAME}_training_data")

    testset.add_training_set(dm)


for test_ind in range(NUM_TESTS):
    data, ground_truth_position, true_process_coefficients = data_generation_utils.generate_real_cv_dataset(data_generation_params)  

    dm = file_management.DatasetManager()
    dm.measurement_set = data
    dm.ground_truth = ground_truth_position
    dm.generation_params = data_generation_params
    dm.process_coefficients = true_process_coefficients
    
    if test_ind % 5 == 0:
        results.plot_trajectory_plot(None, data, ground_truth_position, data_generation_params, None)
    testset.add_dataset(dm)


file_management.FileManager.write(testset, f"datasets/{DATASET_NAME}")

input(f"{colorama.Fore.GREEN}Done.")