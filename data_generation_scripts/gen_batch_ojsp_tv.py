import numpy as np
import colorama

import settings
import data_generation_utils
from file_management import FileManager, TestsetManager, DatasetManager

import results

############################################################################
# DATA GENERATION
############################################################################

GENERATION_SEED = 1725833504
print("SEED", GENERATION_SEED)
training_generation_params = settings.DataGenerationParameters(GENERATION_SEED)
training_generation_params.set_current()

training_generation_params.simulation_time = 101
training_generation_params.observation_dimensions = 2

training_generation_params.ar_generation_params.process_order = 3
training_generation_params.ar_generation_params.time_varying = True
training_generation_params.ar_generation_params.innovation_variance = 1E0
training_generation_params.ar_generation_params.phase_uniform_max_pi_factor = 0.4
training_generation_params.ar_generation_params.walk_scale = None 
training_generation_params.ar_generation_params.min_damping = None 
training_generation_params.ar_generation_params.max_damping = -np.log(0.9)

training_generation_params.ar_generation_params.damping_gamma_shape = None 
training_generation_params.ar_generation_params.damping_gamma_scale = None

training_generation_params.ar_generation_params.oscillating = True

training_generation_params.measurement_params.noise_variance = 1E-6
training_generation_params.measurement_params.clutter_rate = 0
training_generation_params.measurement_params.observation_probability = 1


### SETUP THE DATA GENERATION PARAMS, MODIFYING SOME FIELDS FROM THE TRAINING SET

data_generation_params = training_generation_params.copy()
data_generation_params.measurement_params.noise_variance = 4E-3
data_generation_params.measurement_params.clutter_rate = 7E-5


DATASET_NAME = "batch_ojsp_tv"

NUM_TRAINING_SETS = 1
NUM_TESTS = 100
testset = TestsetManager()

for training_ind in range(NUM_TRAINING_SETS):
    data, ground_truth_position, true_process_coefficients = data_generation_utils.generate_ojsp_dataset(training_generation_params)  

    dm = DatasetManager()
    dm.measurement_set = data
    dm.ground_truth = ground_truth_position
    dm.generation_params = training_generation_params
    dm.process_coefficients = true_process_coefficients
    
    if training_ind == 0:
        results.plot_trajectory_plot(None, data, ground_truth_position, training_generation_params, f"{DATASET_NAME}_training_data")
        results.plot_ar_coefficient_estimates(None, None, true_process_coefficients, f"{DATASET_NAME}_process_coefficients")
        results.plot_pole_trajectory(true_process_coefficients, f"{DATASET_NAME}_pole_trajectory")

    testset.add_training_set(dm)


for test_ind in range(NUM_TESTS):
    data, ground_truth_position, true_process_coefficients = data_generation_utils.generate_ojsp_dataset(data_generation_params)  

    dm = DatasetManager()
    dm.measurement_set = data
    dm.ground_truth = ground_truth_position
    dm.generation_params = data_generation_params
    dm.process_coefficients = true_process_coefficients
    
    if test_ind in [0, 5, 10, 15, 20]:
        results.plot_trajectory_plot(None, data, ground_truth_position, data_generation_params, f"{DATASET_NAME}_trajectory_d_{test_ind + 1}" if test_ind == 0 else None)
    testset.add_dataset(dm)


FileManager.write(testset, f"datasets/{DATASET_NAME}")

input(f"{colorama.Fore.GREEN}Done.")