import colorama
import numpy as np
from tqdm import tqdm

import settings 
import pf_utils 
import classic_models


def optimise_tvar_tracker(cluttered_data, ground_truth_position, true_process_coefficients, tvar_tracker_params : settings.TVARTrackerParameters):
    print(f"{colorama.Fore.CYAN}Optimising TVAR")
    inn_to_noise_range = np.logspace(start=-1, stop=1, num=25)

    best_value = -np.inf 
    best_pars = None 
    best_params = None 
    if tvar_tracker_params.sigma_oracle_prior:
        for pars in tqdm(inn_to_noise_range):
            test_params = tvar_tracker_params.copy()
            test_params.set_current()
            test_params.process_instance_params.innovation_variance = pars
            test_params.process_instance_params.innovation_to_noise_ratio = test_params.process_instance_params.innovation_variance / test_params.measurement_params.noise_variance
            value = pf_utils.get_model_performance(cluttered_data, ground_truth_position, true_process_coefficients, test_params.model_order, test_params)
            if value > best_value:
                best_value = value 
                best_pars = pars
                best_params = test_params  
        
        if inn_to_noise_range.size > 1 and best_pars in [min(inn_to_noise_range), max(inn_to_noise_range)]:
            print(f"{colorama.Fore.LIGHTYELLOW_EX}Warning: the optimal parameter is on the boundary of the search region.")
        print(f"Best value: {colorama.Fore.GREEN}{best_value} {colorama.Fore.WHITE}at {colorama.Fore.LIGHTYELLOW_EX}{best_pars}")
    else:
        best_params = tvar_tracker_params.copy()
    return best_params 


def optimise_cv_tracker(cluttered_data, ground_truth_position, burn_in, cv_tracker_params : settings.ClassicalStructuralTrackerParameters):
    print(f"{colorama.Fore.CYAN}Optimising CV")
    cv_params_range = np.logspace(start=-4, stop=-2, num=100)
    best_value = -np.inf 
    best_pars = None 
    best_params = None 
    for pars in tqdm(cv_params_range):
        test_params = cv_tracker_params.copy()
        test_params.state_dynamic_variance = pars
        model = classic_models.setup_cv(ground_truth_position, test_params)
        value = classic_models.get_model_performance(model, cluttered_data, ground_truth_position, burn_in, test_params)
        if value > best_value:
            best_value = value 
            best_pars = pars
            best_params = test_params  
    
    if cv_params_range.size > 1 and best_pars in [min(cv_params_range), max(cv_params_range)]:
        print(f"{colorama.Fore.LIGHTYELLOW_EX}Warning: the optimal parameter is on the boundary of the search region.")

    print(f"Best value: {colorama.Fore.GREEN}{best_value} {colorama.Fore.WHITE}at {colorama.Fore.LIGHTYELLOW_EX}{best_pars}")
    return best_params


def optimise_wna_tracker(cluttered_data, ground_truth_position, burn_in, wna_tracker_params : settings.ClassicalStructuralTrackerParameters):
    print(f"{colorama.Fore.CYAN}Optimising WNA")
    wna_params_range = np.logspace(start=-2, stop=2, num=100)
    best_value = -np.inf 
    best_pars = None 
    best_params = None 
    for pars in tqdm(wna_params_range):
        test_params = wna_tracker_params.copy()
        test_params.state_dynamic_variance = pars
        model = classic_models.setup_wna(ground_truth_position, test_params)
        value = classic_models.get_model_performance(model, cluttered_data, ground_truth_position, burn_in, test_params)
        if value > best_value:
            best_value = value 
            best_pars = pars
            best_params = test_params  
    
    if wna_params_range.size > 1 and best_pars in [min(wna_params_range), max(wna_params_range)]:
        print(f"{colorama.Fore.LIGHTYELLOW_EX}Warning: the optimal parameter is on the boundary of the search region.")
    print(f"Best value: {colorama.Fore.GREEN}{best_value} {colorama.Fore.WHITE}at {colorama.Fore.LIGHTYELLOW_EX}{best_pars}")
    return best_params


def optimise_singer_tracker(cluttered_data, ground_truth_position, burn_in, singer_tracker_params : settings.ClassicalStructuralTrackerParameters):
    print(f"{colorama.Fore.CYAN}Optimising Singer")
    singer_params_range = []

    scale_range = np.logspace(start=-1, stop=2, num=10)
    alpha_range = np.logspace(start=-2, stop=1, num=10)

    for scale in scale_range:
        for alpha in alpha_range:
            singer_params_range.append([scale, alpha])
    best_value = -np.inf 
    best_pars = None 
    best_params = None 
    for pars in tqdm(singer_params_range):
        test_params = singer_tracker_params.copy()
        test_params.state_dynamic_variance = pars[0]
        test_params.singer_rate = pars[1]
        model = classic_models.setup_singer(ground_truth_position, test_params)
        value = classic_models.get_model_performance(model, cluttered_data, ground_truth_position, burn_in, test_params)
        if value > best_value:
            best_value = value 
            best_pars = pars
            best_params = test_params  
    
    if (scale_range.size > 1 and best_pars[0] in [min(scale_range), max(scale_range)]) or (alpha_range.size > 1 and best_pars[1] in [min(alpha_range), max(alpha_range)]):
        print(f"{colorama.Fore.LIGHTYELLOW_EX}Warning: the optimal parameter is on the boundary of the search region.")
    print(f"Best value: {colorama.Fore.GREEN}{best_value} {colorama.Fore.WHITE}at {colorama.Fore.LIGHTYELLOW_EX}{best_pars}")
    return best_params 