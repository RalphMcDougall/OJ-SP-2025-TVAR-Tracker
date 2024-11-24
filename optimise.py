import colorama
import numpy as np
from tqdm import tqdm

import settings 
import pf_utils 
import comparison_models


def optimise_tvar_tracker(cluttered_data : list[np.ndarray], ground_truth_position : np.ndarray, true_process_coefficients : np.ndarray, tvar_tracker_params : settings.TVARTrackerParameters):
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


def optimise_cv_tracker(cluttered_data : np.ndarray, ground_truth_position : np.ndarray, burn_in : int, cv_tracker_params : settings.ClassicalStructuralTrackerParameters):
    print(f"{colorama.Fore.CYAN}Optimising CV")
    cv_params_range = np.logspace(start=-4, stop=-2, num=100)
    best_value = -np.inf 
    best_pars = None 
    best_params = None 
    for pars in tqdm(cv_params_range):
        test_params = cv_tracker_params.copy()
        test_params.state_dynamic_variance = pars
        model = comparison_models.setup_cv(ground_truth_position, test_params)
        value = comparison_models.get_model_performance(model, cluttered_data, ground_truth_position, burn_in, test_params)
        if value > best_value:
            best_value = value 
            best_pars = pars
            best_params = test_params  
    
    if cv_params_range.size > 1 and best_pars in [min(cv_params_range), max(cv_params_range)]:
        print(f"{colorama.Fore.LIGHTYELLOW_EX}Warning: the optimal parameter is on the boundary of the search region.")

    print(f"Best value: {colorama.Fore.GREEN}{best_value} {colorama.Fore.WHITE}at {colorama.Fore.LIGHTYELLOW_EX}{best_pars}")
    return best_params


def optimise_wna_tracker(cluttered_data : np.ndarray, ground_truth_position : np.ndarray, burn_in : int, wna_tracker_params : settings.ClassicalStructuralTrackerParameters):
    print(f"{colorama.Fore.CYAN}Optimising WNA")
    wna_params_range = np.logspace(start=-2, stop=2, num=100)
    best_value = -np.inf 
    best_pars = None 
    best_params = None 
    for pars in tqdm(wna_params_range):
        test_params = wna_tracker_params.copy()
        test_params.state_dynamic_variance = pars
        model = comparison_models.setup_wna(ground_truth_position, test_params)
        value = comparison_models.get_model_performance(model, cluttered_data, ground_truth_position, burn_in, test_params)
        if value > best_value:
            best_value = value 
            best_pars = pars
            best_params = test_params  
    
    if wna_params_range.size > 1 and best_pars in [min(wna_params_range), max(wna_params_range)]:
        print(f"{colorama.Fore.LIGHTYELLOW_EX}Warning: the optimal parameter is on the boundary of the search region.")
    print(f"Best value: {colorama.Fore.GREEN}{best_value} {colorama.Fore.WHITE}at {colorama.Fore.LIGHTYELLOW_EX}{best_pars}")
    return best_params


def optimise_wcm_ar_tracker(cluttered_data : np.ndarray, ground_truth_position : np.ndarray, burn_in : int, wcm_ar_tracker_params : settings.CovarianceMethodARParameters):
    print(f"{colorama.Fore.CYAN}Optimising WCM-AR")
    wcm_ar_params_range = np.logspace(start=-2, stop=2, num=100)
    best_value = -np.inf 
    best_pars = None 
    best_params = None 
    for pars in tqdm(wcm_ar_params_range):
        test_params = wcm_ar_tracker_params.copy()
        test_params.state_dynamic_variance = pars
        model = comparison_models.setup_windowed_covariance_method_ar(ground_truth_position, test_params)
        value = comparison_models.get_model_performance(model, cluttered_data, ground_truth_position, burn_in, test_params)
        if value > best_value:
            best_value = value 
            best_pars = pars
            best_params = test_params  
    
    if wcm_ar_params_range.size > 1 and best_pars in [min(wcm_ar_params_range), max(wcm_ar_params_range)]:
        print(f"{colorama.Fore.LIGHTYELLOW_EX}Warning: the optimal parameter is on the boundary of the search region.")
    print(f"Best value: {colorama.Fore.GREEN}{best_value} {colorama.Fore.WHITE}at {colorama.Fore.LIGHTYELLOW_EX}{best_pars}")
    return best_params


def optimise_singer_tracker(cluttered_data : np.ndarray, ground_truth_position : np.ndarray, burn_in : int, singer_tracker_params : settings.ClassicalStructuralTrackerParameters):
    print(f"{colorama.Fore.CYAN}Optimising Singer")
    singer_params_range = []

    scale_range = np.logspace(start=-1, stop=1, num=10)
    alpha_range = np.logspace(start=-2, stop=np.log10(-np.log(1 / 20)), num=10)

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
        model = comparison_models.setup_singer(ground_truth_position, test_params)
        value = comparison_models.get_model_performance(model, cluttered_data, ground_truth_position, burn_in, test_params)
        if value > best_value:
            best_value = value 
            best_pars = pars
            best_params = test_params  
    
    if (scale_range.size > 1 and best_pars[0] in [min(scale_range), max(scale_range)]) or (alpha_range.size > 1 and best_pars[1] in [min(alpha_range), max(alpha_range)]):
        print(f"{colorama.Fore.LIGHTYELLOW_EX}Warning: the optimal parameter is on the boundary of the search region.")
    print(f"Best value: {colorama.Fore.GREEN}{best_value} {colorama.Fore.WHITE}at {colorama.Fore.LIGHTYELLOW_EX}{best_pars}")
    return best_params 


def optimise_double_singer_imm(cluttered_data : np.ndarray, ground_truth_position : np.ndarray, burn_in : int, imm_params : settings.IMMTrackerParameters):
    print(f"{colorama.Fore.CYAN}Optimising double-Singer IMM")
    
    imm_params_range = []

    scale_range_1 = np.logspace(start=-1, stop=1, num=8)
    alpha_range_1 = np.logspace(start=-1, stop=np.log10(-np.log(1 / 20)), num=8)
    scale_range_2 = np.logspace(start=-1, stop=1, num=8)
    alpha_range_2 = np.logspace(start=-1, stop=np.log10(-np.log(1 / 20)), num=8)

    for scale_1 in scale_range_1:
        for alpha_1 in alpha_range_1:
            for scale_2 in scale_range_2:
                if scale_2 < scale_1:
                    continue 
                for alpha_2 in alpha_range_2:
                    imm_params_range.append([scale_1, alpha_1, scale_2, alpha_2])
    
    best_value = -np.inf 
    best_params = None 
    best_pars = None 

    for pars in tqdm(imm_params_range):
        test_params = imm_params.copy()

        test_params.model_params[0].state_dynamic_variance = pars[0]
        test_params.model_params[0].singer_rate = pars[1]
        test_params.model_params[1].state_dynamic_variance = pars[2]
        test_params.model_params[1].singer_rate = pars[3]

        model = comparison_models.setup_double_singer_imm(ground_truth_position, test_params)
        value = comparison_models.get_imm_model_performance(model, cluttered_data, ground_truth_position, burn_in, test_params)

        if value > best_value:
            best_value = value
            best_pars = pars
            best_params = test_params.copy()

    print(f"Best value: {colorama.Fore.GREEN}{best_value} {colorama.Fore.WHITE}at {colorama.Fore.LIGHTYELLOW_EX}{best_pars}")
    return best_params


def optimise_jin2017_tracker(cluttered_data : np.ndarray, ground_truth_position : np.ndarray, burn_in : int, jin2017_tracker_params : settings.Jin2017TrackerParameters, data_generation_params : settings.DataGenerationParameters):
    print(f"{colorama.Fore.CYAN}Optimising JinAR2017")
    jin2017_params_range = []

    for polynomial_order in range(1, 4):
        for model_order in range(polynomial_order + 2, 6):
            for window_size in range(10, 60, 10):
                for innovation_variance in np.logspace(start=-1, stop=1, num=10):
                    jin2017_params_range.append([model_order, polynomial_order, window_size, innovation_variance])
    
    best_value = -np.inf 
    best_params = None 
    best_pars = None 

    for pars in tqdm(jin2017_params_range):
        test_params = jin2017_tracker_params.copy()
        test_params.model_order = pars[0]
        test_params.polynomial_order = pars[1]
        test_params.window = pars[2]
        test_params.innovation_variance = pars[3]

        value = comparison_models.get_jin2017_model_performance(cluttered_data, ground_truth_position, data_generation_params, test_params, burn_in)

        if value > best_value:
            best_value = value
            best_pars = pars
            best_params = test_params.copy()

    print(f"Best value: {colorama.Fore.GREEN}{best_value} {colorama.Fore.WHITE}at {colorama.Fore.LIGHTYELLOW_EX}{best_pars}")
    return best_params