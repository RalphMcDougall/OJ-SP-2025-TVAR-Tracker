import math
import time
import colorama
import numpy as np
from datetime import datetime


class Seeder():
    def __init__(self, seed):
        self.seed = seed 
    
    def set_current(self):
        assert self.seed is not None, "Seed cannot be None"
        #print(f"Setting seed to {colorama.Fore.LIGHTYELLOW_EX}{self.seed}")
        np.random.seed(self.seed)


class ARGenerationParameters:
    def __init__(self):
        self.process_order = None

        self.innovation_variance = None 
        self.time_varying = None 
        self.damping_gamma_shape = None 
        self.damping_gamma_scale = None 
        self.min_damping = None 
        self.max_damping = None 
        self.walk_scale = None 
        self.phase_uniform_max_pi_factor = None

        self.oscillating = None
    
    def copy(self):
        copied_params = ARGenerationParameters()
        copied_params.process_order = self.process_order

        copied_params.innovation_variance = self.innovation_variance
        copied_params.time_varying = self.time_varying
        copied_params.damping_gamma_shape = self.damping_gamma_shape
        copied_params.damping_gamma_scale = self.damping_gamma_scale
        copied_params.min_damping = self.min_damping
        copied_params.max_damping = self.max_damping
        copied_params.walk_scale = self.walk_scale
        copied_params.phase_uniform_max_pi_factor = self.phase_uniform_max_pi_factor

        copied_params.oscillating = self.oscillating

        return copied_params


class MeasurementParameters:
    def __init__(self):
        self.noise_variance = None 
        self.clutter_rate = None 
        self.observation_probability = None

    
    def copy(self):
        new_params = MeasurementParameters()

        new_params.noise_variance = self.noise_variance 
        new_params.clutter_rate = self.clutter_rate
        new_params.observation_probability = self.observation_probability

        return new_params
    

    def set_global_clutter_volume(self, v):
        self._global_clutter_volume = v


    def get_global_clutter_volume(self):
        return self._global_clutter_volume


class ProcessInstanceParameters:
    def __init__(self):
        self.innovation_variance = None
        self.innovation_to_noise_ratio = None 
    
    def copy(self):
        new_params = ProcessInstanceParameters()

        new_params.innovation_variance = self.innovation_variance
        new_params.innovation_to_noise_ratio = self.innovation_to_noise_ratio
        
        return new_params

    @property 
    def noise_variance(self):
        return self.innovation_variance / self.innovation_to_noise_ratio
    
    @property
    def noise_to_innovation_ratio(self):
        return 1 / self.innovation_to_noise_ratio


class ParameterKFSetupParameters:
    def __init__(self):
        self.prior_parameter_variance = None 
        self.initial_innovation_variance = None 
        self.initial_noise_variance = None 
        self.state_transition_variance = None 

        self.state_innovation_to_process_innovation = None
    
    @property 
    def process_innovation_to_state_innovation(self):
        return 1 / self.state_innovation_to_process_innovation


class PredictionParameters:
    def __init__(self):
        self.look_ahead = None
        self.learning_time = None 



class EMParameters:
    def __init__(self):
        self.num_em_iterations = None 
        self.noise_walk_variance = None 
        self.innovation_walk_variance = None 


class PFParameters:
    def __init__(self):

        self.num_particles = None 
        self.min_particle_factor = None 

        self.a_oracle_prior = None 
        self.z_oracle_prior = None  
        self.sigma_oracle_prior = None 
    
    @property
    def min_particles(self):
        return math.ceil(self.num_particles * self.min_particle_factor)


class TrackerParameters(Seeder): # TODO: Can probably refactor this out
    def __init__(self, seed = None):
        super().__init__(seed)

        self.process_instance_params = ProcessInstanceParameters()
        self.parameter_kf_setup_params = ParameterKFSetupParameters()
        self.measurement_params = MeasurementParameters()


class TVARTrackerParameters(Seeder):
    def __init__(self, seed = None):
        super().__init__(seed)
        
        self.simulation_time = None 
        self.observation_dimensions = None 

        self.model_order = None

        self.constant_poles = []

        self.process_instance_params = ProcessInstanceParameters()
        self.measurement_params = MeasurementParameters()
        
        self.num_particles = None 
        self.min_particle_factor = None 

        self.a_oracle_prior = None 
        self.z_oracle_prior = None  
        self.sigma_oracle_prior = None 

        self.state_innovation_to_process_innovation = None
    

    def copy(self):
        new_params = TVARTrackerParameters(self.seed)

        new_params.simulation_time = self.simulation_time
        new_params.observation_dimensions = self.observation_dimensions

        new_params.model_order = self.model_order

        new_params.process_instance_params = self.process_instance_params.copy()
        new_params.measurement_params = self.measurement_params.copy()
        
        new_params.num_particles = self.num_particles
        new_params.min_particle_factor = self.min_particle_factor

        new_params.a_oracle_prior = self.a_oracle_prior
        new_params.z_oracle_prior = self.z_oracle_prior
        new_params.sigma_oracle_prior = self.sigma_oracle_prior

        new_params.state_innovation_to_process_innovation = self.state_innovation_to_process_innovation

        return new_params 
    
    @property
    def min_particles(self):
        return np.ceil(self.min_particle_factor * self.num_particles)


class ClassicalStructuralTrackerParameters:
    def __init__(self):
        self.simulation_time = None 
        self.observation_dimensions = None

        self.measurement_params = MeasurementParameters()

        self.z_oracle_prior = None  

        self.state_dynamic_variance = None 
        self.singer_rate = None
    

    def copy(self):
        new_params = ClassicalStructuralTrackerParameters()

        new_params.simulation_time = self.simulation_time
        new_params.observation_dimensions = self.observation_dimensions

        new_params.measurement_params = self.measurement_params.copy()

        new_params.z_oracle_prior = self.z_oracle_prior

        new_params.state_dynamic_variance = self.state_dynamic_variance
        new_params.singer_rate = self.singer_rate
        
        return new_params


class IMMTrackerParameters:
    def __init__(self):
        self.simulation_time = None 
        self.observation_dimensions = None

        self.measurement_params = MeasurementParameters()

        self.z_oracle_prior = None  

        self.model_params : list[ClassicalStructuralTrackerParameters] | None = None 
    

    def copy(self):
        new_params = IMMTrackerParameters()

        new_params.simulation_time = self.simulation_time
        new_params.observation_dimensions = self.observation_dimensions

        new_params.measurement_params = self.measurement_params.copy()

        new_params.z_oracle_prior = self.z_oracle_prior

        new_params.model_params = [m_params.copy() for m_params in self.model_params]
        
        return new_params


class Jin2017TrackerParameters:
    def __init__(self):
        self.simulation_time = None 
        self.observation_dimensions = None

        self.measurement_params = MeasurementParameters()

        self.z_oracle_prior = None  

        self.model_order = None 
        self.polynomial_order = None 
        self.window = None 
        self.innovation_variance = None

    def copy(self):
        new_params = Jin2017TrackerParameters()

        new_params.simulation_time = self.simulation_time
        new_params.observation_dimensions = self.observation_dimensions

        new_params.measurement_params = self.measurement_params.copy()

        new_params.z_oracle_prior = self.z_oracle_prior

        new_params.model_order = self.model_order
        new_params.polynomial_order = self.polynomial_order
        new_params.window = self.window 
        new_params.innovation_variance = self.innovation_variance

        return new_params


class CovarianceMethodARParameters:
    def __init__(self):
        self.simulation_time = None 
        self.observation_dimensions = None

        self.measurement_params = MeasurementParameters()

        self.z_oracle_prior = None  

        self.state_dynamic_variance = None 
        self.model_order = None

    def copy(self):
        new_params = CovarianceMethodARParameters()

        new_params.simulation_time = self.simulation_time
        new_params.observation_dimensions = self.observation_dimensions

        new_params.measurement_params = self.measurement_params.copy()

        new_params.z_oracle_prior = self.z_oracle_prior

        new_params.state_dynamic_variance = self.state_dynamic_variance
        new_params.model_order = self.model_order

        return new_params


class DataGenerationParameters(Seeder):
    def __init__(self, seed = None):
        super().__init__(seed)

        self.simulation_time = None 
        self.observation_dimensions = None

        self.ar_generation_params = ARGenerationParameters()
        self.measurement_params = MeasurementParameters()
    
    def copy(self):
        copied_params = DataGenerationParameters(self.seed)

        copied_params.simulation_time = self.simulation_time
        copied_params.observation_dimensions = self.observation_dimensions

        copied_params.ar_generation_params = self.ar_generation_params.copy()
        copied_params.measurement_params = self.measurement_params.copy()

        return copied_params


def inject_data_generation_params(params : TVARTrackerParameters | ClassicalStructuralTrackerParameters | Jin2017TrackerParameters | CovarianceMethodARParameters, data_generation_params : DataGenerationParameters):
    new_params = params.copy()

    new_params.simulation_time = data_generation_params.simulation_time
    new_params.observation_dimensions = data_generation_params.observation_dimensions

    new_params.measurement_params = data_generation_params.measurement_params.copy()

    return new_params 


def inject_imm_data_generation_params(params : IMMTrackerParameters, data_generation_params : DataGenerationParameters):
    new_params = params.copy()

    new_params.simulation_time = data_generation_params.simulation_time
    new_params.observation_dimensions = data_generation_params.observation_dimensions

    new_params.measurement_params = data_generation_params.measurement_params.copy()

    for i in range(len(params.model_params)):
        new_params.model_params[i] = inject_data_generation_params(new_params.model_params[i], data_generation_params)

    return new_params 