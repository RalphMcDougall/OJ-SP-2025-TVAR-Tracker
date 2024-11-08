import numpy as np
import jsonpickle
import colorama

import data_generation_utils
import settings 

colorama.init(autoreset=True)


class DatasetManager:
    def __init__(self):
        self.generation_params : settings.DataGenerationParameters = None 
        self.ground_truth : np.ndarray = None
        self.measurement_set : data_generation_utils.MeasurementSet = None 
        self.process_coefficients : np.ndarray = None


class TestsetManager:
    def __init__(self):
        self.training_sets : list[DatasetManager] = []
        self.datasets : list[DatasetManager] = []
    
    def add_training_set(self, training_set : DatasetManager):
        self.training_sets.append(training_set)

    def add_dataset(self, dataset : DatasetManager):
        self.datasets.append(dataset)


class FileManager:
    @staticmethod
    def construct_full_file_name(file_name):
        return f"{file_name}.json"

    @staticmethod
    def write(obj : any, file_name : str):
        full_file_name = FileManager.construct_full_file_name(file_name)
        with open(full_file_name, "w") as f:
            f.write(jsonpickle.encode(obj))
        print(f"{colorama.Fore.GREEN}Successfully wrote to: {colorama.Fore.LIGHTYELLOW_EX}{full_file_name}")

    
    @staticmethod
    def read(file_name : str):
        obj = None 
        full_file_name = FileManager.construct_full_file_name(file_name)
        with open(full_file_name, "r") as f:
            obj = jsonpickle.decode(f.read())

        print(f"{colorama.Fore.GREEN}Successfully read from: {colorama.Fore.LIGHTYELLOW_EX}{full_file_name}")
        return obj 