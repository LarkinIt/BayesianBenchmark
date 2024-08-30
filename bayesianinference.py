from abc import ABC, abstractmethod
from modelproblem import ModelProblem

class BayesianInference(ABC):
    def __init__(self, seed: int, n_ensemble: int, model_problem: ModelProblem, n_cpus: int):
        self.seed = seed
        self.n_ensemble = n_ensemble
        self.model_problem = model_problem
        self.n_cpus = n_cpus
    
    @abstractmethod 
    def initialize(self):
        pass 
    
    @abstractmethod
    def run(self):
        pass