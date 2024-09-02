import petab
from pprint import pprint
import pocomc as pc
import pypesto.objective
from petab.v1.parameters import get_priors_from_df
from scipy.stats import uniform, norm
#from abc import ABC, abstractmethod
import pypesto.objective.roadrunner as pypesto_rr


class ModelProblem():
	def __init__(self, model_string):
		self.model_name = model_string


	def initialize(self):
		model_name =  self.model_name
	   
		petab_yaml = f"./{model_name}/{model_name}.yaml"
		petab_problem = petab.v1.Problem.from_yaml(petab_yaml)
		importer = pypesto_rr.PetabImporterRR(petab_problem)
		problem = importer.create_problem()
		
		self.problem = problem
		self.petab_problem = petab_problem
		self.n_dim = len(get_priors_from_df(petab_problem.parameter_df,
								  mode="objective"))


	def create_poco_priors(self):
		df = self.petab_problem.parameter_df
		prior_info = get_priors_from_df(df, mode="objective")
		prior_list = []

		# the list returned from get_priors_from_df is always
		# in the following order:
		# 1) prior type (string)
		# 2) prior parameters (tuple)
		# 3) parameter scale (string - log or linear)
		# 4) parameter bounds (tuple)
		# ignore #3 and #4 
		for info in prior_info:
			type, prior_pars, _, _ = info
			if "uniform" in type.lower():
				lb, ub = prior_pars
				prior = uniform(loc=lb, scale=ub-lb)
			elif "normal" in type.lower():
				mean, std = prior_pars
				prior = norm(log=mean, scale=std)
			prior_list.append(prior)
		return prior_list


	def log_likelihood_wrapper(self, x):
		result = self.problem.objective(x, mode="mode_fun", return_dict=True)
		return -1*result["fval"]