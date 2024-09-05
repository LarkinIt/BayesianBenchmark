import petab
from pprint import pprint
from functools import update_wrapper
from petab.v1.parameters import get_priors_from_df
from scipy.stats import uniform, norm
import pypesto.objective.roadrunner as pypesto_rr
from pypesto.objective import AggregatedObjective

# Taken from https://stackoverflow.com/questions/63644282/how-to-replace-every-function-call-with-the-wrapper-call-in-the-original-class
class ObjectiveWrapper(AggregatedObjective):
	def __new__(cls, AggregatedObjective):
		instance = super().__new__(cls) 
		instance.module = AggregatedObjective

		for funcname in dir(AggregatedObjective):
			print(funcname)
			if funcname != "__call__":
				continue
			else:
				func = getattr(AggregatedObjective, funcname)
				wrapped = ObjectiveWrapper._wrap(func)
				setattr(instance, funcname, wrapped)
				continue
		return instance


	@staticmethod
	def _wrap(func):
		"""
		Wraps *func* with additional code.
		"""
		# we define a wrapper function. This will execute all additional code
		# before and after the "real" function.
		def wrapped(*args, **kwargs):
			print("before-call:", func, args, kwargs)
			output = func(*args, **kwargs)
			print("after-call:", func, args, kwargs, output)
			return output
		# Use "update_wrapper" to keep docstrings and other function metadata
		# intact
		update_wrapper(wrapped, func)
		# We can now return the wrapped function
		return wrapped

class ModelProblem():
	def __init__(self, model_string):
		self.model_name = model_string

	def initialize(self):
		model_name =  self.model_name
	   
		petab_yaml = f"./{model_name}/{model_name}.yaml"
		petab_problem = petab.v1.Problem.from_yaml(petab_yaml)
		importer = pypesto_rr.PetabImporterRR(petab_problem)
		problem = importer.create_problem()

		# set tolerances for ode solver
		solver_options = pypesto_rr.SolverOptions(
			relative_tolerance = 5e-8,
			absolute_tolerance = 1e-6
			)
		"""print(type(problem.objective).__name__)
		objective = problem.objective
		wrapped_objective = ObjectiveWrapper(objective)
		problem.objective - wrapped_objective
		problem.objective.solver_options = solver_options
		exit()
		"""
		self.problem = problem
		self.petab_problem = petab_problem
		prior_info = get_priors_from_df(petab_problem.parameter_df,
								  mode="objective")
		self.prior_info = prior_info
		self.n_dim = len(prior_info)
		self.bounds = [x[3] for x in prior_info]
		self.n_fun_calls = 0

	def create_poco_priors(self):
		prior_list = []

		# the list returned from get_priors_from_df is always
		# in the following order:
		# 1) prior type (string)
		# 2) prior parameters (tuple)
		# 3) parameter scale (string - log or linear)
		# 4) parameter bounds (tuple)
		# ignore #3 and #4 
		for info in self.prior_info:
			type, prior_pars, _, _ = info
			if "uniform" in type.lower():
				lb, ub = prior_pars
				prior = uniform(loc=lb, scale=ub-lb)
			elif "normal" in type.lower():
				mean, std = prior_pars
				prior = norm(loc=mean, scale=std)
			prior_list.append(prior)
		return prior_list


	def log_likelihood_wrapper(self, x):
		try:
			result = self.problem.objective(x, mode="mode_fun", return_dict=True)
			fval = -1*result["fval"]
		except:
			fval = -1e10
		# ! IMPORTANT: self.n_fun_calls only tracks total number of function calls
		# ! when using PT-MCMC since it does NOT run in parallel
		# ! You can only use this with pocoMC when n_cpus = 1
		self.n_fun_calls += 1
		return fval