import os
import numpy as np
from scipy.stats import qmc
from bayesianinference import BayesianInference
from modelproblem import ModelProblem
import pypesto.engine as eng
import pypesto.sample as sample
import pypesto.history as history

class pestoSampler(BayesianInference):
	def __init__(
			self, 
			seed: int, 
			n_ensemble: int, 
			model_problem: ModelProblem,
			n_cpus: int,
			n_iter: int
			):
		super().__init__(seed, n_ensemble, model_problem, n_cpus)
		self.n_iter = n_iter


	def initialize(self):
		mod_prob = self.model_problem

		lhs = qmc.LatinHypercube(d=mod_prob.n_dim, seed=self.seed)
		scale_x0 = lhs.random(n=self.n_ensemble)
		lbs = [x[0] for x in mod_prob.bounds]
		ubs = [x[1] for x in mod_prob.bounds]
		x0 = qmc.scale(scale_x0, l_bounds=lbs, u_bounds=ubs)

		sampler = sample.AdaptiveParallelTemperingSampler(
			internal_sampler=sample.AdaptiveMetropolisSampler(),
			n_chains=self.n_ensemble
			)
		sampler.initialize(mod_prob.problem, list(x0))

		# change the history of each internal 
		# sampler to save objective function calls
		for internal_sampler in sampler.samplers:
			history_options = history.HistoryOptions(trace_record=True)
			hist_opts = history.HistoryOptions.assert_instance(history_options)
			internal_sampler.neglogpost.__setattr__(
				"history", 
				mod_prob.problem.objective.create_history(
					"ch1",
					x_names=mod_prob.problem.objective.x_names,
					options=hist_opts
					)
				)
		
		self.sampler = sampler

	def process_results(self):
		sampler = self.sampler
		poco_results = sampler.results 

		algo_specific_info = {}
		algo_specific_info["precondition"] = self.precondition
		algo_specific_info["efficiency"] = poco_results["efficiency"]
		algo_specific_info["n_mcmc"] = poco_results["steps"]
		algo_specific_info["effective_ss"] = poco_results["ess"]
		algo_specific_info["betas"] = poco_results["beta"]
		algo_specific_info["u"] = poco_results["u"]
		
		all_results = {}
		all_results["seed"] = self.seed
		all_results["n_ensemble"] = self.n_ensemble
		all_results["method"] = self.method
		all_results["problem"] = self.model_problem.model_name
		all_results["acceptance"] = poco_results["accept"]
		all_results["posterior_samples"] = poco_results["x"][-1, :, :]
		all_results["posterior_weights"] = np.exp(poco_results["logw"])
		all_results["posterior_llh"] = poco_results["logl"][-1, :]
		all_results["all_samples"] = poco_results["x"]
		all_results["all_llh"] = poco_results["logl"]
		all_results["n_fun_calls"] = poco_results["calls"]
		all_results["algo_specific_info"] = algo_specific_info
		return all_results
			
	def run(self):
		#mod_prob = self.model_problem
		sampler = self.sampler
		
		"""
		if self.n_cpus > 1:
			engine = eng.SingleCoreEngine()
		else:
			n_procs = self.n_cpus
			if self.n_cpus == 0:
				n_procs = os.cpu_count()
			engine = eng.MultiProcessEngine(n_procs=n_procs)
		"""
		sampler.sample(n_samples=self.n_iter)

		results = self.process_results()
		return results