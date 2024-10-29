import os
import numpy as np
from scipy.stats import qmc
from bayesianinference import BayesianInference
from modelproblem import ModelProblem
import pypesto.sample as sample
from pypesto.sample.geweke_test import burn_in_by_sequential_geweke

class pestoSampler(BayesianInference):
	def __init__(
			self, 
			seed: int, 
			n_ensemble: int, 
			model_problem: ModelProblem,
			n_cpus: int,
			method: str,
			n_iter: int,
			n_chains: int
			):
		super().__init__(seed, n_ensemble, model_problem, n_cpus, method)
		self.n_iter = n_iter
		self.n_chains = n_chains

	def initialize(self):
		mod_prob = self.model_problem

		lhs = qmc.LatinHypercube(d=mod_prob.n_dim, seed=self.seed)
		scale_x0 = lhs.random(n=self.n_chains)
		lbs = [x[0] for x in mod_prob.bounds]
		ubs = [x[1] for x in mod_prob.bounds]
		x0 = qmc.scale(scale_x0, l_bounds=lbs, u_bounds=ubs)

		self.x0 = list(x0)
		sampler = sample.AdaptiveParallelTemperingSampler(
			internal_sampler=sample.AdaptiveMetropolisSampler(),
			n_chains=self.n_chains
			)
	
		sampler.initialize(mod_prob.problem, list(x0))
			
		for internal_sampler in sampler.samplers:
			internal_sampler.neglogpost = self.model_problem.log_likelihood_wrapper
		
		self.sampler = sampler

	def create_posterior_ensemble(self):
		sampler = self.sampler
		samples = sampler.get_samples()

		# get the lowest temperature chain index
		# Note: remember that beta is the inverse of the temperature so 
		# we want the chain with the max beta
		ch_idx = np.argmax(samples.betas)
		chain = np.array(samples.trace_x[ch_idx, :, :])
		burn_in_idx = burn_in_by_sequential_geweke(chain)

		#trim_trace_x = samples.trace_x[ch_idx, burn_in_idx:, :]
		#trim_trace_llhs = -1*samples.trace_neglogpost[ch_idx, burn_in_idx:]
		#trim_trace_priors = samples.trace_neglogprior[ch_idx, burn_in_idx:]
		#choice_idxs = np.random.choice(range(0, trim_trace_llhs.shape[0]), size=self.n_ensemble, replace=False)
		#posterior_samples = trim_trace_x[choice_idxs, :]
		#posterior_llhs = trim_trace_llhs[choice_idxs]
		#posterior_priors = trim_trace_priors[choice_idxs]
		return burn_in_idx #, posterior_samples, posterior_llhs, posterior_priors


	def process_results(self):
		sampler = self.sampler
		algo_specific_info = {}
		algo_specific_info["betas"] = sampler.betas
		bi_idx = self.create_posterior_ensemble()
		algo_specific_info["burn_in_idx"] = bi_idx

		all_results = {}
		all_results["seed"] = self.seed
		all_results["n_ensemble"] = self.n_ensemble
		all_results["method"] = self.method
		all_results["problem"] = self.model_problem.model_name

		all_results["n_iter"] = self.n_iter+1
		all_results["iters"] = np.array(range(self.n_iter+1))
		all_results["n_chains"] = self.n_chains
		
		all_samples = np.array([x.trace_x for x in sampler.samplers])
		all_samples = np.swapaxes(all_samples, 0, 1)
		all_weights = np.ones(shape=all_samples.shape[:-1])
		all_llhs = -1*np.array([x.trace_neglogpost for x in sampler.samplers])
		all_llhs = np.swapaxes(all_llhs, 0, 1)
		all_priors = np.array([x.trace_neglogprior for x in sampler.samplers])
		all_priors = np.swapaxes(all_priors, 0, 1)
		
		all_results["all_samples"] = all_samples
		all_results["all_weights"] = all_weights
		all_results["all_llhs"] = all_llhs
		all_results["all_priors"] = all_priors

		#all_results["posterior_samples"] = post_samples
		#all_results["posterior_weights"] = np.array([1/len(post_llhs) for x in post_llhs])
		#all_results["posterior_llhs"] = post_llhs
		#all_results["posterior_priors"] = post_pris

		n_fun_calls = self.model_problem.n_fun_calls
		all_results["n_fun_calls"] = n_fun_calls
		all_results["algo_specific_info"] = algo_specific_info
		return all_results
			
	def run(self):
		sampler = self.sampler
		sampler.sample(n_samples=self.n_iter)
		results = self.process_results()
		return results