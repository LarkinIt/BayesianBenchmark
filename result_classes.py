import itertools
import numpy as np
from scipy.stats import distributions

class Result:
	def __init__(self, result_dict) -> None:
		for key in result_dict:
			setattr(self, key, result_dict[key])
		if self.method == "ptmcmc":
			burn_in_idx = self.algo_specific_info["burn_in_idx"]
			n_chains = self.n_chains
			#print(f"{burn_in_idx}, \t{n_chains}")
			self.n_fun_calls = (burn_in_idx+1)*n_chains

	def get_sampling_ratio(self, par_bounds, par_idx=0, inlog=False) -> float:
		"""
		Measures the ratio of the sampling space 
		explored for a given parameter index
		"""
		bound_diff = par_bounds[par_idx][1] - par_bounds[par_idx][0]
		par_samples = self.all_samples[:, :, par_idx]
		if inlog:
			par_samples = 10**par_samples
		max_val = np.max(par_samples)
		min_val = np.min(par_samples)
		sample_diff = max_val - min_val
		return sample_diff/bound_diff
	
	def get_convergence(self, llh_threshold):
		idxs = np.where(self.all_llhs > llh_threshold)
		first_iter = np.min(idxs[0])

		if self.method == "ptmcmc":
			#print(self.all_llhs.shape, self.all_llhs[5:,0], idxs)
			# ! TO DO: replace this with a more exact calculation
			#print(first_iter)
			conv_calls = first_iter * self.n_chains
		else:
			conv_calls = self.algo_specific_info["calls_by_iter"][first_iter]
		return conv_calls


class MethodResults:
	def __init__(self, method) -> None:
		self.all_runs = []
		self.method = method
		if method == "pmc":
			self.abbr = "PMC"
			self.label = "Preconditioned Monte Carlo"
		elif method == "smc":
			self.abbr = "SMC"
			self.label = "Sequential Monte Carlo"
		elif method == "ptmcmc":
			self.abbr = "PT-MCMC"
			self.label = "Parallel Tempering MCMC"
	
	def add_result(self, result_obj):
		self.all_runs.append(result_obj)

	def get_fun_calls(self) -> np.array:
		all_calls = [x.n_fun_calls for x in self.all_runs]
		return np.array(all_calls)
	
	def get_llhs(self) -> np.array:
		all_llhs = [x.posterior_llhs for x in self.all_runs]
		return np.array(all_llhs)
	
	def get_sampling_efficiency(self, bounds, par_idx, inlog=False) -> np.array:
		all_ratios = [x.get_sampling_ratio(bounds, par_idx, inlog) for x in self.all_runs]
		return np.array(all_ratios)
	
	def get_convergence_times(self, llh_threshold):
		all_convs = [x.get_convergence(llh_threshold) for x in self.all_runs]
		return np.array(all_convs)

	
	# Source: https://stackoverflow.com/questions/40044375/how-to-calculate-the-kolmogorov-smirnov-statistic-between-two-weighted-samples
	def ks_weighted(self, data1, data2, wei1, wei2, alternative='two-sided'):
		ix1 = np.argsort(data1)
		ix2 = np.argsort(data2)
		data1 = data1[ix1]
		data2 = data2[ix2]
		wei1 = wei1[ix1]
		wei2 = wei2[ix2]
		data = np.concatenate([data1, data2])
		cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
		cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
		cdf1we = cwei1[np.searchsorted(data1, data, side='right')]
		cdf2we = cwei2[np.searchsorted(data2, data, side='right')]
		d = np.max(np.abs(cdf1we - cdf2we))
		# calculate p-value
		n1 = data1.shape[0]
		n2 = data2.shape[0]
		m, n = sorted([float(n1), float(n2)], reverse=True)
		en = m * n / (m + n)
		if alternative == 'two-sided':
			prob = distributions.kstwo.sf(d, np.round(en))
		else:
			z = np.sqrt(en) * d
			# Use Hodges' suggested approximation Eqn 5.3
			# Requires m to be the larger of (n1, n2)
			expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
			prob = np.exp(expt)
		return d, prob
	
	def calc_pairwise_matrix(self, par_index):
		n_runs = len(self.all_runs)
		combos = itertools.combinations(range(n_runs), 2)
		ks_matrix = np.zeros(shape=(n_runs, n_runs))
		pval_matrix = np.zeros(shape=(n_runs, n_runs))
		for i, j in combos:
			runA = self.all_runs[i]
			runB = self.all_runs[j]
			
			param_samplesA = runA.posterior_samples[:, par_index]
			param_samplesB = runB.posterior_samples[:, par_index]
			ks_stat, pval = self.ks_weighted(param_samplesA, param_samplesB,
											runA.posterior_weights, runB.posterior_weights)
			ks_matrix[j, i] = ks_stat
			ks_matrix[i, j] = ks_stat
			pval_matrix[j, i] = pval
			pval_matrix[i, j] = pval
		return ks_matrix, pval_matrix