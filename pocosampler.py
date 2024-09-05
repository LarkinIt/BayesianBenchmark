import os
import numpy as np
import pocomc as pc 
from bayesianinference import BayesianInference
from modelproblem import ModelProblem

class pocoSampler(BayesianInference):
    def __init__(
        self,
        seed: int,
        n_ensemble: int,
        model_problem: ModelProblem,
        n_cpus: int,
        method: str
        ):
        super().__init__(seed, n_ensemble, model_problem, n_cpus, method)
        self.precondition = False
        if method == "pmc":
            self.precondition = True

    def initialize(self):
        mod_prob = self.model_problem
        prior_list = mod_prob.create_poco_priors()
        prior = pc.Prior(prior_list)
        llh = mod_prob.log_likelihood_wrapper
        
        pool_no = None
        if self.n_cpus > 1:
            pool_no = self.n_cpus
        elif self.n_cpus == 0:
            pool_no = os.cpu_count()
        sampler = pc.Sampler(
            prior=prior,
            likelihood=llh,
            n_effective=self.n_ensemble,
            n_active=self.n_ensemble,
            precondition=self.precondition,
            random_state=self.seed,
            pool=pool_no,
            sample="rwm"
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
        all_results["n_fun_calls"] = sampler.calls
        all_results["algo_specific_info"] = algo_specific_info

        #print("POCO OUTPUT: ", poco_results["calls"])
        #print("POCO 2 OUTPUT: ", sampler.calls)
        #print("LOG LIKELIHOOD OUTPUT: ", self.model_problem.n_fun_calls)
        return all_results
            
    def run(self):
        self.sampler.run(progress=True)
        results = self.process_results()
        return results