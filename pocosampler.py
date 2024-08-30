import pocomc as pc 
from bayesianinference import BayesianInference
from modelproblem import ModelProblem

class pocoSampler(BayesianInference):
    def __init__(self, seed: int, n_ensemble: int, model_problem: ModelProblem, n_cpus: int, method):
        super().__init__(seed, n_ensemble, model_problem, n_cpus)
        self.method = method
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

        sampler = pc.Sampler(
            prior=prior,
            likelihood=llh,
            n_effective=self.n_ensemble,
            n_active=self.n_ensemble,
            precondition=self.precondition,
            pool=pool_no
        )
        self.sampler = sampler
        
    def run(self):
        self.sampler.run(progress=True)