import os
import gc
import gzip
import pickle
import argparse
import numpy as np
from memory_profiler import profile
from modelproblem import ModelProblem
from pocosampler import pocoSampler
from pestosampler import pestoSampler

os.environ["OMP_NUM_THREADS"] = "1"

#@profile
def run_model_calibration(args):
    print(args)
    seed = args.seed
    method = args.method
    problem = args.problem
    n_ensemble = args.n_ensemble
    n_cpus = args.n_cpus
    output_dir = args.output_dir

    # Create directory if it does not already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set RNG seed
    np.random.seed(seed=seed)

    mod_prob = ModelProblem(problem)
    mod_prob.initialize()
    
    if method == "ptmcmc":
        sampler = pestoSampler(
            seed,
            n_ensemble,
            mod_prob,
            n_cpus,
            method,
            args.n_iter,
            args.n_chains
            )
    else:
        sampler = pocoSampler(
            seed,
            n_ensemble,
            mod_prob,
            n_cpus,
            method
            )
    sampler.initialize()
    results = sampler.run()

    results_fname = f"{output_dir}/{problem}_{method}_{seed}seed_down.pkl"

    gc.disable()
    try:
        gc.collect()
        with gzip.open(results_fname, "w") as fp:
            pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        gc.enable()
    print(f"Results saved to {results_fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=1, help="Seed number used for NumPy Random Number Generator.")
    parser.add_argument("-m", "--method", type=str, choices={"pmc", "smc", "ptmcmc"}, help="Bayesian inference method to use to perform parameter estimation. Choices include ``ptmcmc`` for Parallel Tempering MCMC, ``smc`` for Sequential Monte Carlo, and ``pmc`` for Preconditioned Monte Carlo.")
    parser.add_argument("-p", "--problem", type=str, help="Parameter estimation problem and corresponding directory (e.g. Michaelis-Menten)")
    # ! WARNING: PTMCMC IS NOT PARALLELIZABLE
    parser.add_argument("-c", "--n_cpus", type=int, default=1, help="Number of CPUs to use. Note: this is only used when method = smc or pmc")
    parser.add_argument("-n", "--n_ensemble", type=int, default=100, help="The number of samples used to make the final posterior ensemble. This is also used as the number of particles used in PMC and SMC")
    parser.add_argument("-i", "--n_iter", type=int, help="Number of MCMC iterations. ONLY used for PT-MCMC")
    parser.add_argument("-w", "--n_chains", type=int, default=4, help="Number of chains used for PT-MCMC")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Directory to save result file to. Defaults to the current directory.")
    args = parser.parse_args()
    
    run_model_calibration(args)