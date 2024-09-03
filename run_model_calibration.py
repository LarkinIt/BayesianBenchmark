import os
import pickle
import argparse
import numpy as np
from modelproblem import ModelProblem
from pocosampler import pocoSampler
from pestosampler import pestoSampler

os.environ["OMP_NUM_THREADS"] = "1"

def run_model_calibration(args):
    print(args)
    seed = args.seed
    method = args.method
    problem = args.problem
    n_ensemble = args.n_ensemble
    n_cpus = args.n_cpus
    output_dir = args.output_dir
    np.random.seed(seed=seed)

    mod_prob = ModelProblem(problem)
    mod_prob.initialize()
    
    if method == "ptmcmc":
        sampler = pestoSampler(seed, n_ensemble, mod_prob, n_cpus, method, args.n_iter)
    else:
        sampler = pocoSampler(seed, n_ensemble, mod_prob, n_cpus, method)
    sampler.initialize()
    results = sampler.run()

    results_fname = f"{output_dir}/{problem}_{method}_{seed}seed.pkl"
    with open(results_fname, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved to {results_fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument("-m", "--method", type=str, choices={"pmc", "smc", "ptmcmc"})
    parser.add_argument("-p", "--problem", type=str)
    # ! WARNING: PTMCMC IS NOT PARALLELIZABLE
    parser.add_argument("-c", "--n_cpus", type=int, default=1)
    parser.add_argument("-n", "--n_ensemble", type=int, default=100)
    parser.add_argument("-i", "--n_iter", type=int)
    parser.add_argument("-o", "--output_dir", type=str, default=".")
    args = parser.parse_args()
    
    run_model_calibration(args)