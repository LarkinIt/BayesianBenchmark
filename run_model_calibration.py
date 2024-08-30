import os
import petab
import argparse
import numpy as np
import pocomc as pc
import pypesto.sample as sample
from modelproblem import ModelProblem
from pocosampler import pocoSampler
import pypesto.objective.roadrunner as pypesto_rr

os.environ["OMP_NUM_THREADS"] = "1"

def run_model_calibration(args):
    print(args)
    seed = args.seed
    method = args.method
    problem = args.problem
    n_ensemble = args.n_ensemble
    n_cpus = args.n_cpus
    
    mod_prob = ModelProblem(problem)
    mod_prob.initialize()
    print(mod_prob.log_likelihood_wrapper([0,0,0]))
    
    if method == "ptmcmc":
        pass
    else:
        sampler = pocoSampler(seed, n_ensemble, mod_prob, n_cpus, method)
    sampler.initialize()
    sampler.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument("-m", "--method", type=str, choices={"pmc", "smc", "ptmcmc"})
    parser.add_argument("-p", "--problem", type=str)
    parser.add_argument("-c", "--n_cpus", type=int, default=1)
    parser.add_argument("-n", "--n_ensemble", type=int, default=100)
    parser.add_argument("-o", "--output_dir", type=str, default=".")
    args = parser.parse_args()
    
    run_model_calibration(args)