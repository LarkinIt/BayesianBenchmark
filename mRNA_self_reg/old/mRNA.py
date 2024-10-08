# this is edited for the mRNA model based on Caroline's mm.py file 

import sys 
sys.path.append("../pocomc/")

import os
import time
import pickle
import roadrunner
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import qmc
from multiprocessing import Pool
import pocomc as pc
from model import Model

if __name__ == "__main__":    
    SEED = 1
    np.random.seed(seed=SEED) # setting seed for reproducibility

    n_cpus = os.cpu_count() # numb of cores on comp
    print('This machine has {} CPUs'.format(n_cpus))

    # mRNA Model Options
    mod_opts = {} # creates a dictionary
    # !!! double check this value !!!
    mod_opts['theta_n'] = 5 # total number of values in big array?
    mod_opts['ODE_params_n'] = 5 # numb of ODE params
    mod_opts['x_n'] = 3 # numb of species

    # print(mod_opts)
    # {'theta_n': 5, 'ODE_params_n': 5, 'x_n': 3}

    # ! TO DO: we currently assume it fit_x0, then ALL species initial conditions are fit
    mod_opts['fit_x0'] = False # fit initial conditions? false meaning no
    ## if fit_x0 = True then initial conditions would be included in theta args? 
    mod_opts['x0'] = [10,5,0] # initial conditions in order (if given)
    # !!! check this
    # observabale_idxs is indices of librr_species, so length must be <_ x_n
    mod_opts['observable_idxs'] = [2] # indices of outputs containing the observables
    mod_opts['fit_sigma'] = False # whether to estimate std of exp data
    mod_opts['theta_true'] = [10.0, 0.5, 10.0, 2.0, 1.0] #guess param values(in log)
    mod_opts['lower_bnds'] = [9.5, 0.0, 9.5, 1.5, 0.5] #lower bounds(in log)
    mod_opts['upper_bnds'] = [10.5, 1.0, 10.5, 2.5, 1.5] #upper bounds(in log)
    # mod_opts['lower_bnds'] = [8.0, 0.0, 8.0, 0.0, -1.0] #lower bounds(in log)
    # mod_opts['upper_bnds'] = [12.0, 2.5, 12.0, 4.0, 3.0] #upper bounds(in log)

    # load data for model
    mod_df = pd.read_csv('synthetic_mRNA_data.csv', header=0, delimiter=",") #reads in data file
    # !!! CHECK THIS 
    mod_opts['ts'] = mod_df['Time'].values #sets data under 't' in csv as 'ts'
    # !!! CHECK THIS
    data = mod_df['mRNA'].values
    mod_opts['data'] = data
    # Load in SBML model using libroadrunner
    sbml_file = "mRNA_sbml.xml"
    librr_model = roadrunner.RoadRunner(sbml_file)
    librr_theta = ["param0", "param1", "param2", "param3", "param4"]
    librr_species = ["S1", "S2", "S3"]
    librr_labels = ["mRNA", "P1", "P2"]

    mod_opts['librr_model'] = librr_model
    mod_opts['librr_theta'] = librr_theta
    mod_opts["librr_species"] = librr_species

    mod = Model(mod_opts)

    n_particles = 1000

    # Initialise particles' positions using samples from the prior 
    sampler = qmc.LatinHypercube(d=mod.theta_n, seed=SEED)
    sample = sampler.random(n=n_particles) 
    # print("sample:", sample.shape)
    # print(sample)
    print("The discrepancy of the sampling (i.e., sample quality): %.4f"%qmc.discrepancy(sample)) #discrepancy is distance between cont. uniform distr. on hypercube & discr. uniform distr. on n distinct sample points
    prior_samples = qmc.scale(sample, l_bounds=mod.lower_bnds, u_bounds=mod.upper_bnds)
    # print(prior_samples.shape)

    # in parallel
    t0 = time.time() #stores current time, date, year, etc. in one float
    with Pool(n_cpus) as pool: #sets up code to run over my n number of cpus on laptop
        
        sampler = pc.Sampler(n_particles = n_particles,
                        n_dim = mod.theta_n,
                        log_likelihood = mod.log_likelihood,
                        log_prior = mod.log_prior,
                        bounds = np.array(list(zip(mod.lower_bnds, mod.upper_bnds))),
                        pool = pool,
                        random_state=SEED,
                        vectorize_likelihood=False,
                        vectorize_prior=False,
                        infer_vectorization=False
                        ) #stores all relevant info from # of parameters being fit (ndim) to the actual results

        sampler.run(prior_samples = prior_samples) #starts with prior sample definied in latin hypercube sampling, and runs it
        result = sampler.results #results of sampler.run on prior_samples

    # in serial     
    # sampler = pc.Sampler(n_particles = n_particles,
    #                 n_dim = mod.theta_n,
    #                 log_likelihood = mod.log_likelihood,
    #                 log_prior = mod.log_prior,
    #                 bounds = np.array(list(zip(mod.lower_bnds, mod.upper_bnds))),
    #                 random_state=SEED,
    #                 vectorize_likelihood=False,
    #                 vectorize_prior=False,
    #                 infer_vectorization=False
    #                 ) #stores all relevant info from # of parameters being fit (ndim) to the actual results

    # sampler.run(prior_samples = prior_samples) #starts with prior sample definied in latin hypercube sampling, and runs it
    # result = sampler.results #results of sampler.run on prior_samples
    
    with open('tester_result.pkl', 'wb') as f: #open that file if exists, if not make that file
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL) #saves result object (dictionary) to pickle file

    # we need to remove the librr simulator from the class in order to successfully save and load the class from a pkl file. 
    del mod.librr_model
    with open('tester_mod.pkl', 'wb') as f: #open that file if exists, if not make that file
        pickle.dump(mod.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL) #saves model dictionary in pickle file

    t1 = time.time() #time after running this section
    seconds = t1-t0 #difference in start and stop time

    elapsed = time.strftime("%H:%M:%S", time.gmtime(seconds)) #converts float to a time quantity we use
    print('\nElapsed time: ', elapsed) #printing time it took for code to run