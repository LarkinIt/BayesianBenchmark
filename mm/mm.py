if __name__ == "__main__":
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

    import pypesto
    import pocomc as pc
    import dynesty as dy
    import pypesto.engine as eng
    import pypesto.sample as sample
    import pypesto.store as store
    import pypesto.optimize as optimize
    from pypesto.ensemble import Ensemble

    SEED = 1
    np.random.seed(seed=SEED)

    n_cpus = os.cpu_count() 
    print('This machine has {} CPUs'.format(n_cpus))

    class Model:
        def __init__(self, opts): #initial settings
            for key in opts: #loops for all labels in the list 'key'
                setattr(self, key, opts[key]) #creates a dictionary where 'key' are the list of labels & 'ops[key]' are the values

        def __call__(self, theta_new):
            theta_new = theta_new
            res = self.log_likelihood(theta_new)
            return res
        
        def change_and_run(self, model_param, x0):
            rr = self.librr_model
            rr.resetAll()
            rr.integrator.absolute_tolerance = 5e-10
            rr.integrator.relative_tolerance = 1e-8

            for spec_name, val in zip(self.librr_species, x0):
                init_species_string = f"init([{spec_name}])"
                rr[init_species_string] = float(val)
                rr.reset()
            
            for name, value in zip(self.librr_theta, model_param):
                rr[name] = float(value)
                rr.reset() # forces initial conditions and BNGL functions to be re-evaluated
            
            t_span = (self.ts[0], self.ts[-1])
            trajs = rr.simulate(t_span[0], t_span[1], int(t_span[1]*100+1))
            return trajs    
            
        def call_sim(self, model_param = None, x0 = None, return_all_species=False): #takes in candidate parameters then solves the ode
            if model_param is None:
                model_param= self.theta_true[:self.ODE_params_n]  
            if x0 is None:
                x0 = self.x0 
            
            trajs = self.change_and_run(model_param, x0)
            if return_all_species:
                return trajs
            
            sim_ts = trajs[:, 0]
            species = trajs[:, 1:]
            # ! TO DO: Assumes simulation includes data ts which is not always the case
            # ! Need to add interpolation for times not included in simulation
            t_idxs = np.where(np.in1d(sim_ts, self.ts))[0]
            return_species = species[t_idxs, self.observable_idxs]
            return return_species
        
        def log_prior(self, theta_new): 
            bools = [(low <= i <= high) for i,low,high in zip(theta_new, self.lower_bnds, self.upper_bnds)] #if generated values are within bounds
            all_in_range = np.all(bools) #if all values are true, then output is true
            if all_in_range: 
                return 0.0 
            return -np.inf #if even one parameter out of bounds, it's false, and returns -infinity

        def log_likelihood(self, theta_new): #how good is this canidate parameter fitting my data (maximize it)
            model_param = theta_new[:self.ODE_params_n] 
            if self.fit_x0: 
                x0 = theta_new[self.ODE_params_n:(self.ODE_params_n + self.x_n)] #sets x0 to 'theta_true' x0 values
            else:
                x0 = self.x0

            if self.fit_sigma:
                sigma = theta_new[-len(self.observable_idxs):] #observable index related to sigma
            else:
                sigma = [1] * len(self.observable_idxs) #makes all sigmas default to 1
            y = self.call_sim(model_param=model_param, x0=x0) #sets y to the y results of solving ODE
            data = self.data #sets data

            # Calculate likelihood
            term1 = -0.5 * np.log(2*np.pi*np.square(sigma))
            term2 = np.square(np.subtract(y, data)) / (2*np.square(sigma))
            logLH = np.sum(term1 - term2)
            return np.array(logLH)

    mod_opts = {} #creates a dictionary
    mod_opts['theta_n'] = 3 #total number of values in big array
    mod_opts['ODE_params_n'] = 3 # num ODE params
    mod_opts['x_n'] = 4 # num species
    mod_opts['sigma_n'] = 0 # num sigmas
    mod_opts['theta_labels'] = ['$k_1$', '$k_2$', '$k_3$'] # parameter labels

    # ! TO DO: we currently assume it fit_x0, then ALL species initial conditions are fit
    mod_opts['fit_x0'] = False # fit initial conditions?
    mod_opts['x0'] = [600,6,0,0] # initial conditions (if given)
    mod_opts['observable_idxs'] = [3] # indices of outputs containing the observables
    mod_opts['fit_sigma'] = False #fit sigma?
    mod_opts['theta_true'] = [-2.77, -1.0, -2.0] #guess param values(in log)
    mod_opts['lower_bnds'] = [-3.0,-1.0,-3.0] #lower bounds(in log)
    mod_opts['upper_bnds'] = [1.0,3.0,3.0] #upper bounds(in log)

    # load data for model
    mod_df = pd.read_csv('mm_data.csv', header=0, delimiter=",") #reads in data file
    mod_opts['ts'] = mod_df['t'].values #sets data under 't' in csv as 'ts'
    data = mod_df['P'].values
    mod_opts['data'] = data
    # Load in SBML model using libroadrunner
    sbml_file = "mm_sbml.xml"
    librr_model = roadrunner.RoadRunner(sbml_file)
    librr_theta = ["log_k1", "log_k2", "log_k3"]
    librr_species = ["S1", "S2", "S3", "S4"]
    librr_labels = ["Substrate", "Enzyme", "Substrate-Enzyme", "Product"]

    mod_opts['librr_model'] = librr_model
    mod_opts['librr_theta'] = librr_theta
    mod_opts["librr_species"] = librr_species

    mod = Model(mod_opts)

    true_trajs = mod.call_sim(return_all_species=True)
    bad_trajs = mod.call_sim(x0=[200,6,0,300], model_param=[0.0,2.0,-2.0], return_all_species=True)

    test_theta = [-3,3,3] # This should return 0.0 
    print(mod.log_prior(test_theta))
    test_bad_theta1 = [-4, 1, 1] # should return -np.inf
    print(mod.log_prior(test_bad_theta1))
    test_bad_theta2 = [1, 4, 3] # should return -np.inf
    print(mod.log_prior(test_bad_theta2))
    test_bad_theta3 = [0, 0, 3.1] # should return -np.inf
    print(mod.log_prior(test_bad_theta3))
    print("--------------------")
    # Sanity checks with log_likelihood
    print(mod.log_likelihood(mod.theta_true))
    print(mod.log_likelihood([-20,2,-5])) 

    n_particles = 1000

    # Initialise particles' positions using samples from the prior 
    sampler = qmc.LatinHypercube(d=mod.theta_n, seed=SEED)
    sample = sampler.random(n=n_particles) 
    print("The discrepancy of the sampling (i.e., sample quality): %.4f"%qmc.discrepancy(sample)) #discrepancy is distance between cont. uniform distr. on hypercube & discr. uniform distr. on n distinct sample points
    prior_samples = qmc.scale(sample, l_bounds=mod.lower_bnds, u_bounds=mod.upper_bnds)
    print(prior_samples.shape)

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

    with open('tester_result.pkl', 'wb') as f: #open that file if exists, if not make that file
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL) #saves result object (dictionary) to pickle file

    with open('tester_mod.pkl', 'wb') as f: #open that file if exists, if not make that file
        pickle.dump(mod.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL) #saves model dictionary in pickle file

    t1 = time.time() #time after running this section
    seconds = t1-t0 #difference in start and stop time

    elapsed = time.strftime("%H:%M:%S", time.gmtime(seconds)) #converts float to a time quantity we use
    print('\nElapsed time: ', elapsed) #printing time it took for code to run
