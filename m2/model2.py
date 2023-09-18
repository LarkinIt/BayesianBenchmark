
if __name__ == "__main__":
    # Importing libraries
    import time
    import numpy as np
    import pandas as pd
    from scipy.integrate import solve_ivp
    from scipy.stats import qmc
    import matplotlib.pyplot as plt
    from multiprocessing import Pool
    import pocomc as pc
    import pickle
    import os

    SEED = 0
    np.random.seed(seed=SEED)

    n_cpus = os.cpu_count() 

    # Defining Model
    class Model:
        def __init__(self, opts): #initial settings
            for key in opts: #loops for all labels in the list 'key'
                setattr(self, key, opts[key]) #creates a dictionary where 'key' are the list of labels & 'ops[key]' are the values

        def __call__(self, theta_new):
            theta_new = theta_new
            res = self.log_likelihood(theta_new)
            return res
        
        def run_sim(self, model_param = None, x0=None): #takes in canidate parameters then solves the ode
            if model_param is None:
                model_param= self.theta_true[:self.ODE_params_n]  #sets the model_params to just the model parameters
            if x0 is None:
                x0 = self.x0 #if x0 not defined, default x0 to the model x0
            t_span = (self.ts[0], self.ts[-1]) #define the time span
            x_n = self.x_n 
            
            result = solve_ivp(self.sys_fun, t_span, y0=x0, 
                            t_eval=self.ts, args=([model_param])) #solve ODE (with model parameters)
            
            return result #returns result.t and result.y
        
        def log_prior(self, theta_new): 
            bools = [(low <= i <= high) for i,low,high in zip(theta_new, self.lower_bnds, self.upper_bnds)] #if generated values are within bounds
            all_in_range = np.all(bools) #if all values are true, then output is true
            if all_in_range: #if true
                return 0.0 #give 0
            return -np.inf #if even one parameter out of bounds, it's false, and returns -infinity

        def log_likelihood(self, theta_new): #how good is this canidate parameter fitting my data (maximize it)
            model_param = theta_new[:self.ODE_params_n] 
            if self.fit_x0: 
                x0 = theta_new[self.ODE_params_n:(self.ODE_params_n + self.x_n)] #sets x0 to 'theta_true' x0 values
            else:
                x0 = self.x0

            if self.fit_sigma:
                sigma = theta_new[-len(self.observable_index):] #observable index related to sigma
            else:
                sigma = [1] * len(self.observable_index) #makes all sigmas default to 1

            y = self.run_sim(model_param=model_param, x0= x0).y #sets y to the y results of solving ODE
            data = self.data #sets data
            if self.x_n > 1:
                y = np.transpose(y)

            # Calculate posterior; how good is parameter in terms of fitting the data
            term1 = -0.5 * np.log(2*np.pi*np.square(sigma))
            term2 = np.square(np.subtract(y, data)) / (2*np.square(sigma))
            logLH = np.sum(term1 - term2)
            return logLH


    # Creating Model 2 Objects
    # Model 2 Options
    mod2_opts = {} #creates a dictionary
    mod2_opts['theta_n'] = 8 #total number of values in big array
    mod2_opts['ODE_params_n'] = 4 # num ODE params
    mod2_opts['x_n'] = 2 # num species
    mod2_opts['sigma_n'] = 2 # num sigmas
    mod2_opts['theta_names'] = ['$k_1$', '$k_2$', '$k_3$', '$k_4$', '$x_1$', '$x_2$','$\sigma_1$','$\sigma_2$'] #names of the parameters (for dictionary)
    mod2_opts['fit_x0'] = True # fit initial conditions?
    mod2_opts['x0'] = [0] # initial conditions (if given)
    mod2_opts['fit_sigma'] = True #fit sigma?
    mod2_opts['observable_index'] = [0]
    mod2_opts['theta_true'] = [8, 1, 1, 1, 2, 0.25, 0.3, 0.3] # true param values
    mod2_opts['lower_bnds'] = [2, 0, 0, 0, -3, -3, 1E-3, 1E-3] #lower bounds
    mod2_opts['upper_bnds'] = [20, 5, 5, 5, 3, 3, 1, 1] #upper bounds

    # load data for model 2
    mod2_df = pd.read_csv('m2_data.csv', header=0, delimiter=",") #reads in data file
    mod2_opts['ts'] = mod2_df['t'].values #sets data under 't' in csv as 'ts'
    mod2_opts['data'] = mod2_df[['x1', 'x2']].values #makes data array and reshapes it

    def mod2_sys(t, y, theta): #makes a new function
        x1, x2 = y 
        k1, k2, k3, k4 = theta #, x1, x2, sigma1, sigma2
        dx1dt = (2 * k1 * x2) - (k2 * (x1**2)) - (k3 * x1 * x2) - (k4 * x1)
        dx2dt = (k2 * (x1**2)) - (k1 * x2)
        return [dx1dt, dx2dt] #the ode given in paper for model 2

    mod2_opts['sys_fun'] = mod2_sys
    mod2 = Model(mod2_opts) #calls mod2 the Model class with the mod2_opts parameters!

    result = mod2.run_sim(mod2_opts['theta_true'][0:4], mod2_opts['theta_true'][4:6] ) #run the simulation on model 2 parameters

    print(mod2.log_likelihood([8, 1, 1, 1, 2, 0.25, 0.3, 0.3]))
    print(mod2.log_likelihood([20, 5, 5, 5, 3, 3, 1, 1]))
    
    fig = plt.figure(figsize=(14,5), dpi=300)
    fig.add_subplot(1, 2, 1)
    plt.plot(result.t, result.y[0, :], 'r-')
    plt.plot(mod2_opts['ts'], mod2_opts['data'][:, 0], 'ro')
    
    fig.add_subplot(1, 2, 2)
    plt.plot(result.t, result.y[1, :], 'b-')
    plt.plot(mod2_opts['ts'], mod2_opts['data'][:, 1], 'bo')    
    
    plt.savefig('mod2.png')
    exit()

    # Latin Hypercube Sampling
    n_particles = 1000 #amount of points in the latin hypercube sampling?

    # Initialise particles' positions using samples from the prior (this is very important, other initialisation will not work).
    sampler = qmc.LatinHypercube(d=mod2.theta_n, seed=SEED) #d is amount of parameters being solved for in model 2
    sample = sampler.random(n=n_particles) #causes sample to be an array of n_particles x mod2.theta_n (1000x3)  
    prior_samples = qmc.scale(sample, l_bounds=mod2.lower_bnds, u_bounds=mod2.upper_bnds) #widens/narrows sample to be within the bounds given

    # Pocomc
    t0 = time.time() #stores current time, date, year, etc. in one float
    with Pool(n_cpus) as pool: #sets up code to run over my n number of cpus on laptop
        
        sampler = pc.Sampler(n_particles = n_particles,
                        n_dim = mod2.theta_n,
                        log_likelihood = mod2,
                        log_prior = mod2.log_prior,
                        bounds = np.array(list(zip(mod2.lower_bnds, mod2.upper_bnds))),
                        pool = pool,
                        random_state=SEED
                        ) #stores all relevant info from # of parameters being fit (ndim) to the actual results

        sampler.run(prior_samples = prior_samples) #starts with prior sample definied in latin hypercube sampling, and runs it
        result = sampler.results #results of sampler.run on prior_samples


    with open('mod2_result.pkl', 'wb') as f: #open that file if exists, if not make that file
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL) #saves result object (dictionary) to pickle file

    with open('mod2_obj.pkl', 'wb') as f: #open that file if exists, if not make that file
        pickle.dump(mod2.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL) #saves model 2 dictionary in pickle file

    t1 = time.time() #time after running this section
    seconds = t1-t0 #difference in start and stop time

    elapsed = time.strftime("%H:%M:%S", time.gmtime(seconds)) #converts float to a time quantity we use
    print('\nElapsed time: ', elapsed) #printing time it took for code to run