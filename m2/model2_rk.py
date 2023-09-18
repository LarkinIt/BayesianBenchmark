if __name__ == "__main__":
    import os
    import time
    import copy
    import math
    import pickle
    import numpy as np
    import pandas as pd
    import pocomc as pc
    from scipy.stats import qmc
    import matplotlib.pyplot as plt
    from multiprocessing import Pool

    n_particles = 1000
    SEED=1
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
        
        # Source: https://henryiii.github.io/compclass/week10/2_rk.html
        def rk45_ivp(self, f, init_y, t_range, model_params, tol=1e-8, attempt_steps=20):
            order = len(init_y)  # Number of equations
            t_eval = self.ts 
            y = [np.array(init_y)]
            t = [t_range[0]]
            err_sum = 0

            # Step size and limits to step size
            h = (t_range[1] - t_range[0]) / attempt_steps
            hmin = h / 64
            hmax = h * 64
            #print("h0: ", h)
            for i in range(len(t_eval) - 1):
                while t[-1] < t_eval[i+1]:
                    #print("\n\tt:", t)
                    # Last step should just be exactly what is needed to finish
                    if t[-1] + h > t_range[1]:
                        h = t_range[1] - t[-1]
                    else:
                        h = np.float64(min(h, t_eval[i + 1] - t[-1]))
                    #print("\th: ", h)
                    #print(f(t[-1], y[-1], model_params))
                    # Compute k1 - k6 for evaluation and error estimate
                    k1 = h * f(t[-1], y[-1], model_params)
                    k2 = h * f(t[-1] + h / 4, y[-1] + k1 / 4, model_params)
                    k3 = h * f(t[-1] + 3 * h / 8, y[-1] + 3 * k1 / 32 + 9 * k2 / 32, model_params)
                    k4 = h * f(
                        t[-1] + 12 * h / 13,
                        y[-1] + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197,
                        model_params
                    )
                    k5 = h * f(
                        t[-1] + h,
                        y[-1] + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104,
                        model_params
                    )
                    k6 = h * f(
                        t[-1] + h / 2,
                        y[-1]
                        + 8 * k1 / 27
                        + 2 * k2
                        - 3544 * k3 / 2565
                        + 1859 * k4 / 4104
                        - 11 * k5 / 40,
                        model_params
                    )

                    # Compute error from higher order RK calculation
                    err = np.abs(
                        k1 / 360 - 128 * k3 / 4275 - 2197 * k4 / 75240 + k5 / 50 + 2 * k6 / 55
                    )

                    # Compute factor to see if step size should be changed
                    s = 0 if err[0] == 0 or err[1] == 0 else 0.84 * (tol * h / err[0]) ** 0.25

                    lower_step = s < 0.75 and h > 2 * hmin
                    raise_step = s > 1.5 and 2 * h < hmax
                    no_change = not raise_step and not lower_step

                    # Accept step and move on
                    if err[0] < tol or err[1] < tol or no_change:
                        temp =  y[-1] + 25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5
                        new_y = [max(x, 0) for x in temp]
                        y.append(new_y)
                        t.append(t[-1] + h)

                    # Grow or shrink the step size if needed
                    if lower_step:
                        h /= 2
                    elif raise_step:
                        h *= 2

            return np.array(t), np.array(y)
        
        
        def run_sim(self, model_param = None, x0 = None): #takes in canidate parameters then solves the ode
            if model_param is None:
                model_param= self.theta_true[:self.ODE_params_n]  #sets the model_params to just the model parameters
            if x0 is None:
                x0 = self.x0 #if x0 not defined, default x0 to the model x0
            t_span = (self.ts[0], self.ts[-1]) #define the time span
            result = self.rk45_ivp(self.sys_fun, x0, t_span, model_param)    
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

            result = self.run_sim(model_param=model_param, x0= x0) #sets y to the y results of solving ODE
            y = result[1]
            data = self.data #sets data
            #if self.x_n > 1:
            #    y = np.transpose(y)
            #    print(y.shape)
            t_idxs = np.where(np.in1d(result[0], self.ts))[0]
            y = y[t_idxs, :]
            # TO DO: If we fit a 1D system, need to address HERE
            
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
    mod2_opts['x0'] = [2, 0.25] # initial conditions (if given)
    mod2_opts['fit_sigma'] = True #fit sigma?
    mod2_opts['observable_index'] = [0, 1]
    mod2_opts['theta_true'] = [8, 1, 1, 1, 2, 0.25, 0.3, 0.3] # true param values
    mod2_opts['lower_bnds'] = [2, 0, 0, 0, -3, -3, 1E-3, 1E-3] #lower bounds
    mod2_opts['upper_bnds'] = [20, 5, 5, 5, 3, 3, 1, 1] #upper bounds

    # load data for model 2
    mod2_df = pd.read_csv('m2_data.csv', header=0, delimiter=",") #reads in data file
    mod2_opts['ts'] = mod2_df['t'].values #sets data under 't' in csv as 'ts'
    mod2_opts['data'] = mod2_df[['x1', 'x2']].values #makes data array and reshapes it

    def mod2_sys(t, y, theta): #makes a new function
        temp = np.empty(shape=(2), dtype=np.float64)
        x1, x2 = y 
        k1, k2, k3, k4 = theta #, x1, x2, sigma1, sigma2
        dx1dt = (2 * k1 * x2) - (k2 * (x1**2)) - (k3 * x1 * x2) - (k4 * x1)
        dx2dt = (k2 * (x1**2)) - (k1 * x2)
        return np.array([dx1dt, dx2dt])

    mod2_opts['sys_fun'] = mod2_sys
    mod2 = Model(mod2_opts)

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
    print(elapsed)