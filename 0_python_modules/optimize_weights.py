import numpy as np
from scipy import optimize
import os
import copy
import sys

def S_fg_calculate(model_int,experimental_intensities):
    # For comparing Raman/ROA spectra - return a number between -1,1; 1 beeing the best
    S_fg = np.sum(\
            model_int*experimental_intensities)/\
        ((np.sum((model_int)**2)*np.sum((experimental_intensities)**2))**0.5)
    return S_fg

# Calculate gaussian smearing
def calculate_point(x_my,x_array,intensities):
    sigma=3
    x=np.abs(x_array-x_my)
    point_sum=np.sum(intensities*(1/sigma)*np.exp(-(x**2)/sigma/sigma/2))
    return point_sum

def shift_raman_roa_data(params,freq,ram,roa,exp_ram,exp_roa):
    par_a,par_b,par_c,par_d = params

    ftt = 1

    freq = freq[::ftt]
    ram = ram[::ftt]
    roa = roa[::ftt]
    exp_ram = exp_ram[::ftt]
    exp_roa = exp_roa[::ftt]

    #shift
    sf_array = (par_b*1/(1+np.exp((-freq+par_d)/par_c))+par_a*(1-1/(1+np.exp((-freq+par_d)/par_c))))
    freq_new_tmp = freq*sf_array

    minv = np.min(freq)
    maxv = np.max(freq)


    freq_new = np.zeros(len(freq))
    ram_new = np.zeros(len(freq))
    roa_new = np.zeros(len(freq))
    for j,freq_v in enumerate(np.linspace(minv,maxv,len(freq))):
        ram_new[j]=calculate_point(freq_v,freq_new_tmp,ram)
        roa_new[j]=calculate_point(freq_v,freq_new_tmp,roa)
        freq_new[j] = freq_v

    return ram_new,roa_new

def recalculate_shifting_function(sim_data,original_sim_data,exp_data,overall_weights_min,kwargs_dict):
    global raman_data_scaled, roa_data_scaled
    global par_a,par_b,par_c,par_d

    try:
        recalculate_raman_shifting_function = kwargs_dict['recalculate_raman_shifting_function']
    except:
        recalculate_raman_shifting_function = True
    if recalculate_raman_shifting_function == True:
        try:
            set_raman_shifting_function_parameters_range = kwargs_dict['set_raman_shifting_function_parameters_range']
        except:
            print("Problem with setting raman shifting function parameters range. Falling back to default values.")
            set_raman_shifting_function_parameters_range = [(0.95,1.05),(0.95,1.05),(15,1000),(500,2500)]

    print("After optimization - recalculating Raman/ROA shifting function.")
          
    freq = original_sim_data[0][0][::1].copy()
    ram = np.dot(original_sim_data[1].transpose(),overall_weights_min)
    roa = np.dot(original_sim_data[2].transpose(),overall_weights_min)
 
    # optimize shifting function using only Raman spectrum
    def scaling_function_par_optimization(sf_raman_params,freq,ram,roa,exp_ram,exp_roa):
        par_a,par_b,par_c,par_d = sf_raman_params
 
        ftt = 2
 
        freq = freq[::ftt]
        ram = ram[::ftt]
        roa = roa[::ftt]
        exp_ram = exp_ram[::ftt]
        exp_roa = exp_roa[::ftt]
 
        #shift
        sf_array = (par_b*1/(1+np.exp((-freq+par_d)/par_c))+par_a*(1-1/(1+np.exp((-freq+par_d)/par_c))))
        freq_new_tmp = freq*sf_array
 
        minv = np.min(freq)
        maxv = np.max(freq)
 
        freq_new = np.zeros(len(freq))
        ram_new = np.zeros(len(freq))
        roa_new = np.zeros(len(freq))
        for j,freq_v in enumerate(np.linspace(minv,maxv,len(freq))):
            ram_new[j] = calculate_point(freq_v,freq_new_tmp,ram)
            roa_new[j] = calculate_point(freq_v,freq_new_tmp,roa)
            freq_new[j] = freq_v

        c_ram_0,c_roa_0 = [500.0, 71.75]
        c_ram_exp,c_roa_exp = [1,1]
 
        # Calculate differences for Raman/ROA - S_fg
        S_fg_ram = S_fg_calculate(ram_new,exp_ram)
        ram_diff = c_ram_0*c_ram*((1 - S_fg_ram)**c_ram_exp)
 
        S_fg_roa = S_fg_calculate(roa_new,exp_roa)
        roa_diff = c_roa_0*c_roa*((1 - S_fg_roa)**c_roa_exp)
        
        # roa factor - at the beginning, ROA->0 so frequecies could be shifted by a weird factor. 
        # As the optimization progress, we strive to use ROA more and more. 
        diff_roa_f = 1.3*S_fg_roa if S_fg_roa > 0 else 0
        
        diff = ram_diff + diff_roa_f*roa_diff
 
        return diff
 
    opt_result=optimize.differential_evolution(scaling_function_par_optimization,\
    bounds=set_raman_shifting_function_parameters_range,\
    args=(freq,ram,roa,exp_data[1],exp_data[2]),\
    maxiter=100)
 
 
    par_a,par_b,par_c,par_d = opt_result.x
    sf_raman_params = par_a,par_b,par_c,par_d

    for idx,i in enumerate(sim_data[1]):
        raman_data_scaled[idx],roa_data_scaled[idx] = shift_raman_roa_data(sf_raman_params,exp_data[0],sim_data[1][idx],sim_data[2][idx],exp_data[1],exp_data[2])
 
    os.makedirs('tmp_files',exist_ok=True)
    with open('tmp_files/raman_roa_optimized_scaling_function_parameters.dat','a') as fw:
        fw.write("{0} {1} {2} {3} \n".format(par_a,par_b,par_c,par_d))

def par_optimization(params,sim_data,exp_data,experimental_data_av,weights_indices_kept,original_weights,original_sim_data,kwargs_dict):
  
    # Set global variables
    global counter
    global par_a,par_b,par_c,par_d
    global raman_data_scaled, roa_data_scaled
    global error_data_array_i
    global weights_min,error_min,error_min_counter,overall_weights_min
    global stop_optimization_when_n_structures_reached
    global relative_optimization_weights
    global c_ram,c_roa,c_sch,c_scc,c_jij

    # Set overall/individual optimization error to 0
    difference = 0
    ram_diff,roa_diff,sch_diff,scc_diff,jij_diff = [0,0,0,0,0]

    # weights this iteration
    weights_i = params/np.sum(params)

    # Exp data available
    ed_raman,ed_roa,ed_shifts_H,ed_shifts_C,ed_spinspin = experimental_data_av

    # Set Raman/ROA scaling function parameters
    sf_raman_params = [par_a,par_b,par_c,par_d]

    # Set initial relative weights & exponents
    # linear 


    c_ram_0,c_roa_0,c_sch_0,c_scc_0,c_jij_0 = [500.0, 71.75, 177.1, 16.83, 18.76]
    c_ram_exp,c_roa_exp,c_sch_exp,c_scc_exp,c_jij_exp = [1,1,1,1,1]

    try:
        # Load data - Raman - these data are scaled as the optimization goes
        sim_raman_sp = np.dot(raman_data_scaled.transpose(),weights_i)
        # Calculate differences for Raman - S_fg
        S_fg_ram = S_fg_calculate(sim_raman_sp,exp_data[1])
        ram_diff = c_ram_0*c_ram*((1 - S_fg_ram)**c_ram_exp)
        if ed_raman == 1:
            # add it to overall difference
            difference += ram_diff
    except:
        ram_diff = 0

    try:
        # Load data - ROA - these data are scaled as the optimization goes
        sim_roa_sp = np.dot(roa_data_scaled.transpose(),weights_i)
        # Calculate differences for ROA - S_fg
        S_fg_roa = S_fg_calculate(sim_roa_sp,exp_data[2])
        roa_diff = c_roa_0*c_roa*((1 - S_fg_roa)**c_roa_exp)
        if ed_roa == 1:        
            # add it to overall difference
            difference += roa_diff
    except:
        roa_diff = 0

    # H NMR chemical shifts

    if sim_data[3].size !=0:
        sim_sch = np.dot(sim_data[3].transpose(),weights_i)
        sch_diff = c_sch_0*c_sch*np.sum((np.abs(sim_sch-exp_data[3]))**c_sch_exp)/len(sim_sch) 
    else:
        sch_diff = 0
    if ed_shifts_H == 1:            
        difference += sch_diff

    # C NMR chemical shifts
    
    if sim_data[4].size !=0:
        sim_scc = np.dot(sim_data[4].transpose(),weights_i)
        scc_diff = c_scc_0*c_scc*np.sum((np.abs(sim_scc-exp_data[4]))**c_scc_exp)/len(sim_scc) 
    else:
        scc_diff = 0
    if ed_shifts_C == 1:            
        difference += scc_diff

    # Jij spin-spin coupling constants
    if sim_data[5].size !=0:
        sim_jij = np.dot(sim_data[5].transpose(),weights_i)
        jij_diff = c_jij_0*c_jij*np.sum((np.abs(sim_jij-exp_data[5]))**c_jij_exp)/len(sim_jij)
    else:
        jij_diff
    if ed_spinspin == 1:            
        difference += jij_diff

    # Structures bias - so we dont converge to 0
    error_diff_arr = np.array([ram_diff,roa_diff,sch_diff,scc_diff,jij_diff])
    error_sum = np.sum(error_diff_arr)

    # Bias #n of structures 
    try:
        bias_n_structures_reached = kwargs_dict['bias_n_structures_reached']
        bias_n_structures_reached = kwargs_dict['bias_n_structures_reached_f']
    except:
        bias_n_structures_reached = True
        bias_n_structures_reached_f = 3/4
        
    str_bias = 0
    structures_selected = np.sum(weights_i > 0.0001*np.max(weights_i))
    structures_selected_f = structures_selected/len(weights_i)
    if bias_n_structures_reached == True:
        if structures_selected_f > bias_n_structures_reached_f:
            str_bias = 0
        else:
            str_bias = error_sum*(bias_n_structures_reached_f - structures_selected/len(weights_i))

    difference += str_bias
        
    # Raman/ROA - recalculate the shifting function?
    try:
        recalculate_raman_shifting_function = kwargs_dict['recalculate_raman_shifting_function']
    except:
        recalculate_raman_shifting_function = True
    if recalculate_raman_shifting_function == True:
        try:
            set_raman_shifting_function_parameters_range = kwargs_dict['set_raman_shifting_function_parameters_range']
        except:
            print("Problem with setting raman shifting function parameters range. Falling back to default values.")
            set_raman_shifting_function_parameters_range = [(0.95,1.05),(0.95,1.05),(15,1000),(500,2500)]

    if (ed_raman == 1) and (recalculate_raman_shifting_function == True):
        if counter in [1_000,10_000,25_000,50_000,100_000,175_000,250_000,400_000] + [750_000 + 250_000*i for i in range(0,100)]:

            print("step:{0}, recalculating Raman/ROA shifting function.".format(counter))
                  
            freq = original_sim_data[0][0][::1].copy()
            ram = np.dot(original_sim_data[1].transpose(),overall_weights_min)
            roa = np.dot(original_sim_data[2].transpose(),overall_weights_min)

            # optimize shifting function using only Raman spectrum
            def scaling_function_par_optimization(sf_raman_params,freq,ram,roa,exp_ram,exp_roa):
                par_a,par_b,par_c,par_d = sf_raman_params

                ftt = 2

                freq = freq[::ftt]
                ram = ram[::ftt]
                roa = roa[::ftt]
                exp_ram = exp_ram[::ftt]
                exp_roa = exp_roa[::ftt]

                #shift
                sf_array = (par_b*1/(1+np.exp((-freq+par_d)/par_c))+par_a*(1-1/(1+np.exp((-freq+par_d)/par_c))))
                freq_new_tmp = freq*sf_array

                minv = np.min(freq)
                maxv = np.max(freq)

                freq_new = np.zeros(len(freq))
                ram_new = np.zeros(len(freq))
                roa_new = np.zeros(len(freq))
                for j,freq_v in enumerate(np.linspace(minv,maxv,len(freq))):
                    ram_new[j] = calculate_point(freq_v,freq_new_tmp,ram)
                    roa_new[j] = calculate_point(freq_v,freq_new_tmp,roa)
                    freq_new[j] = freq_v

                # Calculate differences for Raman/ROA - S_fg
                S_fg_ram = S_fg_calculate(ram_new,exp_ram)
                ram_diff = c_ram_0*c_ram*((1 - S_fg_ram)**c_ram_exp)

                S_fg_roa = S_fg_calculate(roa_new,exp_roa)
                roa_diff = c_roa_0*c_roa*((1 - S_fg_roa)**c_roa_exp)
                
                # roa factor - at the beginning, ROA->0 so frequecies could be shifted by a weird factor. 
                # As the optimization progress, we strive to use ROA more and more. 
                diff_roa_f = 1.3*S_fg_roa if S_fg_roa > 0 else 0
                
                diff = ram_diff + diff_roa_f*roa_diff

                return diff

            opt_result=optimize.differential_evolution(scaling_function_par_optimization,\
            bounds=set_raman_shifting_function_parameters_range,\
            args=(freq,ram,roa,exp_data[1],exp_data[2]),\
            maxiter=100)


            par_a,par_b,par_c,par_d = opt_result.x
            sf_raman_params = par_a,par_b,par_c,par_d

            for idx,i in enumerate(sim_data[1]):
                raman_data_scaled[idx],roa_data_scaled[idx] = shift_raman_roa_data(sf_raman_params,exp_data[0],sim_data[1][idx],sim_data[2][idx],exp_data[1],exp_data[2])

            os.makedirs('tmp_files',exist_ok=True)
            with open('tmp_files/raman_roa_optimized_scaling_function_parameters.dat','a') as fw:
                fw.write("{0} {1} {2} {3} \n".format(par_a,par_b,par_c,par_d))

    # assign weights_min/error_min
    if difference <= error_min:
        error_min = 1*difference
        weights_min = 1*weights_i
        error_min_counter = int(1*counter)
        
        # Lets update the overall weghts min  array
        for wi in zip(weights_indices_kept,weights_i):
            overall_weights_min[wi[0]] = wi[1]        
        


    # print diff
    if counter%10000 == 0:
        print("step:{0}, non-zero structures: {8}, overall cost: {7:5.2f} ram:{1:5.2f} roa:{2:5.2f} sch:{3:5.2f} scc:{4:5.2f} jij: {5:5.2f} strb: {6:5.2f}"\
              .format(counter,ram_diff,roa_diff,sch_diff,scc_diff,jij_diff,str_bias,difference,structures_selected))

    if counter%1 == 0:
        error_data_array_i.append([ram_diff,roa_diff,sch_diff,scc_diff,jij_diff,structures_selected])

    # increase counter v
    counter+=1
    
    return difference


def optimize_weights(all_data,experimental_data_av,**kwargs):

    # assign sim/exp data
    sim_data = all_data.sim_data_array
    exp_data = all_data.exp_data_array

    # error data_array
    global error_data_array_i
    error_data_array_i = []

   # weights/error min
    global error_min,weights_min,error_min_counter,overall_weights_min
    error_min = 10**10
    weights_min = np.ones(len(sim_data[0]))
    overall_weights_min = np.ones(len(sim_data[0]))
    error_min_counter = 0

    ### Set arguments from kwargs ###

    global stop_optimization_when_n_structures_reached
    global relative_optimization_weights
    global stop_optimization_when_n_structures_reached
    global c_ram,c_roa,c_sch,c_scc,c_jij

    try:
        if len(kwargs['stop_optimization_when_n_structures_reached']) == 1:
            stop_optimization_when_n_structures_reached = kwargs['stop_optimization_when_n_structures_reached']
        else:
            if all_data.iteration >= 1:
                stop_optimization_when_n_structures_reached = kwargs['stop_optimization_when_n_structures_reached'][all_data.iteration-1]
            else:
                stop_optimization_when_n_structures_reached = 30
    except:
        stop_optimization_when_n_structures_reached = 100

    try:
        stop_optimization_when_error_not_decreasing = kwargs['stop_optimization_when_error_not_decreasing']
    except:
        stop_optimization_when_error_not_decreasing = True

    try:
        recalculate_raman_shifting_function = kwargs['recalculate_raman_shifting_function']
    except:
        recalculate_raman_shifting_function = True

    # Get relative optimization weights calculation
    try:
        relative_optimization_weights = kwargs['relative_optimization_weights']
        print("Relative optimization weights are: ",kwargs['relative_optimization_weights'])
    except:
        print("Problem with setting relative optimization weights - uset-input. Setting to defaults.")
        relative_optimization_weights = [1,1,1,1,1]
    c_ram,c_roa,c_sch,c_scc,c_jij = relative_optimization_weights

    ### Set arguments from kwargs - end ###

    global counter
    global par_a,par_b,par_c,par_d
    counter = 0
    par_a,par_b,par_c,par_d = all_data.sim.scaling_function

    # Set global Raman/ROA data which is gonan be scaled
    global raman_data_scaled, roa_data_scaled
    raman_data_scaled = sim_data[1].copy()
    roa_data_scaled = sim_data[2].copy()

    ############## Initial round ####################
    # Prepare weights array where the optimized weights will be stored
    
    def callback_DE(x,convergence):            
        if np.sum(x>0) < 300:
            return True
        
    # to calculate weights

    weights_indices_kept = np.arange(np.shape(sim_data[0])[0])
    original_weights = np.shape(sim_data[0])[0]
    
    nonzero_structures = len(weights_indices_kept)

    if nonzero_structures >= 1000:
        set_pop = 1
        set_maxiter = 20
    elif (nonzero_structures >= 500) and (nonzero_structures < 1000):
        set_pop = 2
        set_maxiter = 20
    elif (nonzero_structures >= 400) and (nonzero_structures < 500):
        set_pop = 2
        set_maxiter = 30
    elif (nonzero_structures >= 300) and (nonzero_structures < 400):
        set_pop = 2
        set_maxiter = 50
    elif (nonzero_structures >= 200) and (nonzero_structures < 300):
        set_pop = 5
        set_maxiter = 100
    else:
        set_pop = 10
        set_maxiter = 100    


    b0=[(0,1) for i in range(len(sim_data[0]))]
    print("First optimization round")
    print("Optimization will stop when # structures is less than {}.".format(stop_optimization_when_n_structures_reached))
    print("Best result obtained during the whole optimization will be picked.")
    str_print = ''
    if experimental_data_av[0] == 1:
        str_print += 'Raman, '
    if experimental_data_av[1] == 1:
        str_print += 'ROA, '
    if experimental_data_av[2] == 1:
        str_print += 'H shifts, '
    if experimental_data_av[3] == 1:
        str_print += 'C shifts, '
    if experimental_data_av[4] == 1:
        str_print += 'Jij, '
    print("Counting errors of: ", str_print[:-2])

    counter=0
    opt_result=optimize.differential_evolution(par_optimization,\
            bounds=b0,\
            tol=10**-10,\
            args=(sim_data,exp_data,experimental_data_av,weights_indices_kept,original_weights,sim_data,kwargs),\
            popsize=set_pop,\
            maxiter=set_maxiter,
            callback = callback_DE,\
            polish=True,)

    print("Best results so far, it: ",error_min_counter,',sum_error: ',round(sum(error_data_array_i[error_min_counter][:-1]),2),',error array: ',np.round(error_data_array_i[error_min_counter],2))

    weights = weights_min/np.sum(weights_min)
    fun_error = 1*error_min
    overall_weights_min = 1*weights

    # at the end of opt round, recalculate optimize shifting fnc 
    if (recalculate_raman_shifting_function == True) and (experimental_data_av[0] == 1):
        recalculate_shifting_function(sim_data,sim_data,exp_data,overall_weights_min,kwargs)
    
    nonzero_structures =  np.sum(weights > 0.0001*np.max(weights))
    it_counter = 0
    while (nonzero_structures > stop_optimization_when_n_structures_reached):
        print("Subsequent optimization round {0}.".format(it_counter))
        
        nonzero_structures =  np.sum(weights > 0.0001*np.max(weights))
        
        # DE optimization settings
        if nonzero_structures >= 1000:
            set_pop = 1
            set_maxiter = 20
        elif (nonzero_structures >= 500) and (nonzero_structures < 1000):
            set_pop = 2
            set_maxiter = 20
        elif (nonzero_structures >= 400) and (nonzero_structures < 500):
            set_pop = 2
            set_maxiter = 30
        elif (nonzero_structures >= 300) and (nonzero_structures < 400):
            set_pop = 2
            set_maxiter = 50
        elif (nonzero_structures >= 200) and (nonzero_structures < 300):
            set_pop = 5
            set_maxiter = 100
        else:
            set_pop = 10
            set_maxiter = 100

        weights_indices = np.arange(len(weights))
        selection_array = weights > 0.0001*np.max(weights)

        sim_data_truncated = []
        for i in sim_data:
            if len(i) != 0:
                sim_data_truncated.append(i[selection_array])
            else:
                sim_data_truncated.append(np.array([]))
        raman_data_scaled = sim_data[1][selection_array]
        roa_data_scaled = sim_data[2][selection_array]

        weights_indices_kept = weights_indices[selection_array]
 
        weights_min = weights[selection_array]

        # opt with truncated arrays -> weights truncated new
        b0=[(0,1) for i in range(len(sim_data_truncated[0]))]
        
        def callback_DE(x,convergence):                
            if np.sum(x>0) < 250:
                return True
            else:
                return False

        opt_result=optimize.differential_evolution(par_optimization,\
                bounds=b0,\
                tol=10**-10,\
                args=(sim_data_truncated,exp_data,experimental_data_av,weights_indices_kept,original_weights,sim_data,kwargs),\
                popsize=set_pop,\
                maxiter=set_maxiter,)

        print("Best results so far, it: ",error_min_counter,',sum_error: ',round(sum(error_data_array_i[error_min_counter][:-1]),2),',error array: ',np.round(error_data_array_i[error_min_counter],2))

        weights_truncated = weights_min/np.sum(weights_min)

        # Lets update the weights array
        for i in zip(weights_indices_kept,weights_truncated):
            weights[i[0]] = i[1]

        # set the min opt error
        if error_min < fun_error:
            overall_weights_min = 1*weights 
            fun_error = 1*error_min 

        # at the end of opt round, recalculate optimize shifting fnc 
        if (recalculate_raman_shifting_function == True) and (experimental_data_av[0] == 1):
            recalculate_shifting_function(sim_data_truncated,sim_data,exp_data,overall_weights_min,kwargs)

        ### Checks to manage the optimization ###
        # Non zero structures
        nonzero_structures_after_opt =  np.sum(weights > 0.0001*np.max(weights))
        if nonzero_structures_after_opt < stop_optimization_when_n_structures_reached:
            print("Obtained less than {0} structures. Stopping the optimization.".format(stop_optimization_when_n_structures_reached))
            nonzero_structures =  np.sum(weights > 0.0001*np.max(weights))
            break 

        # Check that the error is decreasing during the optimization
        # If not and we did not reach # stop_optimization_when_n_structures_reached, then try to cut some structures
        elif (1.025*opt_result.fun > fun_error) or (1.05*nonzero_structures_after_opt > stop_optimization_when_n_structures_reached):
#            print('Trying to set weights of last 10 % structures to 0.')
            sa = np.sort(np.argwhere(weights>0))
            sa = sa[:int(0.10*len(sa))]
            weights[sa] = 0
 
            if (np.sum(weights>0) < stop_optimization_when_n_structures_reached):
                print("Obtained less than {0} structures. Stopping the optimization.".format(stop_optimization_when_n_structures_reached))
 
        # set a stop so the loop does not run infinitelly
        if it_counter > 100:
            print("Max number of iterations reached. Stopping.")
            break
        it_counter += 1

#     if kwargs['recalculate_raman_shifting_function'] == True:
#         # final recalculation of a shifting fnc
#         counter = 250_000_00
#         par_optimization(overall_weights_min,sim_data,exp_data,experimental_data_av,np.arange(np.shape(sim_data[0])[0]),overall_weights_min,sim_data,kwargs)
        
    # convert error array to numpy array
    error_data_array_i = np.array(error_data_array_i)

    class store_results():
        def __init__(self,weights_min,error_min,error_data_array_i,error_min_counter):
            self.weights = weights_min/np.sum(weights_min)
            self.error = error_min
            self.error_data_array_i = error_data_array_i
            self.error_min_counter = error_min_counter
 
    stored_results = store_results(overall_weights_min,error_min,error_data_array_i,error_min_counter)
    

    return stored_results



