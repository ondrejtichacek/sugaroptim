import os
import pprint
import pickle

if __name__ == "__main__":

    top_dir='glc_bm_test'

    os.chdir(top_dir)

    n_iteration = 1
    
    with open(f"results_{n_iteration}.pkl", 'rb') as file:
        mds, all_data, error_data_array, stored_results = pickle.load(file)

    mds.convergence_plots(all_data) # check the convergence