from mdshark.main import *
from mdshark.common import run
import pprint
import pickle

from mdshark import m_molecular_features

LOCAL = "/home/ondrej/repo/vlada_optim_sugars/sugaroptim/"
REMOTE = "aurum:/home1/tichacek/sugaroptim/sugaroptim/"

if __name__ == "__main__":

    # Load molecular features 
    molecule_features = m_molecular_features.MoleculeFeatures(
            'glc_bm_test/needed_files/md_inp_files/*.gro', 
            'glc_bm_test/needed_files/md_inp_files/*.itp')

    mds = MDSharkOptimizer(
            top_dir='glc_bm_test',
            generate_structures=10,
            molecule_features=molecule_features)

    mds.initialize_structures()
    # run(f"rsync -av --delete {LOCAL}/glc_bm_test/new_iteration_0/input_files/ {REMOTE}/glc_bm_test/new_iteration_0/input_files/")

    n_iteration = 0
    mds.g_submit(n_iteration)

    # run(f"rsync -av {REMOTE}/glc_bm_test/new_iteration_0/input_files/ {LOCAL}/glc_bm_test/new_iteration_0/input_files/")

    n_iteration = 1
    mds.calculate_features(n_iteration)
    all_data, error_data_array, stored_results = mds.optimize(n_iteration)
    
    with open(f"results_{n_iteration}.pkl", 'wb') as file:
        pickle.dump((all_data, error_data_array, stored_results,), file)

    with open(f"results_{n_iteration}.pkl", 'rb') as file:
      
        all_data, error_data_array, stored_results = pickle.load(file)


    mds.convergence_plots(all_data) # check the convergence

    pass