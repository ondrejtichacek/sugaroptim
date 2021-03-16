from mdshark.main import *
from mdshark.common import run
import pprint
import pickle

from mdshark import m_molecular_features

if __name__ == "__main__":

    top_dir = 'glc_bm_test'

    weights = None
    all_data = None

    for n_iteration in range(1, 2):

        if True:
            with open(f"{top_dir}/results_{n_iteration-1}.pkl", 'rb') as file:      
                mds, all_data, error_data_array, stored_results = pickle.load(file)
                weights = stored_results.weights

        else:
            pass

        # Load molecular features 
        molecule_features = m_molecular_features.MoleculeFeatures(
                f'{top_dir}/needed_files/md_inp_files/*.gro', 
                f'{top_dir}/needed_files/md_inp_files/*.itp')

        mds = MDSharkOptimizer(
                top_dir=top_dir,
                generate_structures=10,
                molecule_features=molecule_features)

        mds.initialize_structures(weights=weights, all_data=all_data)

        mds.g_submit(n_iteration)

        mds.calculate_features(n_iteration)
        all_data, error_data_array, stored_results = mds.optimize(n_iteration)
        
        weights = stored_results.weights

        with open(f"{top_dir}/results_{n_iteration}.pkl", 'wb') as file:
            pickle.dump((mds, all_data, error_data_array, stored_results,), file)