from mdshark.main import *
from mdshark.common import run
import pprint
import pickle

from mdshark import m_molecular_features
from mdshark.common import logger

if __name__ == "__main__":

    top_dir = os.getcwd() + '/glc_bm_test'

    weights = None
    all_data = None

    it_start = 0
    it_start = 2
    num_iterations = 1

    for n_iteration in range(it_start, it_start + num_iterations):

        logger.info(f"Starting mdshark optimization loop")

        if n_iteration == 0:
            logger.info(f" -- it 0")

            # Load molecular features 
            molecule_features = m_molecular_features.MoleculeFeatures(
                    f'{top_dir}/needed_files/md_inp_files/*.gro', 
                    f'{top_dir}/needed_files/md_inp_files/*.itp')

            mds = MDSharkOptimizer(
                    top_dir=top_dir,
                    generate_structures=10,
                    molecule_features=molecule_features)

        else: 
            logger.info(f" -- it {n_iteration}, loading result from the previous iteration ...")

            with open(f"{top_dir}/results_{n_iteration-1}.pkl", 'rb') as file:      
                mds, all_data, error_data_array, stored_results = pickle.load(file)
                weights = stored_results.weights

        os.chdir(top_dir)
        mds.n_iteration = n_iteration

        mds.initialize_structures(weights=weights, all_data=all_data)

        mds.g_submit(n_iteration)

        mds.calculate_features(n_iteration)
        all_data, error_data_array, stored_results = mds.optimize(n_iteration)
        
        weights = stored_results.weights

        with open(f"{top_dir}/results_{n_iteration}.pkl", 'wb') as file:
            pickle.dump((mds, all_data, error_data_array, stored_results,), file)