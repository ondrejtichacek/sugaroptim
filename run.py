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
    # it_start = 6
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

            data = {
            'all_data': None,
            'error_data_array': None,
            'stored_results': None,
            'weights': None,
        }

        else:
            logger.info(
                f" -- it {n_iteration}, loading result from the previous iteration ...")

            with open(f"{top_dir}/results_{n_iteration-1}.pkl", 'rb') as file:
                mds, data, = pickle.load(file)

        os.chdir(top_dir)
        mds.n_iteration = n_iteration

        mds.initialize_structures(weights=data['weights'], all_data=data['all_data'])

        # Generate new structures
        mds.generate_new_structures()

        # Optimize MD frames
        mds.optimize_new_structures()

        # Generate gaussian input files
        mds.create_qm_input_files()

        # Perform gaussian calculations
        mds.perform_qm_calculations()

        # Extract data and calculate features
        mds.calculate_features()

        # Perform heuristic optimization
        all_data, error_data_array, stored_results = mds.optimize()

        # Save data
        data = {
            'all_data': all_data,
            'error_data_array': error_data_array,
            'stored_results': stored_results,
            'weights': stored_results.weights,
        }

        with open(f"{top_dir}/results_{n_iteration}.pkl", 'wb') as file:
            pickle.dump(
                (mds, data,), file)
