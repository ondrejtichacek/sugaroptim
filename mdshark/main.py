import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import submitit
from tqdm import tqdm
from dataclasses import dataclass

from mdshark import \
    common, \
    config, \
    extract_sim_data, \
    prepare_g_input_files, \
    m_molecular_features, \
    optimize_MD_frame, \
    load_sim_exp_data, \
    store_and_plot_spectra, \
    optimize_weights, \
    calculate_optimized_features_distribution, \
    optimize_MD_frame, \
    generate_n_iteration_MD_structures, \
    generate_initial_MD_structures

from mdshark.common import run, run_submit, logger, get_default_executor

use_submitit = True

@dataclass
class MDSharkOptimizer:
    top_dir: str = None
    num_structures: int = 50  # number of generated structures
    stop_optimization_when_n_structures_reached_C: int = None
    molecule_features: m_molecular_features.MoleculeFeatures = None

    # this speeds up optimization of MD frames, ! be careful with number > 2x # of cpus
    md_num_workers: int = os.cpu_count()

    qm_memory: int = 40
    qm_nproc: int = 36

    def __post_init__(self):

        if self.stop_optimization_when_n_structures_reached_C is None:
            self.stop_optimization_when_n_structures_reached_C = 3 * \
                self.num_structures // 5

        self.n_iteration = 0  # n iteration

        ##################################
        # General settings
        # if True, then all iterations will be calculated (to see the progress), else only latest one
        self.calculate_all_C = False

        ##################################
        # MD optimization settings
        # do you want to freeze amide bonds (HNCO dihedral andles) in original conformation?
        self.freeze_amide_bonds_C = False

        ##################################
        # Obtain optimized weights
        # overide optimization default settings ###

        # when bias_n_structures_reached_f_C [%] of the structures are non-zero, proceed to subsequent opt round.
        self.bias_n_structures_reached_C = False
        self.bias_n_structures_reached_f_C = 1/2
        self.recalculate_raman_shifting_function_C = True
        self.set_raman_shifting_function_parameters_range_C = [
            (0.95, 1.1), (0.95, 1.0), (15, 1000), (500, 2500)]
        self.relative_optimization_weights_C = [
            1, 1, 1, 1, 1]  # Ram/ROA/H/C/Jij
        # set what % to sample according to the optimized distribution, rest will be random
        self.set_K_nd_C = 2/3
        self.do_not_filter_C = True  # do not filter H/C/Jij dat

        # Experimental data for? iteration 1 or more
        ed_raman = 1
        ed_roa = 1
        ed_shifts_H = 1
        ed_shifts_C = 1
        ed_spinspin = 1
        self.experimental_data_av = [
            ed_raman, ed_roa, ed_shifts_H, ed_shifts_C, ed_spinspin]

        ##################################
        ### simulation data extraction ###
        # Raman/ROA data extraction -> use what freq range you need (calculating bigger interval costs more)
        self.raman_frequency_range_C = [50, 2000]
        # Use Raman shifting fuction? If False, freqeuncies will not be shifted
        self.use_raman_shifting_function_C = True

        ##################################
        # QM calculations settings
        # used VDW parameters in QM calculations
        self.oniom_vdw_parameters_kwargs = {
            'oniom_vdw_parameters_C': 'glycam_saccharides'}
        # self.oniom_vdw_parameters_kwargs = {'oniom_vdw_parameters_C': 'proteins_1'}
        # self.oniom_vdw_parameters_kwargs = {'oniom_vdw_parameters_C': 'proteins_2'}
        # self.oniom_vdw_parameters_kwargs = {'oniom_vdw_parameters_C': 'charmm-cgenff', 'mol_file_path': 'needed_files/md_inp_files/mol.itp', 'ffnonbonded_file_path': 'needed_files/md_inp_files/top/charmm36-mar2019.ff/ffnonbonded.itp'}

        # prepare QM input files method
        self.prepare_qm_input_files_C = 'm2_raman_high'  # m1_raman_low/m2_raman_high

        # gaussian input files nproc & memory
        self.qm_m_nproc = [self.qm_memory, self.qm_nproc]

        # Calculate Raman/ROA  x H/C NMR shifts  x Jij couplings?
        c_vib = 1
        c_shifts = 1
        c_spinspin = 1
        self.w_calculate = [c_vib, c_shifts, c_spinspin]

    def perform_qm_calculations(self):

        logger.info("Performing QM calculations")

        write_folder = Path(f"new_iteration_{self.n_iteration}/input_files")

        cmd = ['bash', 'qsub_ROA_calc.inp.dqs']

        jobs = []
        for i in range(self.num_structures):
            cwd = write_folder / f"f{self.n_iteration}_{i:05d}"
            job = run_submit(cmd, cwd, 'gaussian', wait_complete=False)
            jobs.append(job)

        num_failed = 0
        for i, job in enumerate(tqdm(jobs)):
            try:
                output = job.result()
            except submitit.core.utils.FailedJobError:
                num_failed += 1
                logger.warning(f"Job {i} failed.")

        if num_failed > 0:
            logger.warning(
                f"{num_failed} out of {self.num_structures} jobs failed.")

            tol = 0.1

            if num_failed / self.num_structures > tol:
                logger.critical(
                    f"{100 * num_failed / self.num_structures} % of jobs failed. Check the output files...")
                raise(ValueError("Too many jobs failed."))

        logger.notice("Done")

    def generate_new_structures(self):

        logger.info("Generating new structures")

        if self.n_iteration == 0:
    
            T = config.sim_duration['generate_initial_MD_structures.generate_new_structures']
            total_duration = int(np.ceil(10 + self.num_structures * T))

            function = generate_initial_MD_structures.generate_new_structures
            args = (self.molecule_features,
                    self.n_iteration,
                    self.num_structures)
            kwargs = {'freeze_amide_bonds': self.freeze_amide_bonds_C}
        else:
            T = config.sim_duration['generate_n_iteration_MD_structures.generate_new_structures']
            total_duration = int(np.ceil(10 + (self.num_structures // 50 + 1) * T))

            # generate new structures using optimized distributions
            function = generate_n_iteration_MD_structures.generate_new_structures
            args = (self.molecule_features,
                    self.n_iteration,
                    self.num_structures, 
                    self.final_distribution)
            kwargs = {'freeze_amide_bonds': self.freeze_amide_bonds_C}

        if use_submitit is True:

            if total_duration > 4*60:
                raise(ValueError('Too many structures (job would not finish in time).'))

            executor = get_default_executor('plumed')
            executor.update_parameters(
                name='generate_new_structures',
                timeout_min=total_duration,
                mem_gb=2,
                cpus_per_task=1,)

            job = executor.submit(function, *args, **kwargs)

            output = job.result()
        else:
            function(*args, **kwargs)

    def optimize_new_structures(self):

        logger.info("Optimizing new structures")

        function = optimize_MD_frame.optimize_individual_frames
        args = (self.n_iteration,
                self.num_structures,
                self.molecule_features)
        kwargs = {'num_workers': self.md_num_workers,
            'freeze_amide_bonds': self.freeze_amide_bonds_C}

        if use_submitit is True:
            executor = get_default_executor('plumed')
            executor.update_parameters(name='optimize_individual_frames')

            job = executor.submit(function, *args, **kwargs)

            output = job.result()
        else:
            function(*args, **kwargs)

    def create_qm_input_files(self):

        logger.info("Preparing QM input files")

        prepare_g_input_files.prepare_qm_input_files(
            self.molecule_features,
            self.n_iteration,
            self.w_calculate,
            self.qm_m_nproc,
            method=self.prepare_qm_input_files_C,
            **self.oniom_vdw_parameters_kwargs)

    def initialize_structures(self, weights=None, all_data=None):

        if self.n_iteration == 0:

            logger.info("Generating initial MD simulation files")

            # Generate initial plumed file and assign it, generate index file, write fromacs mdp files
            self.molecule_features.write_plumed('needed_files/plumed_original.dat')
            self.molecule_features.gro_to_pdb()
            self.molecule_features.make_index_file()
            self.molecule_features.write_gromacs_mdp_files()

        elif self.n_iteration > 0:

            logger.info("Calculating distribution of optimized features")

            # Assign final distribution
            self.final_distribution = calculate_optimized_features_distribution.calculate_new_distribution(
                weights,
                all_data.sim,
                self.molecule_features,
                K=self.set_K_nd_C)

        else:
            raise(ValueError(f"n_iteration is {self.n_iteration}"))

    @staticmethod
    def calculate_features_after_qm_optimization(molecular_features, directory):
        """
        calculate features after QM optimization (Plumed)
        """

        for file_i in glob.glob(directory+'/f*'):
            molecular_features.write_plumed_print_features(
                file_i+'/plumed_features.dat', file_i+'/', 1)
            run("printf \"0 \\n\"|gmx trjconv -f {0} -s {0} -o {1}/trj_tmp.xtc".format(
                file_i+"/structure_final.gro", file_i))
            run("plumed driver --plumed {0}/plumed_features.dat --mf_xtc {0}/trj_tmp.xtc".format(
                file_i))

            if Path('needed_files/plumed_additional_features.dat').exists():
                molecular_features.write_additional_plumed_print_features(
                    file_i+'/plumed_additional_features.dat', file_i+'/')
                run("plumed driver --plumed {0}/plumed_additional_features.dat --mf_xtc {0}/trj_tmp.xtc".format(
                    file_i))

    def calculate_features(self):

        it_analyze = self.n_iteration

        # Extract data from iteration X
        extract_sim_data.calculate_all_sim_data(
            f'new_iteration_{it_analyze}',
            self.molecule_features,
            use_raman_shifting_function=self.use_raman_shifting_function_C,
            raman_frequency_range=self.raman_frequency_range_C)

        self.calculate_features_after_qm_optimization(
            self.molecule_features,
            f'new_iteration_{it_analyze}/input_files')

    def optimize(self, stop_optimization_when_n_structures_reached_C=None):
        """
        Calculate progressively the new distribution and check convergence
        """

        if stop_optimization_when_n_structures_reached_C is None:
            stop_optimization_when_n_structures_reached_C = self.stop_optimization_when_n_structures_reached_C

        self.new_distribution = []
        self.spectra_data = []
        error_data_array = []

        # assert(self.n_iteration >= 1)

        if self.calculate_all_C == True:

            all_data = load_sim_exp_data.data(
                'new_iteration_0', 0, self.molecule_features,
                self.experimental_data_av,
                do_not_filter=self.do_not_filter_C)

            weights0 = np.ones(len(all_data.sim.file_array))
            weights0 /= np.sum(weights0)

            self.spectra_data.append(store_and_plot_spectra.spectra(
                weights0, all_data, self.experimental_data_av))

            # calculate new distribution
            self.new_distribution.append(calculate_optimized_features_distribution.
                                         calculate_new_distribution(weights0, all_data.sim, self.molecule_features, K=1))

            for i in range(self.n_iteration+1):

                # load data
                path = f'new_iteration_[0-{i}]'
                all_data = load_sim_exp_data.data(
                    path, i, self.molecule_features,
                    self.experimental_data_av,
                    do_not_filter=self.do_not_filter_C)

                # optimize the weights
                stored_results = optimize_weights.optimize_weights(
                    all_data, self.experimental_data_av,
                    recalculate_raman_shifting_function=self.recalculate_raman_shifting_function_C,
                    set_raman_shifting_function_parameters_range=self.set_raman_shifting_function_parameters_range_C,
                    stop_optimization_when_n_structures_reached=stop_optimization_when_n_structures_reached_C,
                    bias_n_structures_reached=self.bias_n_structures_reached_C,
                    bias_n_structures_reached_f=self.bias_n_structures_reached_f_C,
                    relative_optimization_weights=self.relative_optimization_weights_C)

                weights = stored_results.weights
                # error = stored_results.error
                error_data_array_i = stored_results.error_data_array_i

                # when converged - we can plot it
                self.spectra_data.append(store_and_plot_spectra.spectra(
                    weights, all_data, self.experimental_data_av))

                # calculate new distribution
                self.new_distribution.append(calculate_optimized_features_distribution.
                                             calculate_new_distribution(weights, all_data.sim, self.molecule_features, K=self.set_K_nd_C))

                # append the error data array
                error_data_array.append(error_data_array_i)

        else:
            path = f'new_iteration_[0-{self.n_iteration}]'
            all_data = load_sim_exp_data.data(
                path, self.n_iteration, self.molecule_features,
                self.experimental_data_av, do_not_filter=self.do_not_filter_C)

            # optimize the weights
            stored_results = optimize_weights.optimize_weights(
                all_data, self.experimental_data_av,
                recalculate_raman_shifting_function=self.recalculate_raman_shifting_function_C,
                set_raman_shifting_function_parameters_range=self.set_raman_shifting_function_parameters_range_C,
                stop_optimization_when_n_structures_reached=stop_optimization_when_n_structures_reached_C,
                bias_n_structures_reached=self.bias_n_structures_reached_C,
                bias_n_structures_reached_f=self.bias_n_structures_reached_f_C,
                relative_optimization_weights=self.relative_optimization_weights_C)

            weights = stored_results.weights
            error_data_array_i = stored_results.error_data_array_i

            # append the error data array
            error_data_array.append(error_data_array_i)
            self.spectra_data.append(store_and_plot_spectra.spectra(
                weights, all_data, self.experimental_data_av))

        return all_data, error_data_array, stored_results

    def convergence_plots(self, all_data):

        f, ax = plt.subplots(self.new_distribution[0].nfeatures, 1, figsize=(
            7, self.new_distribution[0].nfeatures*5))

        for i in range(np.shape(self.spectra_data)[0]):
            nd = self.new_distribution[i]
            for j in nd.importance_list[:, 0]:
                idx = int(j)
                ax[idx].plot(nd.nd_array[idx][0], nd.nd_array[idx][3], label=i)
                ax[idx].title.set_text(all_data.sim.features_def_list[idx])
                ax[idx].legend(loc="upper right", fontsize=20)
        f.subplots_adjust(hspace=0.4)
        plt.show()

        for i in range(np.shape(self.spectra_data)[0]):
            self.spectra_data[i].plot()


if False:
    # ===================================================================
    # Check last iteration errors

    plt.figure(figsize=(18, 8))
    legend_labels = ['Raman', 'ROA', 'H', 'C', 'J']
    for i in np.arange(np.shape(error_data_array[-1])[-1]-1):
        x = error_data_array[-1][:, i]
        plt.plot(x/x[0], label=legend_labels[i], lw=1)

    overall_err = np.sum(
        error_data_array[-1][:, :-1], axis=1)/len(legend_labels)
    overall_err = overall_err/overall_err[0]
    plt.plot(overall_err, label='overall error', lw=3)
    plt.plot(stored_results.error_min_counter,
             overall_err[stored_results.error_min_counter], 'o', color='k', ms=30, mfc='none', mew=3)
    plt.grid()
    plt.legend(framealpha=0.0, loc=1)
    plt.xlim([0, 1.35*len(x)])
    plt.show()

    plt.figure(figsize=(18, 8))
    structures = error_data_array[-1][:, -1]
    plt.plot(structures, label='# of structures')
    plt.plot(stored_results.error_min_counter,
             structures[stored_results.error_min_counter], 'o', color='k', ms=30, mfc='none', mew=3)

    plt.legend(framealpha=0.0, loc=1)
    plt.xlim([0, 1.35*len(x)])
    plt.grid()
    plt.show()

    err = error_data_array_i[0][:-1]-error_data_array_i[-1][:-1]
    proposed_weights = np.array(self.relative_optimization_weights_C)*np.array(
        [1/(i/np.min(err[err > 0])) if i != 0 else 1 for i in err])

    weights = stored_results.weights

    print("# of non-zero structures/structures")
    print(np.sum(weights > 0.0001*np.max(weights)), len(weights))

    print("Proposed relative optimization weights")
    print(proposed_weights)
    print("Previous rel opt weights")
    print(self.relative_optimization_weights_C)

    # ===================================================================
    # Check spectra

    for i in range(len(spectra_data)):
        print("iteration {0}".format(i))
        spectra_data[i].plot_nmr_shifts_h()
    for i in range(len(spectra_data)):
        print("iteration {0}".format(i))
        spectra_data[i].plot_nmr_shifts_c()
    for i in range(len(spectra_data)):
        print("iteration {0}".format(i))
        spectra_data[i].plot_nmr_spinspin()
    for i in range(len(spectra_data)):
        print("iteration {0}".format(i))
        spectra_data[i].plot_raman()
