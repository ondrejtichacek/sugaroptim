import os
import subprocess
import glob
import shutil
import multiprocessing
import mdtraj
from functools import partial
import numpy as np

import pathlib
from pathlib import Path

import config
import common
from common import run

# Function to write plumed restraint file for MD optimization
def make_opt_plumed_file(frame,molecule_features,kwargs_dict):
    # Make plumed list
    molecule_features.write_plumed_print_features(frame+'_plumed.dat',frame+'_',1)

    # first get values of the frame
    run(f"printf \"0\n\"|{config.path['gmx']} trjconv -f {frame} -s {frame} -o {frame}_plumed_restraint_values.xtc")
    run(f"plumed driver --mf_xtc {frame}_plumed_restraint_values.xtc --plumed {frame}_plumed.dat")

    restrain_features_values = np.loadtxt(frame+'_plumed_output_features.dat')
    restrain_features_values = restrain_features_values[1:]
    run(f'rm {frame}_plumed.dat {frame}_plumed_output_features.dat {frame}_plumed_restraint_values.xtc')

    molecule_features.write_plumed_file_MD_opt(frame+'_plumed_restraint.dat',frame+'_',1000,kwargs_dict,restrain_features_values)

# MD optimize individual frames
def optimize_frame(i,molecule_features,kwargs_dict):
    # Make plumed file
    make_opt_plumed_file(i,molecule_features,kwargs_dict)
    topol=glob.glob('*.top')[0]

    # min
    run(f"{config.path['gmx']} grompp -f md_min.mdp -c {i} -p {topol} -o {i}_min.tpr -n index.ndx -po {i}_mdout1.mdp -maxwarn 5&> {i}_grompp_min.err")
    run(f"{config.path['mdrun_plumed']} -s {i}_min.tpr -v -deffnm {i}_min -nt 1 -plumed {i}_plumed_restraint.dat &> {i}_min.err")
    # MD opt
    run(f"{config.path['gmx']} grompp -f md_opt.mdp -c {i}_min.gro -p {topol} -o {i}_opt.tpr -n index.ndx -r {i}_min.gro -po {i}_mdout2.mdp -maxwarn 5 &> {i}_grompp_opt.err")
    run(f"{config.path['mdrun_plumed']} -s {i}_opt.tpr -v -deffnm {i}_opt -nsteps 20000 -nt 1 -plumed {i}_plumed_restraint.dat &> {i}_opt.err")
    # to xtc
    run(f"printf \"0\n\"|{config.path['gmx']} trjconv -f {i}_opt.gro -s {i}_opt.tpr -o {i}_opt.xtc &> {i}_trjconv.err")
# do the MD optimization for all
def optimize_individual_frames(n_iteration,number_of_new_structures,molecule_features_class,**kwargs):
    p_dir=os.getcwd()

    grofile =  p_dir+'/'+glob.glob('needed_files/md_inp_files/*gro')[0]
    itp_file = p_dir+'/'+glob.glob('needed_files/md_inp_files/*itp')[0]
    pdb_file = p_dir+'/'+glob.glob('needed_files/md_inp_files/*pdb')[0]
    tpr_file = p_dir+"/new_iteration_{0}/MD_trj/job.tpr".format(n_iteration)
    xtc_file = p_dir+"/new_iteration_{0}/MD_trj/job_cat.xtc".format(n_iteration)

    try:
        shutil.rmtree("new_iteration_{0}/MD_frames".format(n_iteration))
    except FileNotFoundError:
        pass
    os.makedirs("new_iteration_{0}/MD_frames".format(n_iteration))

    # go to md frames dir
    c_path=os.getcwd()
    os.chdir('new_iteration_{0}/MD_frames'.format(n_iteration))

    t_xtc = mdtraj.load(xtc_file,top = pdb_file)

    # specify that you take every xth frame and make it
    freq_to_take = int(len(t_xtc)/number_of_new_structures)
    run(f"printf \"1\n 0\n\"|{config.path['gmx']} trjconv -f {xtc_file} -s {tpr_file} -pbc mol -center -o job_c.xtc -skip {freq_to_take}")

    # Now MD minimize them
    run('cp -r {0}/needed_files/md_inp_files/* .'.format(p_dir))
    Path("frame*.gro").unlink(missing_ok=True)
    run(f"printf \"0\n\"|{config.path['gmx']} trjconv -f job_c.xtc -s {tpr_file} -sep -o frame.gro")

    # select all frames
    all_frames=glob.glob('frame*gro')

    # do the optimization
    try:
        NUM_WORKERS = kwargs['num_workers']
    except:
        NUM_WORKERS = 1
        print("Problem with setting NUM_WORKERS_C, falling back to single worker.")

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        pool.map(partial(optimize_frame,molecule_features = molecule_features_class,kwargs_dict = kwargs), all_frames)
        pass

    # cat the frames together
    run(f"{config.path['gmx']} trjcat -f frame*opt.xtc -o frames_opt_cat.xtc -cat")
    molecule_features_class.write_plumed_print_features('plumed_features.dat','',1)

    run("plumed driver --plumed plumed_features.dat --mf_xtc frames_opt_cat.xtc")

    os.chdir(c_path)



