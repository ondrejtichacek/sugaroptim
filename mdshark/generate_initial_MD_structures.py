import os
import subprocess
import shutil
import glob
from tqdm import tqdm

from mdshark import config
from mdshark.common import run, logger

def run_initial_md_simulation(cluster_sim_nt,n_iteration):
    # Now copy md files here
    for file_i in glob.glob('needed_files/md_inp_files/*'):
        dest_dir="new_iteration_{0}/MD/.".format(n_iteration)
        run('cp -r {0} {1}'.format(file_i,dest_dir))
    # Now run the simulations

    c_path=os.getcwd()
    os.chdir('new_iteration_{0}/MD'.format(n_iteration))

    grofile=glob.glob('*.gro')[0]
    topfile=glob.glob('*.top')[0]
    indexfile=glob.glob('*.ndx')[0]

    if cluster_sim_nt == 0:
        run('cp {0} structure_start_sim.gro'.format(grofile))
    else:
        run('cp ../MD_trj/structure_start_sim_prev.gro structure_start_sim.gro')
    
    try:
        run(f"{config.path['gmx']} grompp -f md_prod_it0.mdp -c structure_start_sim.gro -p {topfile} -o job.tpr -n {indexfile} -maxwarn 5")
        run(f"{config.path['mdrun_plumed']} -s job.tpr -v -deffnm job{cluster_sim_nt}  -nsteps 5000 -nt 1 -plumed  plumed_restraint.dat")
    except subprocess.CalledProcessError as e:
        logger.warning("Exception caught - md simulation error. This may be expected. See the log above in case of other issues.")

    os.chdir(c_path)

def generate_new_structures(molecule_features,n_iteration,number_of_new_structures,**kwargs):

    logger.notice("Generating new structures with md simulation")

    try:
        shutil.rmtree('new_iteration_{0}'.format(n_iteration))
    except FileNotFoundError:
        pass

    os.makedirs("new_iteration_{0}".format(n_iteration))
    os.makedirs("new_iteration_{0}/MD_trj".format(n_iteration))

    # make the simulation until there are no crashes (while statement)
    pbar = tqdm(total=number_of_new_structures)
    cluster_sim_nt = 0
    while cluster_sim_nt < number_of_new_structures:
        try:
            shutil.rmtree("new_iteration_{0}/MD".format(n_iteration))
        except FileNotFoundError:
            pass
        os.makedirs("new_iteration_{0}/MD".format(n_iteration))

        # generate metadynamics plumed file
        molecule_features.write_plumed_file_MD0("new_iteration_{0}/MD/plumed_restraint.dat".format(n_iteration),'',1000,kwargs)

        # run the MD simulation
        run_initial_md_simulation(cluster_sim_nt, n_iteration)

        # Check that it did not crash
        if glob.glob("new_iteration_{0}/MD/step*".format(n_iteration)) == []:
            run('cp new_iteration_{0}/MD/job{1}.xtc new_iteration_{0}/MD_trj/.'.format(n_iteration,cluster_sim_nt))
            run('cp new_iteration_{0}/MD/job{1}.gro new_iteration_{0}/MD_trj/structure_start_sim_prev.gro'.format(n_iteration,cluster_sim_nt))
            cluster_sim_nt += 1
            pbar.update(1)
        else:
            pass
    pbar.close()

    # get rid of the first structure of all job*.xtc files - due to the restarts, these are the same
    c_path=os.getcwd()
    os.chdir('new_iteration_{0}/MD_trj'.format(n_iteration))
    for i in glob.glob('job*xtc'):
        run(f"printf \"0\n\"|{config.path['gmx']} trjconv -f {i} -s structure_start_sim_prev.gro -o job_tmp.xtc -b 1")
        run("mv job_tmp.xtc {0}".format(i))
    os.chdir(c_path)

    # calculate features - make plumed file
    molecule_features.write_plumed_file_MD0_check_features("new_iteration_{0}/MD_trj/plumed_features.dat".format(n_iteration),'',1)
    c_path=os.getcwd()
    os.chdir('new_iteration_{0}/MD_trj'.format(n_iteration))
    grofile=glob.glob(c_path+'/needed_files/md_inp_files/*gro')[0]
    run('cp ../MD/job.tpr job.tpr')
    run(f"{config.path['gmx']} trjcat -f job*.xtc -o job_cat.xtc -cat")
    run('cp {0} structure.gro'.format(grofile))
    run(f"printf \"1\n 0\n\"|{config.path['gmx']} trjconv -f job_cat.xtc -s job.tpr -pbc mol -center -o job_cat_center.xtc")

    run("plumed driver --plumed plumed_features.dat --mf_xtc job_cat_center.xtc")

    # copy tpr file
    run('cp ../MD/job.tpr job.tpr')

    os.chdir(c_path)
