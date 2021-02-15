import os
import shutil
import glob

def run_initial_md_simulation(cluster_sim_nt,n_iteration):
    # Now copy md files here
    for file_i in glob.glob('needed_files/md_inp_files/*'):
        dest_dir="new_iteration_{0}/MD/.".format(n_iteration)
        os.system('cp -r {0} {1}'.format(file_i,dest_dir))
    # Now run the simulations

    c_path=os.getcwd()
    os.chdir('new_iteration_{0}/MD'.format(n_iteration))

    grofile=glob.glob('*.gro')[0]
    topfile=glob.glob('*.top')[0]
    indexfile=glob.glob('*.ndx')[0]

    if cluster_sim_nt == 0:
        os.system('cp {0} structure_start_sim.gro'.format(grofile))
    else:
        os.system('cp ../MD_trj/structure_start_sim_prev.gro structure_start_sim.gro')

    os.system("gmx grompp -f md_prod_it0.mdp -c structure_start_sim.gro -p {0} -o job.tpr -n {1} -maxwarn 5".format(topfile,indexfile))
    os.system("mdrun_plumed -s job.tpr -v -deffnm job{0}  -nsteps 5000 -nt 1 -plumed  plumed_restraint.dat".format(cluster_sim_nt))

    os.chdir(c_path)

def generate_new_structures(molecule_features,n_iteration,number_of_new_structures,**kwargs):
    try:
        shutil.rmtree('new_iteration_{0}'.format(n_iteration))
    except FileNotFoundError:
        pass

    os.makedirs("new_iteration_{0}".format(n_iteration))
    os.makedirs("new_iteration_{0}/MD_trj".format(n_iteration))

    # make the simulation until there are no crashes (while statement)
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
        run_initial_md_simulation(cluster_sim_nt,n_iteration)

        # Check that it did not crash
        if glob.glob("new_iteration_{0}/MD/step*".format(n_iteration)) == []:
            os.system('cp new_iteration_{0}/MD/job{1}.xtc new_iteration_{0}/MD_trj/.'.format(n_iteration,cluster_sim_nt))
            os.system('cp new_iteration_{0}/MD/job{1}.gro new_iteration_{0}/MD_trj/structure_start_sim_prev.gro'.format(n_iteration,cluster_sim_nt))
            cluster_sim_nt += 1
        else:
            pass

    # get rid of the first structure of all job*.xtc files - due to the restarts, these are the same
    c_path=os.getcwd()
    os.chdir('new_iteration_{0}/MD_trj'.format(n_iteration))
    for i in glob.glob('job*xtc'):
        os.system("printf \"0\n\"|gmx trjconv -f {0} -s structure_start_sim_prev.gro -o job_tmp.xtc -b 1".format(i))
        os.system("mv job_tmp.xtc {0}".format(i))
    os.chdir(c_path)

    # calculate features - make plumed file
    molecule_features.write_plumed_file_MD0_check_features("new_iteration_{0}/MD_trj/plumed_features.dat".format(n_iteration),'',1)
    c_path=os.getcwd()
    os.chdir('new_iteration_{0}/MD_trj'.format(n_iteration))
    os.system("gmx trjcat -f job*.xtc -o job_cat.xtc -cat")
    grofile=glob.glob(c_path+'/needed_files/md_inp_files/*gro')[0]
    os.system('cp ../MD/job.tpr job.tpr')
    os.system("gmx trjcat -f job*.xtc -o job_cat.xtc -cat")
    os.system('cp {0} structure.gro'.format(grofile))
    os.system("printf \"1\n 0\n\"|gmx trjconv -f job_cat.xtc -s job.tpr -pbc mol -center -o job_cat_center.xtc")

    os.system("plumed driver --plumed plumed_features.dat --mf_xtc job_cat_center.xtc")

    # copy tpr file
    os.system('cp ../MD/job.tpr job.tpr')

    os.chdir(c_path)
