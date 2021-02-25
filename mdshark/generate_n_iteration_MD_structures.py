import shutil
import subprocess
import numpy as np
import glob
import shutil
import os

from mdshark import config
from mdshark.common import run

def generate_restrained_MD_plumed_file_targeted_MD(molecule_features,n_iteration,kwargs_dict):
    with open("new_iteration_{0}/MD/plumed_restraint.dat".format(n_iteration),'w') as fw:

        # write molecule_features
        fw.write(molecule_features.write_plumed_features())
        fw.write(molecule_features.write_plumed_stereochemical_centers())
        fw.write(molecule_features.write_plumed_improper_dihedrals())
        fw.write(molecule_features.write_plumed_amide_bonds())

        fw.write(molecule_features.write_plumed_stereochemical_centers_restraints())
        fw.write(molecule_features.write_plumed_improper_dihedrals_restraints())
        fw.write(molecule_features.write_plumed_amide_bond_restraints(kwargs_dict))

        fw.write(molecule_features.write_plumed_stereochemical_centers_restraints_print(1000,''))
        fw.write(molecule_features.write_plumed_improper_dihedrals_restraints_print(1000,''))
        fw.write(molecule_features.write_plumed_amide_bond_restraints_print(1000,''))
        fw.write(molecule_features.write_plumed_features_print(1000,''))
        
        fw.write('\n')

        # Now make metadynamics sim - here we are restraining the molecule!
        for idx,i in enumerate(molecule_features.features_def_list_idx):
            if i == 1:
                fw.write("metad{0}: METAD FILE=HILLS{0} ARG={1} PACE=10 \
SIGMA=0.3 DAMPFACTOR=100 TAU=1 TARGET=target{0}.dat \
GRID_MIN=-pi GRID_MAX=pi GRID_BIN=300 GRID_WSTRIDE=25000 \
GRID_WFILE=GRIDfile{0}.dat\n".format(idx,molecule_features.features_def_list[idx]))
            elif i == 21:
                fw.write("metad{0}: METAD FILE=HILLS{0} ARG={1} PACE=10 \
SIGMA=0.3 DAMPFACTOR=100 TAU=1 TARGET=target{0}.dat \
GRID_MIN=0 GRID_MAX=2pi GRID_BIN=300 GRID_WSTRIDE=25000 \
GRID_WFILE=GRIDfile{0}.dat\n".format(idx,molecule_features.features_def_list[idx]))
            elif i == 22:
                fw.write("metad{0}: METAD FILE=HILLS{0} ARG={1} PACE=10 \
SIGMA=0.3 DAMPFACTOR=100 TAU=1 TARGET=target{0}.dat \
GRID_MIN=0 GRID_MAX=pi GRID_BIN=300 GRID_WSTRIDE=25000 \
GRID_WFILE=GRIDfile{0}.dat\n".format(idx,molecule_features.features_def_list[idx]))
            elif i == 31:
                fw.write("metad{0}: METAD FILE=HILLS{0} ARG={1} PACE=10 \
SIGMA=0.3 DAMPFACTOR=100 TAU=1 TARGET=target{0}.dat \
GRID_MIN=-pi GRID_MAX=pi GRID_BIN=300 GRID_WSTRIDE=25000 \
GRID_WFILE=GRIDfile{0}.dat\n".format(idx,molecule_features.features_def_list[idx]))



# used just to generate files from some distribution
def generate_targeted_MD_files(molecule_features,final_distribution,n_iteration):
    with open("generate_MD_distributions/dist{0}/plumed_restraint.dat".format(n_iteration),'w') as fw:

        # write molecule_features
        fw.write(molecule_features.write_plumed_features())
        fw.write('\n')

        # Now make metadynamics sim - here we are restraining the molecule!
        for idx,i in enumerate(molecule_features.features_def_list_idx):
            if i == 1:
                fw.write("metad{0}: METAD FILE=HILLS{0} ARG={1} PACE=10 \
SIGMA=0.3 DAMPFACTOR=100 TAU=1 TARGET=target{0}.dat \
GRID_MIN=-pi GRID_MAX=pi GRID_BIN=300 GRID_WSTRIDE=25000 \
GRID_WFILE=GRIDfile{0}.dat\n".format(idx,molecule_features.features_def_list[idx]))
            elif i == 21:
                fw.write("metad{0}: METAD FILE=HILLS{0} ARG={1} PACE=10 \
SIGMA=0.3 DAMPFACTOR=100 TAU=1 TARGET=target{0}.dat \
GRID_MIN=0 GRID_MAX=2pi GRID_BIN=300 GRID_WSTRIDE=25000 \
GRID_WFILE=GRIDfile{0}.dat\n".format(idx,molecule_features.features_def_list[idx]))
            elif i == 22:
                fw.write("metad{0}: METAD FILE=HILLS{0} ARG={1} PACE=10 \
SIGMA=0.3 DAMPFACTOR=100 TAU=1 TARGET=target{0}.dat \
GRID_MIN=0 GRID_MAX=pi GRID_BIN=300 GRID_WSTRIDE=25000 \
GRID_WFILE=GRIDfile{0}.dat\n".format(idx,molecule_features.features_def_list[idx]))
            elif i == 31:
                fw.write("metad{0}: METAD FILE=HILLS{0} ARG={1} PACE=10 \
SIGMA=0.3 DAMPFACTOR=100 TAU=1 TARGET=target{0}.dat \
GRID_MIN=-pi GRID_MAX=pi GRID_BIN=300 GRID_WSTRIDE=25000 \
GRID_WFILE=GRIDfile{0}.dat\n".format(idx,molecule_features.features_def_list[idx]))

    # Write the target distributions
    #for idx,i in enumerate(np.arange(all_data.sim.features[0])):
    for idx in np.arange(len(molecule_features.features_def_list_idx)):
        # Is it dihedral angle?
        if molecule_features.features_def_list_idx[idx]==1:
            x=final_distribution.nd_array[idx][0]
            yfes=final_distribution.nd_array[idx][4]
            ygrad=final_distribution.nd_array[idx][5]
            with open("generate_MD_distributions/dist{0}/target{1}.dat".format(n_iteration,idx),'w') as fw:
                fw.write("#! FIELDS {0} metad{1}.target der_{0}\n".format(molecule_features.features_def_list[idx],idx))
                fw.write("#! SET min_{0} -pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET max_{0} pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET nbins_{0}  {1}\n".format(molecule_features.features_def_list[idx],len(x[:-1])))
                fw.write("#! SET periodic_{0} true\n".format(molecule_features.features_def_list[idx]))
                for idx_x,xx in enumerate(x[:-1]):
                    fw.write("{0}  {1}  {2}\n".format(x[idx_x],yfes[idx_x],ygrad[idx_x]))

        # Is it puckering coordinate 6 member ring - phi?
        elif molecule_features.features_def_list_idx[idx]==21:
            x=final_distribution.nd_array[idx][0]
            yfes=final_distribution.nd_array[idx][4]
            ygrad=final_distribution.nd_array[idx][5]

            with open("generate_MD_distributions/dist{0}/target{1}.dat".format(n_iteration,idx),'w') as fw:
                fw.write("#! FIELDS {0} metad{1}.target der_{0}\n".format(molecule_features.features_def_list[idx],idx))
                fw.write("#! SET min_{0} 0\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET max_{0} 2pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET nbins_{0}  {1}\n".format(molecule_features.features_def_list[idx],len(x[:-1])))
                fw.write("#! SET periodic_{0} true\n".format(molecule_features.features_def_list[idx]))
                for idx_x,xx in enumerate(x[:-1]):
                    fw.write("{0}  {1}  {2}\n".format(x[idx_x],yfes[idx_x],ygrad[idx_x]))
        # Is it puckering coordinate 6 member ring - theta?
        elif molecule_features.features_def_list_idx[idx]==22:
            x=final_distribution.nd_array[idx][0]
            yfes=final_distribution.nd_array[idx][4]
            ygrad=final_distribution.nd_array[idx][5]

            with open("generate_MD_distributions/dist{0}/target{1}.dat".format(n_iteration,idx),'w') as fw:
                fw.write("#! FIELDS {0} metad{1}.target der_{0}\n".format(molecule_features.features_def_list[idx],idx))
                fw.write("#! SET min_{0} 0\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET max_{0} pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET nbins_{0}  {1}\n".format(molecule_features.features_def_list[idx],len(x[:-1])))
                fw.write("#! SET periodic_{0} false\n".format(molecule_features.features_def_list[idx]))
                for idx_x,xx in enumerate(x[:-1]):
                    fw.write("{0}  {1}  {2}\n".format(x[idx_x],yfes[idx_x],ygrad[idx_x]))
        # Is it puckering coordinate 5 member ring - phs
        elif molecule_features.features_def_list_idx[idx]==31:
            x=final_distribution.nd_array[idx][0]
            yfes=final_distribution.nd_array[idx][4]
            ygrad=final_distribution.nd_array[idx][5]

            with open("generate_MD_distributions/dist{0}/target{1}.dat".format(n_iteration,idx),'w') as fw:
                fw.write("#! FIELDS {0} metad{1}.target der_{0}\n".format(molecule_features.features_def_list[idx],idx))
                fw.write("#! SET min_{0} -pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET max_{0} pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET nbins_{0}  {1}\n".format(molecule_features.features_def_list[idx],len(x[:-1])))
                fw.write("#! SET periodic_{0} true\n".format(molecule_features.features_def_list[idx]))
                for idx_x,xx in enumerate(x[:-1]):
                    fw.write("{0}  {1}  {2}\n".format(x[idx_x],yfes[idx_x],ygrad[idx_x]))




def write_target_distribution(molecule_features,final_distribution,n_iteration):
    # Write the target distributions
    for idx in np.arange(len(molecule_features.features_def_list_idx)):
        # Is it dihedral angle?
        if molecule_features.features_def_list_idx[idx]==1:
            x=final_distribution.nd_array[idx][0]
            yfes=final_distribution.nd_array[idx][4]
            ygrad=final_distribution.nd_array[idx][5]
            with open("new_iteration_{0}/MD/target{1}.dat".format(n_iteration,idx),'w') as fw:
                fw.write("#! FIELDS {0} metad{1}.target der_{0}\n".format(molecule_features.features_def_list[idx],idx))
                fw.write("#! SET min_{0} -pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET max_{0} pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET nbins_{0}  {1}\n".format(molecule_features.features_def_list[idx],len(x[:-1])))
                fw.write("#! SET periodic_{0} true\n".format(molecule_features.features_def_list[idx]))
                for idx_x,xx in enumerate(x[:-1]):
                    fw.write("{0}  {1}  {2}\n".format(x[idx_x],yfes[idx_x],ygrad[idx_x]))

        # Is it puckering coordinate 6 member ring - phi?
        elif molecule_features.features_def_list_idx[idx]==21:
            x=final_distribution.nd_array[idx][0]
            yfes=final_distribution.nd_array[idx][4]
            ygrad=final_distribution.nd_array[idx][5]

            with open("new_iteration_{0}/MD/target{1}.dat".format(n_iteration,idx),'w') as fw:
                fw.write("#! FIELDS {0} metad{1}.target der_{0}\n".format(molecule_features.features_def_list[idx],idx))
                fw.write("#! SET min_{0} 0\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET max_{0} 2pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET nbins_{0}  {1}\n".format(molecule_features.features_def_list[idx],len(x[:-1])))
                fw.write("#! SET periodic_{0} true\n".format(molecule_features.features_def_list[idx]))
                for idx_x,xx in enumerate(x[:-1]):
                    fw.write("{0}  {1}  {2}\n".format(x[idx_x],yfes[idx_x],ygrad[idx_x]))
        # Is it puckering coordinate 6 member ring - theta?
        elif molecule_features.features_def_list_idx[idx]==22:
            x=final_distribution.nd_array[idx][0]
            yfes=final_distribution.nd_array[idx][4]
            ygrad=final_distribution.nd_array[idx][5]

            with open("new_iteration_{0}/MD/target{1}.dat".format(n_iteration,idx),'w') as fw:
                fw.write("#! FIELDS {0} metad{1}.target der_{0}\n".format(molecule_features.features_def_list[idx],idx))
                fw.write("#! SET min_{0} 0\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET max_{0} pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET nbins_{0}  {1}\n".format(molecule_features.features_def_list[idx],len(x[:-1])))
                fw.write("#! SET periodic_{0} false\n".format(molecule_features.features_def_list[idx]))
                for idx_x,xx in enumerate(x[:-1]):
                    fw.write("{0}  {1}  {2}\n".format(x[idx_x],yfes[idx_x],ygrad[idx_x]))
        # Is it puckering coordinate 5 member ring - phs
        elif molecule_features.features_def_list_idx[idx]==31:
            x=final_distribution.nd_array[idx][0]
            yfes=final_distribution.nd_array[idx][4]
            ygrad=final_distribution.nd_array[idx][5]

            with open("new_iteration_{0}/MD/target{1}.dat".format(n_iteration,idx),'w') as fw:
                fw.write("#! FIELDS {0} metad{1}.target der_{0}\n".format(molecule_features.features_def_list[idx],idx))
                fw.write("#! SET min_{0} -pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET max_{0} pi\n".format(molecule_features.features_def_list[idx]))
                fw.write("#! SET nbins_{0}  {1}\n".format(molecule_features.features_def_list[idx],len(x[:-1])))
                fw.write("#! SET periodic_{0} true\n".format(molecule_features.features_def_list[idx]))
                for idx_x,xx in enumerate(x[:-1]):
                    fw.write("{0}  {1}  {2}\n".format(x[idx_x],yfes[idx_x],ygrad[idx_x]))


def run_targeted_md_simulation(cluster_sim_nt,n_iteration):
    # Now copy md files here
    for file in glob.glob('needed_files/md_inp_files/*'):
        dest_dir="new_iteration_{0}/MD/.".format(n_iteration)
        run('cp -r {0} {1}'.format(file,dest_dir))
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

    run(f"{config.path['gmx']} grompp -f md_prod_itX.mdp -c structure_start_sim.gro -p {topfile} -o job.tpr -n {indexfile} -maxwarn 5")
    run(f"{config.path['mdrun_plumed']} -s job.tpr -v -deffnm job{cluster_sim_nt}  -nsteps 100000 -nt 1 -plumed  plumed_restraint.dat")

    os.chdir(c_path)

def generate_new_structures(molecule_features,n_iteration,number_of_new_structures,final_distribution,**kwargs):
    try:
        shutil.rmtree('new_iteration_{0}'.format(n_iteration))
    except FileNotFoundError:
        pass

    os.makedirs("new_iteration_{0}".format(n_iteration))
    os.makedirs("new_iteration_{0}/MD_trj".format(n_iteration))

    # make the simulation until there are no crashes (while statement)
    cluster_sim_nt = 0
    ns=int(number_of_new_structures/500*10)
    # Set minimal number of structures to 1*50 = 50
    if ns == 0:
        ns = 1
    while cluster_sim_nt < ns:
        try:
            shutil.rmtree("new_iteration_{0}/MD".format(n_iteration))
        except FileNotFoundError:
            pass
        os.makedirs("new_iteration_{0}/MD".format(n_iteration))

        # generate metadynamics plumed file
        generate_restrained_MD_plumed_file_targeted_MD(molecule_features,n_iteration,kwargs)
        # generate target distribution files
        write_target_distribution(molecule_features,final_distribution,n_iteration)
        # run the MD simulation
        run_targeted_md_simulation(cluster_sim_nt,n_iteration)

        # Check that it did not crash
        if glob.glob("new_iteration_{0}/MD/step*".format(n_iteration)) == []:
            run('cp new_iteration_{0}/MD/job{1}.xtc new_iteration_{0}/MD_trj/.'.format(n_iteration,cluster_sim_nt))
            run('cp new_iteration_{0}/MD/job{1}.gro new_iteration_{0}/MD_trj/structure_start_sim_prev.gro'.format(n_iteration,cluster_sim_nt))
            cluster_sim_nt += 1
        else:
            pass

#    # get rid of the first structure of all job*.xtc files - due to the restarts, these are the same
#    c_path=os.getcwd()
#    os.chdir('new_iteration_{0}/MD_trj'.format(n_iteration))
#    for i in glob.glob('job*xtc'):
#        run(f"printf \"0\n\"|{config.path['gmx']} trjconv -f {i} -s structure_start_sim_prev.gro -o job_tmp.xtc -b 1")
#        run("mv job_tmp.xtc {0}".format(i))
#    os.chdir(c_path)

    # cat them together
    c_path=os.getcwd()
    os.chdir('new_iteration_{0}/MD_trj'.format(n_iteration))
    run(f"{config.path['gmx']} trjcat -f job*.xtc -o job_cat.xtc -cat")
    grofile=glob.glob(c_path+'/needed_files/md_inp_files/*gro')[0]
    run('cp ../MD/job.tpr job.tpr')
    run(f"{config.path['gmx']} trjcat -f job*.xtc -o job_cat.xtc -cat")
    run('cp {0} structure.gro'.format(grofile))
    run(f"printf \"1\n 0\n\"|{config.path['gmx']} trjconv -f job_cat.xtc -s job.tpr -pbc mol -center -o job_cat_center.xtc")

    # copy tpr file
    run('cp ../MD/job.tpr job.tpr')

    os.chdir(c_path)




def generate_targeted_MD_distributions(dist_i,molecule_features,final_distribution):
    try:
        shutil.rmtree('generate_MD_distributions/dist{0}'.format(dist_i))
    except FileNotFoundError:
        pass

    os.makedirs('generate_MD_distributions/dist{0}'.format(dist_i))

    # generate target distribution files
    generate_targeted_MD_files(molecule_features,final_distribution,dist_i)


