#!/bin/python
import glob
import shutil
import numpy as np
import os
import math
from glob import glob
import time
import math as m
import itertools
import glob
import mdtraj
import sys

from mdshark.common import run, run_submit, logger

# for 3 bonds Jij
Jij_intercept,Jij_slope = [0,1]
# For 4 bond Jij
Jij_intercept3,Jij_slope3 = [0.0, 1]

################################################################
#vibrational data

width=7.5
temp=300
# minv=100
# maxv=2100
#
kb=1.38064852*10**-23
c=299792458
hc=6.62607004*10**-34
# So that spectrum is ~1 at 1800 cm-1
normalization_factor=1700
normalization_factor_ir=12*1e-5
normalization_factor_vcd=200*1e-5


################################################################
# Define all functions
################################################################






def average_methyl_amonia_groups(trjdir):
    equivalent=[]

    #global equivalent
    grofile=glob.glob(trjdir+'/f*/*gro')[0]
    t=mdtraj.load(grofile)

    mol_xyz=t.xyz[0][t.top.select("not (water or name NA or name CL)")]
    mol_name_0=[t.topology.atom(i).element.symbol for i in np.arange(len(mol_xyz))]
    mol_name_1=[t.topology.atom(i).name[0] for i in np.arange(len(mol_xyz))]
    mol_name=['' for i in np.arange(len(mol_name_0))]
    for idx,i in enumerate(zip(mol_name_0,mol_name_1)):
        if i[0] != 'VS':
            mol_name[idx]=str(i[0])
        else:
            mol_name[idx]=str(i[1])

    for idx_i,i in enumerate(mol_xyz):
        al=[]
        for idx_j,j in enumerate(mol_xyz):
            dist=(np.linalg.norm(i-j))
            if dist<0.12 and dist !=0 and mol_name[idx_j]=='H':
                al.append(idx_j+1)
        if len(al)==3 and mol_name[idx_i] == 'C':
            equivalent.append(al)
        if len(al)==3 and mol_name[idx_i] == 'N':
            equivalent.append(al)
    return equivalent



def get_nmr_shifts_data(wdir_a):
    equivalent=average_methyl_amonia_groups(wdir_a)
    ################# 1) Prepare and check whether there is everithing that is needed ###############
    fun_file_c_a=0
    atom_list=[]

    # get atom list
    for jf in next(os.walk(wdir_a))[1]:
        grofile=glob.glob(wdir_a+jf+'/*.gro')[0]
        if os.path.isfile(grofile):
            t=mdtraj.load(grofile)
            atoms=len(t.top.select('not (water or name NA or name CL)'))
            break
        else:
            continue

    atom_list=[[] for i in np.arange(atoms)]

    an_read=0
    an_c=0
    for jf in next(os.walk(wdir_a))[1]:
        logfile=wdir_a+jf+'/g_nmr_shifts.inp.log'
        if os.path.isfile(logfile):
            with open(wdir_a+jf+'/g_nmr_shifts.inp.log','r') as f_roa:
                for line in f_roa:
                    if 'Input orientation:' in line:
                        an_read=1
                    if an_read==1 and an_c <= 5:
                        an_c+=1
                    if an_read==1 and an_c > 5 and an_c <= atoms+5:
                        idx=int(line.split()[0])-1
                        ls=line.split()[1]
                        if ls=='1':
                            atom_list[idx]=[idx+1,'H']
                        elif ls=='6':
                            atom_list[idx]=[idx+1,'C']
                        elif ls=='7':
                            atom_list[idx]=[idx+1,'N']
                        elif ls=='8':
                            atom_list[idx]=[idx+1,'O']
                        elif ls=='15':
                            atom_list[idx]=[idx+1,'P']
                        elif ls=='16':
                            atom_list[idx]=[idx+1,'S']
                        else:
                            logger.critical('error-element not found')
                        an_c+=1
            break
        else:
            continue

    # Prepare array for shifts
    sc_C_av=[]
    sc_H_av=[]

    for a1 in range(atoms):
        if atom_list[a1][1] =='H':
            sc_H_av.append([float(0),int(atom_list[a1][0])])
        if atom_list[a1][1] =='C':
            sc_C_av.append([float(0),int(atom_list[a1][0])])
    sc_H_av=np.array(sc_H_av)
    sc_C_av=np.array(sc_C_av)

    ################# 2) Go through all directories and extract data ###############
    for jf in next(os.walk(wdir_a))[1]:
        #print(jf)
        fun_file_c_a+=1
        sc_H=np.zeros(atoms)
        sc_C=np.zeros(atoms)
        with open(wdir_a+jf+'/g_nmr_shifts.inp.log','r') as f_roa:
            for line in f_roa:
                if 'Isotropic' in line and 'Anisotropy' in line:
                    if line.split()[1] =='H' and int(line.split()[0]) <=atoms:
                        atom_idx_l=int(line.split()[0]) - 1
                        shift=(intercept_H-float(line.split()[4]))/-slope_H
                        sc_H[atom_idx_l] = shift
                    if line.split()[1] =='C' and int(line.split()[0]) <=atoms:
                        atom_idx_l=int(line.split()[0]) - 1
                        shift=(intercept_C-float(line.split()[4]))/-slope_C
                        sc_C[atom_idx_l] = shift
        sc_H = sc_H[sc_H != 0]
        sc_C = sc_C[sc_C != 0]

        ################# 2.5) Set chemical shifts of equivalent atoms to their average value ###############

        for eq_atoms in equivalent:
            sum_sc=0
            for eqa in eq_atoms:
                where=np.where(sc_H_av[:,1]==eqa)[0][0]
                sum_sc+=sc_H[where]
            for eqa in eq_atoms:
                where=np.where(sc_H_av[:,1]==eqa)[0][0]
                sc_H[where]=sum_sc/len(eq_atoms)

        ################# 4) Write down the individual files ###############
        with open(str(wdir_a)+str(jf)+'/sch.dat','w') as f_sch,\
            open(str(wdir_a)+str(jf)+'/scc.dat','w') as f_scc:

            # NMR data

            for i in range(len(sc_H)):
                f_sch.write('{0} {1} \n'.format(sc_H_av[i][1],sc_H[i]))
            for i in range(len(sc_C)):
                f_scc.write('{0} {1} \n'.format(sc_C_av[i][1],sc_C[i]))


        ################# 5) Wrte down the averages ###############
        for i in range(len(sc_H)):
            sc_H_av[i][0]+=sc_H[i]/1.0
        for i in range(len(sc_C)):
            sc_C_av[i][0]+=sc_C[i]/1.0


    sc_H_av[:,0]=sc_H_av[:,0]/1./fun_file_c_a
    sc_C_av[:,0]=sc_C_av[:,0]/1./fun_file_c_a


    with open(str(wdir_a)+'sch.dat','w') as f_sch,\
    open(str(wdir_a)+'scc.dat','w') as f_scc:

        # NMR data
        for i in range(len(sc_H_av)):
            f_sch.write('{0} {1} \n'.format(sc_H_av[i][1],sc_H_av[i][0]))
        for i in range(len(sc_C_av)):
            f_scc.write('{0} {1} \n'.format(sc_C_av[i][1],sc_C_av[i][0]))

    return None

###################################################

def get_nmr_spinspin_data(wdir_a,molecule_features):

    # Get equivalent atoms
    equivalent=average_methyl_amonia_groups(wdir_a)

    # Sort Jij atoms to 1-3, 1-4, and the rest Jij couplings
    atom_number_list = np.arange(len(molecule_features.atom_list))+1
    import itertools
    atom_combinations = list(itertools.combinations_with_replacement(atom_number_list,2))

    Jij_deff_list = np.zeros(len(atom_combinations))
    for idx,i in  enumerate(atom_combinations):
        tv = 0

        y1=molecule_features.bonds12[:,0] == i[0]
        y2=molecule_features.bonds12[:,1] == i[1]
        if (y1&y2).any() == True:
            tv = 3
        y1=molecule_features.bonds13[:,0] == i[0]
        y2=molecule_features.bonds13[:,1] == i[1]
        if (y1&y2).any() == True:
            tv = 4
        Jij_deff_list[idx] = tv




    for jf in next(os.walk(wdir_a))[1]:
        #print(jf)
        # get NAtoms
        with open(wdir_a+jf+'/g_nmr_spinspin.inp.log','r') as fi:
            for line in fi:
                if 'NAtoms=' in line:
                    atoms = int(line.split()[1])
                    break

        # Prepare things for Jij
        jj_lines_c=0
        linestp=[0]
        lines_p_o=[]
        for i in range(0,m.ceil(atoms/5)):
            n=atoms-5*i+1
            lines_p_o.append(atoms-5*i)
            jj_lines_c+=n
            linestp.append(jj_lines_c)
        linestp.pop(len(linestp)-1)

        lc_jj=0
        jj_lines=[]
        x=np.zeros(atoms)
        Jij,x=np.meshgrid(x,x)
        Jij_curr=np.zeros((int((atoms**2+atoms)/2),3))

        # Read Fermi Contacts
        fc_read=0
        with open(wdir_a+jf+'/g_nmr_spinspin.inp.log','r') as f_roa:
            for line in f_roa:
                if 'Fermi Contact (FC) contribution to J (Hz):' in line:
                    fc_read=1
                if fc_read==1 and lc_jj<=jj_lines_c:
                    if lc_jj>0:
                        jj_lines.append(line.split())
                    lc_jj+=1


        # Process the Fermi Contacts - sort it and find values that are nonzero
        for i in reversed(linestp):
            jj_lines.pop(i)
        arr = [[float((i.replace('D','E'))) for i in line] for line in jj_lines]

        # New triangle array
        narr=arr[0:atoms]
        for i in arr[atoms:len(arr)]:
            narr[int(i[0])-1]=narr[int(i[0])-1]+i[1:len(i)]

        # Construct Jij array where all coupling constants are.
        for l in narr:
            ai=int(l[0]-1)
            for ak in range(1,len(l)):
                Jij[ak-1][ai]=l[ak]

        # Filter it
        Jij_curr = []
        for a1 in range(atoms):
            for a2 in range(atoms):
                if a2>=a1 and a1 <= molecule_features.natoms and a2 <= molecule_features.natoms:
                    #Jij_scaled = (Jij[a1][a2] - Jij_intercept)/Jij_slope
                    #Jij_curr.append([Jij_scaled,a1+1,a2+1])
                    Jij_curr.append([Jij[a1][a2],a1+1,a2+1])

        Jij_curr = np.array(Jij_curr)

        # now scale according to deff list
        def scale_jij(par,val_to_scale):
            if par == 0:
                # do nothing if not 13 or 14 Jij
                a = 0
                b = 1
            elif par == 3:
                a = Jij_intercept3
                b = Jij_slope3
            elif par == 4:
                a = Jij_intercept
                b = Jij_slope
            return (val_to_scale - a)/b

        # Now map it
        for idx,i in enumerate(zip(Jij_deff_list,Jij_curr[:,0])):

            Jij_curr[idx][0]=scale_jij(i[0],i[1])

        # Average over equivalent atoms
        idx_sum=np.zeros(3,dtype=int)
        for i in [int(i) for i in list(set(Jij_curr[:,1])|set(Jij_curr[:,2]))]:
            for eq_group in equivalent:
                if i not in eq_group:
                    sum_group=0
                    for idx_eq,j in enumerate(eq_group):
                        pair=sorted([i,j])
                        #print(pair)
                        idx=np.where((Jij_curr[:,1]==pair[0])&(Jij_curr[:,2]==pair[1]))[0][0]
                        idx_sum[idx_eq]=int(idx)
                    av_value = np.average(Jij_curr[idx_sum][:,0])
                    for idx_change in idx_sum:
                        Jij_curr[idx_change][0]=av_value

        # # We ended up with huge array (n,3) - throw away the ones that are zero
        # jijn=[]
        # for i in Jij_curr:
        #     if np.abs(i[0]) >0:
        #         jijn.append([i[0],int(i[1]),int(i[2])])
        # jijn=np.array(jijn)
        # Jij_curr=jijn

        # Throw away water water_molecules if present

        # jijn=[]
        # for i in Jij_curr:
        #     if i[1]<=molecule_features.natoms or i[2]<=molecule_features.natoms:
        #         jijn.append([i[0],int(i[1]),int(i[2])])
        # jijn=np.array(jijn)
        # Jij_curr=jijn




        ################# 4) Write down the individual files ###############
        with open(str(wdir_a)+str(jf)+'/jij.dat','w') as f_jij, open(str(wdir_a)+str(jf)+'/jij_only.dat','w') as f_jij_only:

            for i in range(len(Jij_curr)):
                f_jij.write('{0:5f} {1:10d} {2:10d} \n'.format(Jij_curr[i][0],int(Jij_curr[i][1]),int(Jij_curr[i][2])))
            for i in range(len(Jij_curr)):
                f_jij_only.write('{0:5f}  \n'.format(Jij_curr[i][0]))

    # And finally average over all to get average values
    # prepare av array
    for jf in next(os.walk(wdir_a))[1]:
        jij_ar = np.loadtxt(wdir_a+jf+'/jij.dat')
        Jij_av_values = np.zeros((np.shape(jij_ar)[0]))
        Jij_av_atoms = jij_ar[:,1:]
        if np.shape(jij_ar)[0] != 0:
            break

    c=0
    for jf in next(os.walk(wdir_a))[1]:
        jij_ar = np.loadtxt(wdir_a+jf+'/jij.dat')
        Jij_av_values += jij_ar[:,0]
        c+=1

    Jij_av = np.hstack((np.array([[i] for i in Jij_av_values/c]),Jij_av_atoms))

    # Write it
    with open(str(wdir_a)+'jij.dat','w') as f_jij,\
    open(str(wdir_a)+'jij_only.dat','w') as f_jij_only:

        for i in range(len(Jij_av)):
            f_jij.write('{0:5f} {1:10d} {2:10d} \n'.format(Jij_av[i][0],int(Jij_av[i][1]),int(Jij_av[i][2])))
        for i in range(len(Jij_av)):
            f_jij_only.write('{0:5f}\n'.format(Jij_av[i][0]))

    return None

def get_vib_data(wdir_a,kwargs_dict):
    
    
    try:
        minv,maxv = kwargs_dict['raman_frequency_range']
    except:
        minv,maxv = [1,3500]
        
    fun_file_c_a=0
    freq_spectra_av=np.zeros([maxv-minv+1],dtype=float)
    raman_spectra_av=np.zeros([maxv-minv+1],dtype=float)
    roa_spectra_av=np.zeros([maxv-minv+1],dtype=float)
    ir_spectra_av=np.zeros([maxv-minv+1],dtype=float)
    vcd_spectra_av=np.zeros([maxv-minv+1],dtype=float)

    # Go through directories of alpha anomer
    for j in next(os.walk(wdir_a))[1]:

        # use shifting function?
        try:
            use_raman_shifting_function = kwargs_dict['use_raman_shifting_function']
        except:
            use_raman_shifting_function = True
            logger.warning("Use raman shifting function is not defined/ill defined, setting to True")
        if use_raman_shifting_function == True:
            with open(wdir_a+j+'/raman_scaling_function.dat','r') as f_sf:
                par_a,par_b,par_c,par_d = np.loadtxt(f_sf)
        else:
            par_a,par_b,par_c,par_d = [1,1,1,1]

        fun_freq_a=[]
        fun_raman_icpu_a=[]
        fun_roa_icpu_a=[]
        fun_ir_icpu_a=[]
        fun_vcd_icpu_a=[]

        # Open the log file and get frequencies, ramana intensities, and ROA intensities
        delete_file=0
        with open(wdir_a+j+'/g_final_ROA_calc.inp.log','r') as f_roa:
            for line in f_roa:
                try:
                    if 'Frequencies --' in line:
                        for i in line[15:71].split():
                            fun_freq_a.append(float(i))
                    elif 'Raman1 Fr= 1--' in line:
                        for i in line[15:71].split():
                            fun_raman_icpu_a.append(float(i))
                    elif 'ROA1   Fr= 1--' in line:
                        for i in line[15:71].split():
                            fun_roa_icpu_a.append(float(i))
                except ValueError:
                    delete_file=1

        if delete_file ==1:
            logger.warning('Wrong (****** problem) vib intensity, deleting {0}'.format(wdir_a+j))
            shutil.rmtree(wdir_a+j)
            continue


        # When checked that we have the values add one
        fun_file_c_a+=1


        fun_freq_a=np.array(fun_freq_a)
        fun_raman_icpu_a=np.array(fun_raman_icpu_a)
        fun_roa_icpu_a=np.array(fun_roa_icpu_a)

        with open(wdir_a+j+'/g_final_ROA_calc.dat','w') as f_w:
            for i in range(0,len(fun_freq_a)):
                f_w.write('{0:20s} {1:20s} {2:20s}  \n'.format(str(fun_freq_a[i]),str(fun_raman_icpu_a[i]),str(fun_roa_icpu_a[i])))

        q1=1/math.pi*(0.5*width)
        q2=(0.5*width)**2
        q3=-hc*100*c/(kb*temp)
        y1_raman_a=fun_raman_icpu_a*q1
        y1_roa_a=fun_roa_icpu_a*q1
        sf_array=(par_a*1/(1+np.exp((-fun_freq_a+par_d)/par_c))+par_b*(1-1/(1+np.exp((-fun_freq_a+par_d)/par_c))))
        y1_freq=fun_freq_a*sf_array

        freq_spectra=np.empty([maxv-minv+1],dtype=float)
        raman_spectra=np.empty([maxv-minv+1],dtype=float)
        roa_spectra=np.empty([maxv-minv+1],dtype=float)

        for k,freq_v in enumerate(range(minv,maxv+1)):
            freq_spectra[k]=freq_v
            suma_raman=np.sum(y1_raman_a/((freq_v-y1_freq)**2+q2))
            suma_raman1=normalization_factor*suma_raman/freq_v*(1/(1-math.exp(q3*freq_v)))
            raman_spectra[k]=suma_raman1

            suma_roa=np.sum(y1_roa_a/((freq_v-y1_freq)**2+q2))
            suma_roa1=normalization_factor*suma_roa/freq_v*(1/(1-math.exp(q3*freq_v)))
            roa_spectra[k]=suma_roa1

        with open(str(wdir_a)+str(j)+'/raman_icpu.dat','w') as f_raman,\
            open(str(wdir_a)+str(j)+'/roa_icpu.dat','w') as f_roa:

            # First Raman/ROA
            for i in range(0,len(freq_spectra)):
                f_raman.write('{0:10s} {1:10s} \n'.format(str(freq_spectra[i]),str(raman_spectra[i])))
                f_roa.write('{0:10s} {1:10s} \n'.format(str(freq_spectra[i]),str(roa_spectra[i])))

        freq_spectra_av=freq_spectra
        raman_spectra_av+=raman_spectra
        roa_spectra_av+=roa_spectra

    freq_spectra_av=freq_spectra_av
    raman_spectra_av=raman_spectra_av/fun_file_c_a
    roa_spectra_av=roa_spectra_av/fun_file_c_a

    with open(wdir_a+'raman_icpu.dat','w') as f_raman,\
    open(str(wdir_a)+'roa_icpu.dat','w') as f_roa:

        for i in range(0,len(freq_spectra_av)):
            f_raman.write('{0:10s} {1:10s} \n'.format(str(freq_spectra_av[i]),str(raman_spectra_av[i])))
            f_roa.write('{0:10s} {1:10s} \n'.format(str(freq_spectra_av[i]),str(roa_spectra_av[i])))

def what_to_calculate(file):
    calc_list=[]
    with open(file,'r') as fr:
        file_r=fr.readlines()
        for line in file_r:
            calc_list.append(int(line.split()[1]))
    return calc_list


def all_calculated(calculate_which_spectra,ipath):
    # Check that all have been calculated
    calculate_check=1
    if calculate_which_spectra[0] == 1:
        if os.path.isfile(ipath+'/g_final_ROA_calc.inp.log'):
            pass
        else:
            calculate_check=0
    if calculate_which_spectra[1] == 1:
        if os.path.isfile(ipath+'/g_nmr_shifts.inp.log'):
            pass
        else:
            calculate_check=0
    if calculate_which_spectra[2] == 1:
        if os.path.isfile(ipath+'/g_nmr_spinspin.inp.log'):
            pass
        else:
            calculate_check=0
    if calculate_check==0:
        logger.warning('Not all .log files found,deleting {0}'.format(ipath))
        shutil.rmtree(ipath)
    return calculate_check


def normal_termination(calculate_which_spectra,ipath):

    # Check that all have been calculated
    # Vib check
    normal_termination=1
    normal_termination_i=0
    normal_termination_NaN_check=0

    if calculate_which_spectra[0] == 1:
        with open(ipath+'/g_final_ROA_calc.inp.log','r') as f_roa:
            for line in f_roa:
                if 'NaN' in line:
                    normal_termination_NaN_check=1
                if 'Normal' in line:
                    normal_termination_i=1
        if normal_termination_i==1 and normal_termination_NaN_check==0:
            pass
        else:
            normal_termination=0

    # NMR shifts Check
    normal_termination_i=0
    if calculate_which_spectra[1] == 1:
        with open(ipath+'/g_nmr_shifts.inp.log','r') as f_roa:
            for line in f_roa:
                if 'Normal' in line:
                    normal_termination_i=1
        if normal_termination_i==1:
            pass
        else:
            normal_termination=0
    # NMR spinspin Check
    normal_termination_i=0
    if calculate_which_spectra[2] == 1:
        with open(ipath+'/g_nmr_spinspin.inp.log','r') as f_roa:
            for line in f_roa:
                if 'Normal' in line:
                    normal_termination_i=1
        if normal_termination_i==1:
            pass
        else:
            normal_termination=0
    # End of checks - delete when not converged
    if normal_termination == 0:
        logger.warning('deleting {0}'.format(ipath))
        shutil.rmtree(ipath)
    return normal_termination


def calculate_all_sim_data(path,molecule_features,**kwargs):
    if path == '':
        folders='new_iteration_[0-99]'
    else:
        folders=path
    for j in glob.glob(folders):
        fpath=j+'/input_files/'
        # Make some checks that eerything is fine
        input_files = glob.glob(j+'/input_files/f*')
        if not input_files:
            raise(FileNotFoundError(f"No input files found in {fpath}"))
        for ipath in input_files:
            # What spectra to calculate
            try:
                calculate_which_spectra=what_to_calculate(ipath+'/calculate.dat')
                # Check that all have been calculated, if not delete
                calculated_check=all_calculated(calculate_which_spectra,ipath)
                if calculated_check==0:
                    continue
                # Check that the calculations have normal terminations
                calculated_check=normal_termination(calculate_which_spectra,ipath)
                if calculated_check==0:
                    continue
            except FileNotFoundError:
                logger.warning(f'Have not found calculate.dat file, deleting {ipath}')
                shutil.rmtree(ipath)

        # Now we know that we have all data needed we go through all directories
        if calculate_which_spectra[0] == 1:
            logger.verbose('Extracting vibrational data.')
            get_vib_data(fpath,kwargs)
            pass
        if calculate_which_spectra[1] == 1:
            logger.verbose('Extracting NMR H/C data.')
            # laod H/C linear regression parameters
            global intercept_H,slope_H,intercept_C,slope_C
            try:
                intercept_H,slope_H = np.loadtxt(ipath+'/nmr_h_shifts_regression_par.dat')
                intercept_C,slope_C = np.loadtxt(ipath+'/nmr_c_shifts_regression_par.dat')
            except:
                logger.critical("NMR chemical shifts regression parameters have not been correctly defined!")
                logger.critical("Using default parameters - this is not correct!")
                intercept_H,slope_H = [30.97537,-0.94003] 
                intercept_C,slope_C = [184.32152,-1.09499]
            get_nmr_shifts_data(fpath)
            pass
        if calculate_which_spectra[2] == 1:
            logger.verbose('Extracting NMR Jij couplings.')
            get_nmr_spinspin_data(fpath,molecule_features)
            pass

#calculate_all_sim_data()
