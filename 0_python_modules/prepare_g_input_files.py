import numpy as np
import os
import mdtraj
import re
import glob
import shutil

import config

#########################################################################
################### QM INPUT FILES GENERATION ###########################
#########################################################################


# Define functions to prepare QM input file
class frame_features:
    """
Class for loading structure in gromacs format *.gro and extracting all unique dihedral angles or puckering coordinates.
Moreover it will write the initial plumed.dat file.
    """
    def __init__(self,frame,molecule_features):

        try:
            water_xyz = np.squeeze(frame.xyz)[molecule_features.water_idx]
        except:
            water_xyz = []

        mol_xyz = np.squeeze(frame.xyz)[molecule_features.mol_idx]
        water_idx = molecule_features.water_idx
        mol_idx = molecule_features.mol_idx

        self.mol_xyz = mol_xyz
        self.mol_idx = mol_idx
        self.water_xyz = water_xyz
        self.water_idx = water_idx

        self.molecule_features = molecule_features

        # not working now
#                 if qm_water == 'amide':
#                     if len(molecule_features.pg_amide_bond) != 0:
#                         mol_xyz_polar = np.squeeze(frame.xyz)[molecule_features.pg_amide_bond-1] # polar amide bond
#                     else:
#                         mol_xyz_polar=[]
#                 elif qm_water == 'polar':
#                     if len(molecule_features.polar_atoms) != 0:
#                         mol_xyz_polar = np.squeeze(frame.xyz)[molecule_features.polar_atoms-1] # polar all
#                     mol_xyz_polar=[]
#                 else:
#                     print('Choose properly possible QM layer of water moelcules - amide/polar?')
#                     break
#                 self.mol_xyz_polar=mol_xyz_polar

        water_dist_cutoff = 0.3
        # find water molecules
        water_around_idx = [] #Array where to save new waters
        for count, (water_atom_idx, water_atom_xyz) in enumerate(zip(molecule_features.water_idx, water_xyz),1):
            if count % 3 == 1 : #Working over three point water molecules
                i_water = [] # Array containing the water checked in the loop
                bSaveWater = False
            i_water.append(water_atom_idx)

            dist = np.linalg.norm(mol_xyz - water_atom_xyz, axis=1, keepdims=True)
            if np.any(dist < water_dist_cutoff):
                bSaveWater = True
            if count % 3 == 0 and bSaveWater == True:
                water_around_idx.append(i_water)
        water_around_idx = np.sort(np.array(water_around_idx).flatten())
        self.water_around_idx = water_around_idx

        # Make connectivity list
        # Count explicit atoms
        explicit_atoms=molecule_features.natoms+len(water_around_idx)
        # Make connectivity lists
        connectivity_list_xyz=np.zeros((explicit_atoms,3))
        connectivity_list_name=np.zeros(explicit_atoms,dtype='str')
        average_coords=list(sum(frame.xyz[0][molecule_features.mol_idx])/len(frame.xyz[0][molecule_features.mol_idx]))
        self.average_coords=average_coords
        atom_counter=0
        for i in list(molecule_features.mol_idx):
            atom_name=frame.topology.atom(int(i)).element.symbol
            atom_xyz=10*(frame.xyz[0][int(i)]-average_coords)
            connectivity_list_xyz[atom_counter]=atom_xyz
            connectivity_list_name[atom_counter]=atom_name
            atom_counter+=1
        for i in list(water_around_idx):
            atom_name=frame.topology.atom(int(i)).element.symbol
            atom_xyz=10*(frame.xyz[0][int(i)]-average_coords)
            connectivity_list_xyz[atom_counter]=atom_xyz
            connectivity_list_name[atom_counter]=atom_name
            atom_counter+=1
        # Make connectivity string
        connectivity_string=''
        xyz_sel=connectivity_list_xyz[:]
        atoms_sel=connectivity_list_name[:]
        for idxi,i in enumerate(xyz_sel):
            dist = np.linalg.norm(xyz_sel - i, axis=1, keepdims=True)
            bonded=np.where((dist<1.65) &  (dist>0))[0]
            var="{0}".format(str(idxi+1),end='rr')
            for j in bonded:
                if j>idxi:

                    if atoms_sel[idxi] == 'H' or atoms_sel[j] == 'H':
                        dd=np.linalg.norm(i - xyz_sel[j])
                        if dd <1.3:
                            var+=' {0} 1.0'.format(str(j+1),end='xx')
                    else:
                        var+=' {0} 1.0'.format(str(j+1),end='yy')
            var+='\n'
            connectivity_string+=var
        self.connectivity_string = connectivity_string
        # Connectivity string done



        ###################
        # Strategy is to find all water molecules that are close to poalr atoms - then substract those
        # from already established list of water molecules that are close
        # Find water atoms which are wround polar atoms (charge >0.25)
#        polar_atoms_cutoff = 0.2
#        polar_atoms_selection_array = np.abs(np.array(molecule_features.individual_charges))>polar_atoms_cutoff
#        mol_polar_atoms_idx = mol_idx[polar_atoms_selection_array]
#        mol_polar_atoms_xyz = np.squeeze(frame.xyz)[mol_polar_atoms_idx]
#
#        # find most polar aatom and least polar atom - assign a first QM/MM water molecules
 #       # This is to ensure that there is at least 1MM and 1 QM wm so that tha approach always works
 #       least_polar_atom_find = np.argmin(np.abs(np.array(molecule_features.individual_charges)))
 #       least_polar_atom_idx = mol_idx[least_polar_atom_find]
 #       least_polar_atom_xyz = np.squeeze(frame.xyz)[least_polar_atom_idx]
 #       most_polar_atom_find = np.argmax(np.abs(np.array(molecule_features.individual_charges)))
 #       most_polar_atom_idx = mol_idx[most_polar_atom_find]
  #      most_polar_atom_xyz = np.squeeze(frame.xyz)[most_polar_atom_idx]
  #
  #
  #
  #      water_dist_cutoff = 0.3
  #      most_polar_atom_water_found=False
 #       least_polar_atom_water_found=False
 #       # find water molecules
 #       water_around_idx = [] #Array where to save new waters
 #       most_polar_atom_water_idx = []
 #       least_polar_atom_water_idx = []
 #       for count, (water_atom_idx, water_atom_xyz) in enumerate(zip(molecule_features.water_idx, water_xyz),1):
 #           if count % 3 == 1 : #Working over three point water molecules
 #               i_water = [] # Array containing the water checked in the loop
 #               bSaveWater = False
 #           i_water.append(water_atom_idx)
 #           dist = np.linalg.norm(mol_polar_atoms_xyz - water_atom_xyz, axis=1, keepdims=True)
 #           if np.any(dist < water_dist_cutoff):
 #               bSaveWater = True
 #           if count % 3 == 0 and bSaveWater == True:
 #               water_around_idx.append(i_water)
 #
 #           # find the most/least polar atom water
 #           if most_polar_atom_water_found == False:
 #               dist_polar_atom = np.linalg.norm(most_polar_atom_xyz - water_atom_xyz)
 #               if dist_polar_atom < 0.45:
 #                   bSaveWater = True
 #               if count % 3 == 0 and bSaveWater == True:
 #                   most_polar_atom_water_idx.append(i_water)
 #                   most_polar_atom_water_found = True
 #           if least_polar_atom_water_found == False:
 #               dist_polar_atom = np.linalg.norm(least_polar_atom_xyz - water_atom_xyz)
 #               if dist_polar_atom < 0.45:
 #                   bSaveWater = True
 #               if count % 3 == 0 and bSaveWater == True:
 #                   least_polar_atom_water_idx.append(i_water)
 #                   least_polar_atom_water_found = True
 #
 #       water_around_idx = np.array(water_around_idx).flatten()
 #
 #       # Now make the lists
 #       # original list is self.water_around_idx
 #       # so lets make similar list, but state what water is QM and what water is MM
 #       self.water_around_polar_atoms_idx = np.array(list((set(water_around_idx)|set(most_polar_atom_water_idx[0]))-set(least_polar_atom_water_idx[0])))
 #       self.water_around_polar_atoms_rest_idx = np.array(list(set(self.water_around_idx)-set(self.water_around_polar_atoms_idx)))
 #       qmw = [[0,i] for i in self.water_around_polar_atoms_idx]
 #       mmw = [[1,i] for i in self.water_around_polar_atoms_rest_idx]
 #       nl = sorted(np.concatenate((qmw,mmw)),key=lambda x:x[1])
 #       self.water_around_idx_polar_sorted = nl
 #
 #
 #
 #
 #


    ###################



    # This class has following variables:

    # self.connectivity_string - connectivity string of all molecule+water molecules(0.3 ang cutoff)
    # self.average_coords - average coordinates of the molecule
    # self.mol_xyz - molecule xyz coordinates
    # self.mol_idx - molecule indices for gromacs
    # self.water_xyz - all water molecules xyz coordinates
    # self.water_idx - all water molecules indices
    # self.molecule_features.atom_list - all unique atoms [H,C,N,O,S etc...]
    # self.water_around_polar_atoms_idx - all water atom indices that are close to atoms with |charge|<cutoff (0.2)
    # self.water_around_polar_atoms_rest_idx - all water atom sindices that are close(0.3nm) but not in the  self.water_around_polar_atoms_idx list


######################################
### General functions - START ###
######################################

# Set oniom vdw parameters
def set_oniom_vdw_parameters(**kwargs):
    
    oniom_vdw_parameters_C = kwargs['oniom_vdw_parameters_C']
    
    # glycam DNA RNA parameters
    if oniom_vdw_parameters_C == 'RNA_DNA_1':
        print("Setting ONIOM vdw parameters to: RND/DNA glycam")
        oniom_vdw_parameters="""VDW OW 1.7682 0.1521
VDW HW 0.0000 0.0000
VDW cc 1.9080 0.0860
VDW na 1.8240 0.1700
VDW ca 1.9080 0.0860
VDW c3 1.9080 0.1094
VDW os 1.6837 0.1700
VDW oh 1.7210 0.2104
VDW h2 1.2870 0.0157
VDW h1 1.3870 0.0157
VDW ho 0.0000 0.0000
VDW nd 1.8240 0.1700
VDW h5 1.3590 0.0150
VDW nb 1.8240 0.1700
VDW nh 1.8240 0.1700
VDW hn 0.6000 0.0157
VDW nc 1.8240 0.1700
VDW c 1.9080 0.0860
VDW n 1.8240 0.1700
VDW o 1.6612 0.2100
VDW ha 1.4590 0.0150
VDW cd 1.9080 0.0860
VDW h4 1.4090 0.0150"""

    # Glycam parameters
    elif oniom_vdw_parameters_C == 'glycam_saccharides':
        print("Setting ONIOM vdw parameters to: glycam saccharides")
        oniom_vdw_parameters="""VDW OW 1.7682 0.1521
VDW HW 0.0000 0.0000
VDW C  1.9080  0.0860
VDW Cg 1.9080  0.1094
VDW H1 1.3870  0.0157
VDW H2 1.2870  0.0157
VDW Ho 0.2000  0.0300
VDW H  0.6000  0.0157
VDW Hc 1.4870  0.0157
VDW Ng 1.8240  0.1699
VDW O  1.6612  0.2099
VDW Oh 1.7210  0.2104
VDW Os 1.6837  0.1670
VDW O2 1.6612  0.2100
VDW c3 1.9080  0.0860"""
    # Protein charmm parameters
    elif oniom_vdw_parameters_C == 'proteins_1':
        print("Setting ONIOM vdw parameters to: proteins_1")
        oniom_vdw_parameters="""VDW OW 1.7682 0.1521
VDW HW 0.0000 0.0000
VDW H 0.2245 0.0460
VDW HA 1.3200 0.0220
VDW HA1 1.3400 0.0450
VDW HA2 1.3400 0.0340
VDW HA3 1.3400 0.0240
VDW CT2A 2.0100 0.0560
VDW C 2.0000 0.1100
VDW N 1.8500 0.2000
VDW NC2 1.8500 0.2000
VDW O 1.7000 0.1200
VDW HS 0.4500 0.1000
VDW HC 0.2245 0.0460
VDW HB1 1.3200 0.0220
VDW HB2 1.3400 0.0280
VDW NH1 1.8500 0.2000
VDW NH2 1.8500 0.2000
VDW NH3 1.8500 0.2000
VDW CT 2.2750 0.0200
VDW CT1 2.0000 0.0320
VDW CT2 2.0100 0.0560
VDW CT3 2.0400 0.0780
VDW CC 2.0000 0.0700
VDW CY 1.9900 0.0730
VDW NY 1.8500 0.2000
VDW CPT 1.8600 0.0990
VDW CAI 1.9900 0.0730
VDW HP 1.3582 0.0300
VDW S 2.0000 0.4500
VDW CPH1 1.8000 0.0500
VDW CPH2 1.8000 0.0500
VDW NR1 1.8500 0.2000
VDW NR2 1.8500 0.2000
VDW NR3 1.8500 0.2000
VDW HR1 0.9000 0.0460
VDW HR2 0.7000 0.0460
VDW HR3 1.4680 0.0078
VDW HP 1.3582 0.0300
VDW CP1 2.2750 0.0200
VDW CP2 2.1750 0.0550
VDW CP3 2.1750 0.0550
VDW NP 1.8500 0.2000
VDW OH1 1.7700 0.1521
VDW OC 1.7000 0.1200"""

    # protien GAFF generated form antechamber
    elif oniom_vdw_parameters_C == 'proteins_2':
        print("Setting ONIOM vdw parameters to: proteins_2")
        oniom_vdw_parameters="""VDW OW 1.7682 0.1521
VDW HW 0.0000 0.0000
VDW ca 1.9080 0.0860
VDW c3 1.9080 0.1094
VDW n4 1.8240 0.1700
VDW n 1.8240 0.1700
VDW c 1.9080 0.0860
VDW o 1.6612 0.2100
VDW hn 0.6000 0.0157
VDW ha 1.4590 0.0150
VDW hc 1.4870 0.0157
VDW hx 1.1000 0.0157
VDW h1 1.3870 0.0157"""

    # automatic charmm picked
    elif oniom_vdw_parameters_C == 'charmm-cgenff':
        try:
            print("Setting ONIOM vdw parameters to charmm-cgenff")

            read_ff_nb = False
            unique_atoms = []
            with open(kwargs['mol_file_path'],'r') as fr:
                for line in fr:
                    if line == '[ atoms ]\n':
                        read_ff_nb = True
                    if line == '[ bonds ]\n':
                        read_ff_nb = False
                    if read_ff_nb == True:
                        try:
                            int(line.split()[0])
                            unique_atoms.append(line.split()[1])
                        except (ValueError,IndexError):
                            pass
            unique_atoms = np.unique(unique_atoms)

            par_arr = []
            for i in unique_atoms:
                with open(kwargs['ffnonbonded_file_path'],'r') as fr:
                    for line in fr:
                        if (len(line.split()) == 7) and (i == line.split()[0]):
                            par_arr.append([i,line.split()[-2],line.split()[-1]])

            oniom_vdw_parameters = 'VDW OW 1.76820 0.15210\nVDW HW 0.0000 0.0000\n'
            for i in par_arr:
                sig = float(i[1])*10/2*(2**(1/6))
                eps = float(i[2])/4.184
                oniom_vdw_parameters += "{0} {1} {2:.5f} {3:.5f}\n".format('VDW',i[0],sig,eps)       
        except Exception as err:
            print(err)
            print("Error when setting ONIOM vdw parameters to charmm-cgenff ")
            print("Check mol.itp/ffnonbonded file paths")

        # delete the last new line
        oniom_vdw_parameters = oniom_vdw_parameters[:-2]
        
    else:
        print("Error: did not found VDW parameters")
    return  oniom_vdw_parameters

# python scripts to handle geometries and prepare final input files from sample
def make_python_scripts(n_iteration,filename_counter,molecule_features,write_folder):

    # Python file which prepare oniom calculations (Raman/ROA)
    g_inp_file_preparation_oniom="""
from sys import argv
sc,f1_goptfile,f2_gfinac_new,f3_gfinalc_sample = argv


atom_names=[]
oniom_levels=[]
xyz=[]

with open(f1_goptfile) as f_in:
    for i,line in enumerate(f_in):
        try:
            if line.split()[0]=='NAtoms=' and line.split()[2]=='NActive=':
                n_atoms=int(line.split()[1])
        except IndexError:
            pass
        try:
            if line.split()[3]=='Coordinates':
                xyz_last_line=i
        except IndexError:
            pass

with open(f1_goptfile) as f_in:
    lines=f_in.readlines()
    for i in range(xyz_last_line+3,xyz_last_line+n_atoms+3):
        xyz.append(lines[i].split()[3:6])


with open(f3_gfinalc_sample,"r+") as f_in:
    for num,line in enumerate(f_in):
        list=line.split()
        if list==[str({0}),str({1}),str({0}),str({1}),str({0}),str({1})]:
            xyz_start=num

with open(f2_gfinac_new,"w+") as f_o:
    with open(f3_gfinalc_sample,"r+") as f_sample:
        f_sample_l=f_sample.readlines()
        for i in range(0,xyz_start+1):
            f_o.write(f_sample_l[i])
        for i in range(0,n_atoms):
            atom_name=f_sample_l[xyz_start+i+1].split()[0]
            oniom_level=f_sample_l[xyz_start+i+1].split()[5]
            opt_level=f_sample_l[xyz_start+i+1].split()[1]
            f_o.write(" %10s %5s %10s %10s %10s %2s\\n"%(atom_name,opt_level,xyz[i][0],xyz[i][1],xyz[i][2],oniom_level))
        for i in range(xyz_start+n_atoms+1,len(f_sample_l)):
            f_o.write(f_sample_l[i])

""".format(molecule_features.charge,molecule_features.multiplicity)


    # Python file which prepare pcm calculations 
    g_inp_file_preparation_pcm="""
from sys import argv
sc,f1_goptfile,f2_gfinac_new,f3_gfinalc_sample = argv


atom_names=[]
xyz=[]

with open(f1_goptfile) as f_in:
    for i,line in enumerate(f_in):
        try:
            if line.split()[0]=='NAtoms=' and line.split()[2]=='NActive=':
                n_atoms=int(line.split()[1])
        except IndexError:
            pass
        try:
            if line.split()[3]=='Coordinates':
                xyz_last_line=i
        except IndexError:
            pass

with open(f1_goptfile) as f_in:
    lines=f_in.readlines()
    for i in range(xyz_last_line+3,xyz_last_line+n_atoms+3):
        xyz.append(lines[i].split()[3:6])

c=-2
c_read=0

with open(f3_gfinalc_sample,"r+") as f_in:
    for num,line in enumerate(f_in):
        list=line.split()
        if list==[str({0}),str({1})]:
            xyz_start=num
            c_read=1
        if c_read==1:
            c+=1
        if c_read==1 and list==[]:
            c_read=0


with open(f2_gfinac_new,"w+") as f_o:
    with open(f3_gfinalc_sample,"r+") as f_sample:
        f_sample_l=f_sample.readlines()
        for i in range(0,xyz_start+1):
            f_o.write(f_sample_l[i])
        for i in range(0,c):
            atom_name=f_sample_l[xyz_start+i+1].split()[0]
            f_o.write(" %3s  %10s %10s %10s\\n"%(atom_name,xyz[i][0],xyz[i][1],xyz[i][2]))
        for i in range(xyz_start+c+1,len(f_sample_l)):
            f_o.write(f_sample_l[i])

""".format(molecule_features.charge,molecule_features.multiplicity)


    g_structure_extraction="""
from sys import argv
sc,f1_goptfile,f2_gfinac_new,f3_gfinalc_sample = argv


xyz=[]

with open(f1_goptfile) as f_in:
    for i,line in enumerate(f_in):
        try:
            if line.split()[0]=='NAtoms=' and line.split()[2]=='NActive=':
                n_atoms=int(line.split()[1])
        except IndexError:
            pass
        try:
            if line.split()[3]=='Coordinates':
                xyz_last_line=i
        except IndexError:
            pass

with open(f1_goptfile) as f_in:
    lines=f_in.readlines()
    for i in range(xyz_last_line+3,xyz_last_line+n_atoms+3):
        xyz.append([float(i)/10 for i in lines[i].split()[3:6]])

with open(f2_gfinac_new,"w+") as f_o, open(f3_gfinalc_sample,"r+") as f_sample:
    f_sample_l=f_sample.readlines()
    for i in f_sample_l[0:2]:
        f_o.write(i)
    for idx,i in enumerate(f_sample_l[2:2+n_atoms]):
        str1=str(i[:20])
        str2w='{0:8.3f}{1:8.3f}{2:8.3f} \\n'.format(xyz[idx][0],xyz[idx][1],xyz[idx][2])
        str_w=str1+str2w
        f_o.write(str_w)
    f_o.write(str(f_sample_l[-1]))
"""
    with open("{2}/f{0}_{1:05d}/g_inp_file_preparation_oniom.py".format(n_iteration,filename_counter,write_folder),"w+") as frame_g_inp_file_preparation:
        frame_g_inp_file_preparation.write(g_inp_file_preparation_oniom)
    with open("{2}/f{0}_{1:05d}/g_inp_file_preparation_pcm.py".format(n_iteration,filename_counter,write_folder),"w+") as frame_g_inp_file_preparation:
        frame_g_inp_file_preparation.write(g_inp_file_preparation_pcm)
    with open("{2}/f{0}_{1:05d}/g_structure_gro_extraction.py".format(n_iteration,filename_counter,write_folder),"w+") as frame_g_inp_file_preparation:
        frame_g_inp_file_preparation.write(g_structure_extraction)

######################################
### Option 1 - low quality - START ###
######################################

## optimization ##
def make_gaussian_inputs_o1_opt(filename_counter,frame, frame_features_i, oniom_vdw_parameters,itp_file,n_iteration,qm_m_nproc,write_folder):
    # First save geometry
    groname = "{2}/f{0}_{1:05d}/structure_initial.gro".format(n_iteration,filename_counter,write_folder)
    memory,nproc=qm_m_nproc
    frame.atom_slice(np.concatenate([frame_features_i.mol_idx, frame_features_i.water_around_idx])).save_gro(groname)

    g_header0="""%chk=g_opt.chk
%mem={0}GB
%nproc={1}
#oniom(pm7:amber=softfirst)=embedcharge nosymm  opt=(nomicro,maxcycles=11) geom=connectivity Scrf=(cpcm,solvent=water)

optimization - 0 step

{2} {3} {2} {3} {2} {3}
""".format(memory,nproc,frame_features_i.molecule_features.charge,frame_features_i.molecule_features.multiplicity)
    g_header1="""%chk=g_opt.chk
%mem={0}GB
%nproc={1}
#oniom(B3LYP/6-31G*:amber=softfirst)=embedcharge nosymm  opt=(nomicro,maxcycles=6) geom=AllCheck Scrf=(cpcm,solvent=water) guess=check


""".format(memory,nproc,frame_features_i.molecule_features.charge,frame_features_i.molecule_features.multiplicity)
    g_header2="""%chk=g_opt.chk
%mem={0}GB
%nproc={1}
#oniom(B3LYP/6-311++G**:amber=softfirst)=embedcharge nosymm  opt=(nomicro,maxcycles=6) geom=AllCheck Scrf=(cpcm,solvent=water) guess=check


""".format(memory,nproc,frame_features_i.molecule_features.charge,frame_features_i.molecule_features.multiplicity)
    g_opt0="{2}/f{0}_{1:05d}/g_opt0.inp".format(n_iteration,filename_counter,write_folder)
    g_opt1="{2}/f{0}_{1:05d}/g_opt1.inp".format(n_iteration,filename_counter,write_folder)
    g_opt2="{2}/f{0}_{1:05d}/g_opt2.inp".format(n_iteration,filename_counter,write_folder)
    with open(g_opt0,'w') as fw0, open(g_opt1,'w') as fw1, open(g_opt2,'w') as fw2:
        fw0.write(g_header0)
        fw1.write(g_header1)
        fw2.write(g_header2)
        # Molecule coordinates
        for i in list(frame_features_i.mol_idx):
            res=str(frame.topology.atom(i).residue)[0:3]
            atom=str(frame.topology.atom(i).name)
            index=str((frame.topology.atom(i).index)+1)
            atom_name=frame.topology.atom(int(i)).element.symbol
            atom_xyz=10*(frame.xyz[0][int(i)]-frame_features_i.average_coords)

            with open(itp_file,"r") as itpf:
                lines=itpf.readlines()
                for line in lines:
                    if re.search(" "+res+" ",line) and re.search(" "+atom+" ",line) and re.search(" "+index+" ",line):
                        atom_type=line.split()[1]
                        atom_charge=line.split()[6]
            fw0.write(" {0}-{1}-{2}  0  {3:0.3f} {4:0.3f} {5:0.3f} H\n".format(atom_name,atom_type,atom_charge,\
                                                    float(atom_xyz[0]),float(atom_xyz[1]),float(atom_xyz[2])))
        # Water coordinates
        for i in list(frame_features_i.water_around_idx):
            atom_name=frame.topology.atom(int(i)).element.symbol
            atom_xyz=10*(frame.xyz[0][int(i)]-frame_features_i.average_coords)
            if atom_name == 'O':
                atom_type='OW'
                atom_charge='-0.834'
            else:
                atom_type='HW'
                atom_charge='0.417'
            fw0.write(" {0}-{1}-{2}  0  {3:0.3f} {4:0.3f} {5:0.3f} L\n".format(atom_name,atom_type,atom_charge,\
                                                    float(atom_xyz[0]),float(atom_xyz[1]),float(atom_xyz[2])))
        # Write connectivity
        fw0.write('\n')
        fw0.write(frame_features_i.connectivity_string)
        fw0.write('\n')
        fw0.write(oniom_vdw_parameters)
        fw0.write('\n\n')
## optimization ##

## Raman/ROA ##
# 6-31G* basis set for frequencies, rDPS basis set for intensities
def make_gaussian_inputs_o1_raman_roa(filename_counter,frame, frame_features_i, oniom_vdw_parameters,itp_file,n_iteration,qm_m_nproc,write_folder):
    # Raman g input files
    # Initial guess for the scaling function
    scaling_function = '0.925 1.0 100 1525'

    memory,nproc=qm_m_nproc
    gh0_ram="""%chk=raman_calc_chk.chk
%mem={0}GB
%nproc={1}
#oniom(B3LYP/6-31G*:amber=softfirst)=embedcharge freq nosymm  geom=(connectivity,nomicro) Scrf=(cpcm,solvent=water)

raman/roa calculation

{2} {3} {2} {3} {2} {3}
""".format(memory,nproc,frame_features_i.molecule_features.charge,frame_features_i.molecule_features.multiplicity)
    gh1_ram="""%chk=raman_calc_chk.chk
%mem={0}GB
%nproc={1}
#oniom(B3LYP/gen:amber=softfirst)=embedcharge polar=roa nosymm geom=(connectivity,nomicro)  Scrf=(cpcm,solvent=water)

raman/roa calculation

{2} {3} {2} {3} {2} {3}
""".format(memory,nproc,frame_features_i.molecule_features.charge,frame_features_i.molecule_features.multiplicity)
    # Raman basis set  - rDPS basis set (3-21G + p(0.2) orbital on hydrogen)
    # for Raman intensities
    # https://www.basissetexchange.org/

    raman_bs_H="""S   2   1.00
0.5447178000D+01       0.1562849787D+00
0.8245472400D+00       0.9046908767D+00
S   1   1.00
0.1831915800D+00       1.0000000
S   1   1.00
0.3600000000D-01       0.1000000000D+01
P   1   1.00
0.2000000000E+00       1.0000000
****
"""
    raman_bs_C="""S   3   1.00
0.1722560000D+03       0.6176690738D-01
0.2591090000D+02       0.3587940429D+00
0.5533350000D+01       0.7007130837D+00
SP   2   1.00
0.3664980000D+01      -0.3958951621D+00       0.2364599466D+00
0.7705450000D+00       0.1215834356D+01       0.8606188057D+00
SP   1   1.00
0.1958570000D+00       0.1000000000D+01       0.1000000000D+01
SP   1   1.00
0.4380000000D-01       0.1000000000D+01       0.1000000000D+01
****
"""
    raman_bs_N="""S   3   1.00
0.2427660000D+03       0.5986570051D-01
0.3648510000D+02       0.3529550030D+00
0.7814490000D+01       0.7065130060D+00
SP   2   1.00
0.5425220000D+01      -0.4133000774D+00       0.2379720162D+00
0.1149150000D+01       0.1224417267D+01       0.8589530586D+00
SP   1   1.00
0.2832050000D+00       0.1000000000D+01       0.1000000000D+01
SP   1   1.00
0.6390000000D-01       0.1000000000D+01       0.1000000000D+01
****
"""
    raman_bs_O="""S   3   1.00
0.3220370000D+03       0.5923939339D-01
0.4843080000D+02       0.3514999608D+00
0.1042060000D+02       0.7076579210D+00
SP   2   1.00
0.7402940000D+01      -0.4044535832D+00       0.2445861070D+00
0.1576200000D+01       0.1221561761D+01       0.8539553735D+00
SP   1   1.00
0.3736840000D+00       0.1000000000D+01       0.1000000000D+01
SP   1   1.00
0.8450000000D-01       0.1000000000D+01       0.1000000000D+01
****
"""
    # Prepare Raman basis set for final intesity calculations
    raman_bs=''
    for atom_j in set(frame_features_i.molecule_features.atom_list):
        atom_i_list=[i+1 for i,val in enumerate(frame_features_i.molecule_features.atom_list) if val==atom_j]
        if atom_j == 'H':
            for k in atom_i_list:
                raman_bs+='{0} '.format(k)
            raman_bs+='0\n'
            raman_bs+=raman_bs_H
        elif atom_j == 'C':
            for k in atom_i_list:
                raman_bs+='{0} '.format(k)
            raman_bs+='0\n'
            raman_bs+=raman_bs_C
        elif atom_j == 'N':
            for k in atom_i_list:
                raman_bs+='{0} '.format(k)
            raman_bs+='0\n'
            raman_bs+=raman_bs_N
        elif atom_j == 'O':
            for k in atom_i_list:
                raman_bs+='{0} '.format(k)
            raman_bs+='0\n'
            raman_bs+=raman_bs_O
        else:
            raman_bs+='{0} 0\n6-311++G**\n****\n'.format(atom_j)

    file_out0="{2}/f{0}_{1:05d}/g_raman_sample0.inp".format(n_iteration,filename_counter,write_folder)
    file_out1="{2}/f{0}_{1:05d}/g_raman_sample1.inp".format(n_iteration,filename_counter,write_folder)
    with open(file_out0,'w') as fw0, open(file_out1,'w') as fw1:
        fw0.write(gh0_ram)
        fw1.write(gh1_ram)
        # Molecule coordinates
        for i in list(frame_features_i.mol_idx):
            res=str(frame.topology.atom(i).residue)[0:3]
            atom=str(frame.topology.atom(i).name)
            index=str((frame.topology.atom(i).index)+1)
            atom_name=frame.topology.atom(int(i)).element.symbol
            atom_xyz=10*(frame.xyz[0][int(i)]-frame_features_i.average_coords)

            with open(itp_file,"r") as itpf:
                lines=itpf.readlines()
                for line in lines:
                    if re.search(" "+res+" ",line) and re.search(" "+atom+" ",line) and re.search(" "+index+" ",line):
                        atom_type=line.split()[1]
                        atom_charge=line.split()[6]
            fw0.write(" {0}-{1}-{2}  0  {3:0.3f} {4:0.3f} {5:0.3f} H\n".format(atom_name,atom_type,atom_charge,\
                                                    float(atom_xyz[0]),float(atom_xyz[1]),float(atom_xyz[2])))
            fw1.write(" {0}-{1}-{2}  0  {3:0.3f} {4:0.3f} {5:0.3f} H\n".format(atom_name,atom_type,atom_charge,\
                                                    float(atom_xyz[0]),float(atom_xyz[1]),float(atom_xyz[2])))
        # Water coordinates
        for i in list(frame_features_i.water_around_idx):
            atom_name=frame.topology.atom(int(i)).element.symbol
            atom_xyz=10*(frame.xyz[0][int(i)]-frame_features_i.average_coords)
            if atom_name == 'O':
                atom_type='OW'
                atom_charge='-0.834'
            else:
                atom_type='HW'
                atom_charge='0.417'
            fw0.write(" {0}-{1}-{2}  0  {3:0.3f} {4:0.3f} {5:0.3f} L\n".format(atom_name,atom_type,atom_charge,\
                                                    float(atom_xyz[0]),float(atom_xyz[1]),float(atom_xyz[2])))
            fw1.write(" {0}-{1}-{2}  0  {3:0.3f} {4:0.3f} {5:0.3f} L\n".format(atom_name,atom_type,atom_charge,\
                                                    float(atom_xyz[0]),float(atom_xyz[1]),float(atom_xyz[2])))
        # Write connectivity
        fw0.write('\n')
        fw1.write('\n')
        fw0.write(frame_features_i.connectivity_string)
        fw1.write(frame_features_i.connectivity_string)
        fw0.write('\n')
        fw1.write('\n')
        fw0.write(oniom_vdw_parameters)
        fw1.write(oniom_vdw_parameters)
        fw0.write('\n\n')
        fw1.write('\n\n')
        fw1.write('532nm')
        fw1.write('\n\n')
        fw1.write(raman_bs)
        fw1.write('\n\n')

    sfw="{2}/f{0}_{1:05d}/raman_scaling_function.dat".format(n_iteration,filename_counter,write_folder)
    with open(sfw,'w') as fw:
        fw.write(scaling_function)
## Raman/ROA ##

## NMR chemical shifts ##
# calculate chemical shieldings using PCM solvations scheme with pcS-2 basis set
def make_gaussian_inputs_o1_nmr_shifts(filename_counter,frame, frame_features_i, oniom_vdw_parameters,itp_file,n_iteration,qm_m_nproc,write_folder,**kwargs):

    # set solvent
    try:
        pcm_solvent = kwargs['set_solvent']
    except:
        pcm_solvent = 'water'

    # Regression parameters for this method (intercept/slope)
    H_reg_par = '31.3296715831336 -1.0084162991609904'
    C_reg_par = '189.7154878886343 -1.1530205889444989'

    sfw="{2}/f{0}_{1:05d}/nmr_h_shifts_regression_par.dat".format(n_iteration,filename_counter,write_folder)
    with open(sfw,'w') as fw:
        fw.write(H_reg_par)
    sfw="{2}/f{0}_{1:05d}/nmr_c_shifts_regression_par.dat".format(n_iteration,filename_counter,write_folder)
    with open(sfw,'w') as fw:
        fw.write(C_reg_par)

    memory,nproc = qm_m_nproc
    g_header0="""%mem={0}GB
%nproc={1}
#LC-BLYP/gen nmr=csgt nosymm  Scrf=(cpcm,solvent={4})

nmr chemical shifts

{2} {3} 
""".format(memory,nproc,frame_features_i.molecule_features.charge,frame_features_i.molecule_features.multiplicity,pcm_solvent)

    # Basis set for chemical shifts
    #pcS-2
    s_bs_H="""H     0
S   4   1.00
      0.754226D+02           0.406941D-02
      0.113499D+02           0.322324D-01
      0.259926D+01           0.150651D+00
      0.735130D+00           0.500000D+00
S   1   1.00
      0.231669D+00           0.100000D+01
S   1   1.00
      0.741474D-01           0.100000D+01
P   2   1.00
      0.960000D+01           0.181540D-01
      0.160000D+01           0.500000D+00
P   1   1.00
      0.450000D+00           0.100000D+01
D   1   1.00
      0.125000D+01           0.100000D+01
****
"""
    s_bs_C="""C     0
S   7   1.00
      0.785710D+04           0.642223D-03
      0.117865D+04           0.493018D-02
      0.268325D+03           0.255246D-01
      0.759483D+02           0.969662D-01
      0.245586D+02           0.278279D+00
      0.862118D+01           0.500000D+00
      0.312784D+01           0.369563D+00
S   7   1.00
      0.117865D+04           0.220880D-04
      0.268325D+03          -0.202720D-03
      0.759483D+02          -0.889512D-03
      0.245586D+02          -0.199457D-01
      0.862118D+01          -0.690043D-01
      0.312784D+01          -0.230090D+00
      0.822020D+00           0.500000D+00
S   1   1.00
      0.330170D+00           0.100000D+01
S   1   1.00
      0.114628D+00           0.100000D+01
P   4   1.00
      0.219536D+03           0.971271D-03
      0.337748D+02           0.202494D-01
      0.767659D+01           0.138773D+00
      0.223567D+01           0.500000D+00
P   1   1.00
      0.764466D+00           0.100000D+01
P   1   1.00
      0.262325D+00           0.100000D+01
P   1   1.00
      0.846377D-01           0.100000D+01
D   1   1.00
      0.140000D+01           0.100000D+01
D   1   1.00
      0.450000D+00           0.100000D+01
F   1   1.00
      0.950000D+00           0.100000D+01
****
"""
    s_bs_N="""N     0
S   7   1.00
      0.111014D+05           0.618632D-03
      0.166525D+04           0.475787D-02
      0.379083D+03           0.246049D-01
      0.107300D+03           0.942768D-01
      0.347472D+02           0.272580D+00
      0.122494D+02           0.500000D+00
      0.447434D+01           0.387367D+00
S   7   1.00
      0.166525D+04           0.224875D-04
      0.379083D+03          -0.213811D-03
      0.107300D+03          -0.102034D-02
      0.347472D+02          -0.213431D-01
      0.122494D+02          -0.791865D-01
      0.447434D+01          -0.251876D+00
      0.124037D+01           0.500000D+00
S   1   1.00
      0.487426D+00           0.100000D+01
S   1   1.00
      0.164842D+00           0.100000D+01
P   4   1.00
      0.311572D+03           0.950298D-03
      0.479342D+02           0.195715D-01
      0.109994D+02           0.135528D+00
      0.324724D+01           0.500000D+00
P   1   1.00
      0.111138D+01           0.100000D+01
P   1   1.00
      0.379881D+00           0.100000D+01
P   1   1.00
      0.120678D+00           0.100000D+01
D   1   1.00
      0.180000D+01           0.100000D+01
D   1   1.00
      0.550000D+00           0.100000D+01
F   1   1.00
      0.102000D+01           0.100000D+01
****
"""
    s_bs_O="""O     0
S   7   1.00
      0.147824D+05           0.607282D-03
      0.221733D+04           0.467273D-02
      0.504741D+03           0.241672D-01
      0.142873D+03           0.929122D-01
      0.463005D+02           0.270386D+00
      0.163373D+02           0.500000D+00
      0.598281D+01           0.392630D+00
S   7   1.00
      0.221733D+04           0.904080D-05
      0.504741D+03          -0.170992D-03
      0.142873D+03          -0.136132D-02
      0.463005D+02          -0.192022D-01
      0.163373D+02          -0.817923D-01
      0.598281D+01          -0.233115D+00
      0.167180D+01           0.500000D+00
S   1   1.00
      0.646621D+00           0.100000D+01
S   1   1.00
      0.216687D+00           0.100000D+01
P   4   1.00
      0.392755D+03           0.983308D-03
      0.604239D+02           0.199046D-01
      0.139351D+02           0.137896D+00
      0.415313D+01           0.500000D+00
P   1   1.00
      0.141579D+01           0.100000D+01
P   1   1.00
      0.475491D+00           0.100000D+01
P   1   1.00
      0.145292D+00           0.100000D+01
D   1   1.00
      0.220000D+01           0.100000D+01
D   1   1.00
      0.650000D+00           0.100000D+01
F   1   1.00
      0.110000D+01           0.100000D+01
****
"""
    # Prepare NMR basis set
    gen_bs=''
    for atom_j in set(frame_features_i.molecule_features.atom_list):
        atom_i_list=[i+1 for i,val in enumerate(frame_features_i.molecule_features.atom_list) if val==atom_j]
        if atom_j == 'H':
            gen_bs+=s_bs_H
        elif atom_j == 'C':
            gen_bs+=s_bs_C
        elif atom_j == 'N':
            gen_bs+=s_bs_N
        elif atom_j == 'O':
            gen_bs+=s_bs_O
        else:
            gen_bs+='{0} 0\n6-311++G**\n****\n'.format(atom_j)

    file_out="{2}/f{0}_{1:05d}/g_nmr_shifts_sample.inp".format(n_iteration,filename_counter,write_folder)
    with open(file_out,'w') as fw0:
        fw0.write(g_header0)
        # Molecule coordinates
        for i in list(frame_features_i.mol_idx):
            res=str(frame.topology.atom(i).residue)[0:3]
            atom=str(frame.topology.atom(i).name)
            index=str((frame.topology.atom(i).index)+1)
            atom_name=frame.topology.atom(int(i)).element.symbol
            atom_xyz=10*(frame.xyz[0][int(i)]-frame_features_i.average_coords)

            with open(itp_file,"r") as itpf:
                lines=itpf.readlines()
                for line in lines:
                    if re.search(" "+res+" ",line) and re.search(" "+atom+" ",line) and re.search(" "+index+" ",line):
                        atom_type=line.split()[1]
                        atom_charge=line.split()[6]
            fw0.write(" {0}  0  {3:0.3f} {4:0.3f} {5:0.3f} \n".format(atom_name,atom_type,atom_charge,\
                                                    float(atom_xyz[0]),float(atom_xyz[1]),float(atom_xyz[2])))
        fw0.write('\n')
        fw0.write(gen_bs)
        fw0.write('\n')
        fw0.write('\n')

## NMR chemical shifts ##

## NMR spin-spin couplings ##
# calculate spin-spin couplings using pcm solvation scheme + pcJ-1 basis set
def make_gaussian_inputs_o1_nmr_spinspin_couplings(filename_counter,frame,frame_features_i,n_iteration,qm_m_nproc,write_folder,**kwargs):

    # set solvent
    try:
        pcm_solvent = kwargs['set_solvent']
    except:
        pcm_solvent = 'water'


    memory,nproc=qm_m_nproc
    g_header="""%mem={0}GB
%nproc={1}
#mPW1PW91/gen nmr=(FCOnly,ReadAtoms) nosymm int=nobasistransform scrf=(cpcm,solvent={4})

spinspin calculation

{2} {3}
""".format(memory,nproc,frame_features_i.molecule_features.charge,frame_features_i.molecule_features.multiplicity,pcm_solvent)
    # Basis set for spin-spin coupling constants
    #pcJ-1
    Jij_bs_H="""H     0
S   2   1.00
0.382870E+04           0.110942E-01
0.153148E+03           0.500000E+00
S   2   1.00
0.122519E+02           0.704473E-01
0.186872E+01           0.500000E+00
S   1   1.00
0.418210E+00           1.0000000
S   1   1.00
0.106100E+00           1.0000000
P   1   1.00
0.9000000000E+01       1.0000000
P   1   1.00
0.1000000000E+01       1.0000000
****
"""
    Jij_bs_C="""C     0
S   4   1.00
0.391451E+06           0.811630E-04
0.156580E+05           0.365259E-02
0.125264E+04           0.641773E-01
0.188566E+03           0.500000E+00
S   2   1.00
0.428391E+02           0.197197E+00
0.118181E+02           0.500000E+00
S   1   1.00
0.355674E+01           1.0000000
S   1   1.00
0.542575E+00           1.0000000
S   1   1.00
0.160585E+00           1.0000000
P   2   1.00
0.594271E+02           0.341178E-01
0.914263E+01           0.500000E+00
P   1   1.00
0.192985E+01           1.0000000
P   1   1.00
0.525223E+00           1.0000000
P   1   1.00
0.136083E+00           1.0000000
D   1   1.00
0.1600000000E+02       1.0000000
D   1   1.00
0.8000000000E+00       1.0000000
****
"""
    Jij_bs_N="""N     0
S   4   1.00
0.544422E+06           0.810423E-04
0.217769E+05           0.364660E-02
0.174215E+04           0.640976E-01
0.262213E+03           0.500000E+00
S   2   1.00
0.595848E+02           0.195519E+00
0.164888E+02           0.500000E+00
S   1   1.00
0.499450E+01           1.0000000
S   1   1.00
0.786729E+00           1.0000000
S   1   1.00
0.228369E+00           1.0000000
P   2   1.00
0.853554E+02           0.334788E-01
0.131316E+02           0.500000E+00
P   1   1.00
0.281081E+01           1.0000000
P   1   1.00
0.763512E+00           1.0000000
P   1   1.00
0.195601E+00           1.0000000
D   1   1.00
0.1800000000E+02       1.0000000
D   1   1.00
0.9000000000E+00       1.0000000
****
"""
    Jij_bs_O="""O     0
S   4   1.00
0.720859E+06           0.809643E-04
0.288343E+05           0.364263E-02
0.230675E+04           0.640470E-01
0.347150E+03           0.500000E+00
S   2   1.00
0.788896E+02           0.194581E+00
0.218763E+02           0.500000E+00
S   1   1.00
0.666456E+01           1.0000000
S   1   1.00
0.106692E+01           1.0000000
S   1   1.00
0.307004E+00           1.0000000
P   2   1.00
0.110642E+03           0.332809E-01
0.170219E+02           0.500000E+00
P   1   1.00
0.368382E+01           1.0000000
P   1   1.00
0.992342E+00           1.0000000
P   1   1.00
0.244868E+00           1.0000000
D   1   1.00
0.2000000000E+02       1.0000000
D   1   1.00
0.1000000000E+01       1.0000000
****
"""
    # Prepare Jij spinspin NMR basis set
    gen_bs=''
    for atom_j in set(frame_features_i.molecule_features.atom_list):
        atom_i_list=[i+1 for i,val in enumerate(frame_features_i.molecule_features.atom_list) if val==atom_j]
        if atom_j == 'H':
            gen_bs+=Jij_bs_H
        elif atom_j == 'C':
            gen_bs+=Jij_bs_C
        elif atom_j == 'N':
            gen_bs+=Jij_bs_N
        elif atom_j == 'O':
            gen_bs+=Jij_bs_O
        else:
            gen_bs+='{0} 0\n6-311++G**\n****\n'.format(atom_j)





    try:
        set_jij_atom_couplings = kwargs['set_jij_atom_couplings']
        Jij_atom_select = 'atoms=' + set_jij_atom_couplings
    except:

        Jij_atom_select=''
        al1=set(frame_features_i.molecule_features.atom_list)
        Jij_atom_select+='atoms=H'
        for i in al1:
            if i=='C':
                Jij_atom_select+=',C'
            elif i=='N':
                Jij_atom_select+=',N'
            else:
                pass

    file_out0="{2}/f{0}_{1:05d}/g_nmr_spinspin_sample.inp".format(n_iteration,filename_counter,write_folder)
    with open(file_out0,'w') as fw:
        fw.write(g_header)
        # Molecule coordinates
        for i in list(frame_features_i.mol_idx):
            atom_name=frame.topology.atom(int(i)).element.symbol
            atom_xyz=10*(frame.xyz[0][int(i)]-frame_features_i.average_coords)
            fw.write(" {0} 0 {1:0.3f} {2:0.3f} {3:0.3f} \n".format(atom_name,\
                                                    float(atom_xyz[0]),float(atom_xyz[1]),float(atom_xyz[2])))

        fw.write('\n')
        fw.write(gen_bs)
        fw.write('\n')
        fw.write(Jij_atom_select)
        fw.write('\n\n')

# dqs submit script
def make_gaussian_inputs_o1_make_dqs_aurum(filename_counter,w_calculate,n_iteration,write_folder):

    if w_calculate[0] != 1:
        c_vib_v='#'
    else:
        c_vib_v=''
    if w_calculate[1] != 1:
        c_shifts_v='#'
    else:
        c_shifts_v=''
    if w_calculate[2] != 1:
        c_spinspin_v='#'
    else:
        c_spinspin_v=''

    final_g_dqs_file=f"""#!/bin/bash
{config.cluster_bash_header['gaussian_o1']}

# optimization
$GAUSS_EXEDIR/g16 <g_opt0.inp >g_opt0.inp.log
$GAUSS_EXEDIR/g16 <g_opt1.inp >g_opt1.inp.log
$GAUSS_EXEDIR/g16 <g_opt2.inp >g_opt2.inp.log
rm g_opt.chk

# get final structure
python g_structure_gro_extraction.py g_opt2.inp.log structure_final.gro structure_initial.gro

# shifts
python g_inp_file_preparation_pcm.py g_opt2.inp.log g_nmr_shifts.inp g_nmr_shifts_sample.inp
{c_shifts_v}$GAUSS_EXEDIR/g16 <g_nmr_shifts.inp >g_nmr_shifts.inp.log

# Raman/ROA
python g_inp_file_preparation_oniom.py g_opt2.inp.log g_final_ROA_calc0.inp g_raman_sample0.inp
{c_vib_v}$GAUSS_EXEDIR/g16 <g_final_ROA_calc0.inp >g_final_ROA_calc0.inp.log
python g_inp_file_preparation_oniom.py g_opt2.inp.log g_final_ROA_calc.inp g_raman_sample1.inp
{c_vib_v}$GAUSS_EXEDIR/g16 <g_final_ROA_calc.inp >g_final_ROA_calc.inp.log
rm raman_calc_chk.chk

# Jij couplings
python g_inp_file_preparation_pcm.py g_opt2.inp.log g_nmr_spinspin.inp g_nmr_spinspin_sample.inp
{c_spinspin_v}$GAUSS_EXEDIR/g16 <g_nmr_spinspin.inp >g_nmr_spinspin.inp.log
#

"""

    with open("{2}/f{0}_{1:05d}/qsub_ROA_calc.inp.dqs".format(n_iteration,filename_counter,write_folder),"w+") as frame_qsub_ROA_calc_dqs:
        frame_qsub_ROA_calc_dqs.write(final_g_dqs_file)

####################################
### Option 1 - low quality - END ###
####################################

#######################################
### Option 2 - high quality - START ###
#######################################

## optimization ##
# same as in low quality o1 - function make_gaussian_inputs_o1_opt

## Raman/ROA ##
# 6-311++G** basis set for frequencies/intensities
def make_gaussian_inputs_o2_raman_roa(filename_counter,frame, frame_features_i, oniom_vdw_parameters,itp_file,n_iteration,qm_m_nproc,write_folder):
    # Raman g input files
    # Initial guess for the scaling function
    scaling_function = '0.98 1.0 15 1210'

    memory,nproc=qm_m_nproc
    gh0_ram="""%chk=raman_calc_chk.chk
%mem={0}GB
%nproc={1}
#oniom(B3LYP/6-311++G**:amber=softfirst)=embedcharge freq=roa nosymm geom=(connectivity,nomicro) Scrf=(cpcm,solvent=water)

raman/roa calculation

{2} {3} {2} {3} {2} {3}
""".format(memory,nproc,frame_features_i.molecule_features.charge,frame_features_i.molecule_features.multiplicity)

    file_out0="{2}/f{0}_{1:05d}/g_raman_sample0.inp".format(n_iteration,filename_counter,write_folder)
    with open(file_out0,'w') as fw0:
        fw0.write(gh0_ram)
        # Molecule coordinates
        for i in list(frame_features_i.mol_idx):
            res=str(frame.topology.atom(i).residue)[0:3]
            atom=str(frame.topology.atom(i).name)
            index=str((frame.topology.atom(i).index)+1)
            atom_name=frame.topology.atom(int(i)).element.symbol
            atom_xyz=10*(frame.xyz[0][int(i)]-frame_features_i.average_coords)

            with open(itp_file,"r") as itpf:
                lines=itpf.readlines()
                for line in lines:
                    if re.search(" "+res+" ",line) and re.search(" "+atom+" ",line) and re.search(" "+index+" ",line):
                        atom_type=line.split()[1]
                        atom_charge=line.split()[6]
            fw0.write(" {0}-{1}-{2}  0  {3:0.3f} {4:0.3f} {5:0.3f} H\n".format(atom_name,atom_type,atom_charge,\
                                                    float(atom_xyz[0]),float(atom_xyz[1]),float(atom_xyz[2])))
        # Water coordinates
        for i in list(frame_features_i.water_around_idx):
            atom_name=frame.topology.atom(int(i)).element.symbol
            atom_xyz=10*(frame.xyz[0][int(i)]-frame_features_i.average_coords)
            if atom_name == 'O':
                atom_type='OW'
                atom_charge='-0.834'
            else:
                atom_type='HW'
                atom_charge='0.417'
            fw0.write(" {0}-{1}-{2}  0  {3:0.3f} {4:0.3f} {5:0.3f} L\n".format(atom_name,atom_type,atom_charge,\
                                                    float(atom_xyz[0]),float(atom_xyz[1]),float(atom_xyz[2])))
        # Write connectivity
        fw0.write('\n')
        fw0.write(frame_features_i.connectivity_string)
        fw0.write('\n')
        fw0.write(oniom_vdw_parameters)
        fw0.write('\n\n')
        fw0.write('532nm')
        fw0.write('\n\n')

    sfw="{2}/f{0}_{1:05d}/raman_scaling_function.dat".format(n_iteration,filename_counter,write_folder)
    with open(sfw,'w') as fw:
        fw.write(scaling_function)
## Raman/ROA ##

## NMR chemical shifts ##
# same as in low quality o1 - function make_gaussian_inputs_o1_nmr_shifts

## NMR spin-spin couplings ##
# same as in low quality o1 - function make_gaussian_inputs_o1_nmr_spinspin_couplings 

# dqs submit script
def make_gaussian_inputs_o2_make_dqs_aurum(filename_counter,w_calculate,n_iteration,write_folder):

    if w_calculate[0] != 1:
        c_vib_v='#'
    else:
        c_vib_v=''
    if w_calculate[1] != 1:
        c_shifts_v='#'
    else:
        c_shifts_v=''
    if w_calculate[2] != 1:
        c_spinspin_v='#'
    else:
        c_spinspin_v=''

    final_g_dqs_file=f"""#!/bin/bash
{config.cluster_bash_header['gaussian_o2']}

# optimization
$GAUSS_EXEDIR/g16 <g_opt0.inp >g_opt0.inp.log
$GAUSS_EXEDIR/g16 <g_opt1.inp >g_opt1.inp.log
$GAUSS_EXEDIR/g16 <g_opt2.inp >g_opt2.inp.log
rm g_opt.chk

# get final structure
python g_structure_gro_extraction.py g_opt2.inp.log structure_final.gro structure_initial.gro

# shifts
python g_inp_file_preparation_pcm.py g_opt2.inp.log g_nmr_shifts.inp g_nmr_shifts_sample.inp
{c_shifts_v}$GAUSS_EXEDIR/g16 <g_nmr_shifts.inp >g_nmr_shifts.inp.log

# Raman/ROA
python g_inp_file_preparation_oniom.py g_opt2.inp.log g_final_ROA_calc.inp g_raman_sample0.inp
{c_vib_v}$GAUSS_EXEDIR/g16 <g_final_ROA_calc.inp >g_final_ROA_calc.inp.log
rm raman_calc_chk.chk

# Jij couplings
python g_inp_file_preparation_pcm.py g_opt2.inp.log g_nmr_spinspin.inp g_nmr_spinspin_sample.inp
{c_spinspin_v}$GAUSS_EXEDIR/g16 <g_nmr_spinspin.inp >g_nmr_spinspin.inp.log
#

"""

    with open("{2}/f{0}_{1:05d}/qsub_ROA_calc.inp.dqs".format(n_iteration,filename_counter,write_folder),"w+") as frame_qsub_ROA_calc_dqs:
        frame_qsub_ROA_calc_dqs.write(final_g_dqs_file)

#####################################
### Option 2 - high quality - END ###
#####################################

###########################################
### Option 3 - very low quality - START ###
###########################################

## optimization - only PCM ##
## only b3lyp/6-31g* 10 steps
def make_gaussian_inputs_o3_opt(filename_counter,frame, frame_features_i, oniom_vdw_parameters,itp_file,n_iteration,qm_m_nproc,write_folder,**kwargs):

    # set solvent
    try:
        pcm_solvent = kwargs['set_solvent']
    except:
        pcm_solvent = 'water'

    # First save geometry
    groname = "{2}/f{0}_{1:05d}/structure_initial.gro".format(n_iteration,filename_counter,write_folder)
    memory,nproc=qm_m_nproc
    frame.atom_slice(frame_features_i.mol_idx).save_gro(groname)

    g_header0="""%chk=g_opt.chk
%mem={0}GB
%nproc={1}
#B3LYP/6-31g* nosymm opt=(maxcycles=11) geom=connectivity Scrf=(cpcm,solvent={4})

optimization - 0 step

{2} {3} 
""".format(memory,nproc,frame_features_i.molecule_features.charge,frame_features_i.molecule_features.multiplicity,pcm_solvent)
    g_opt0="{2}/f{0}_{1:05d}/g_opt0.inp".format(n_iteration,filename_counter,write_folder)
    with open(g_opt0,'w') as fw0:
        fw0.write(g_header0)
        # Molecule coordinates
        for i in list(frame_features_i.mol_idx):
            res=str(frame.topology.atom(i).residue)[0:3]
            atom=str(frame.topology.atom(i).name)
            index=str((frame.topology.atom(i).index)+1)
            atom_name=frame.topology.atom(int(i)).element.symbol
            atom_xyz=10*(frame.xyz[0][int(i)]-frame_features_i.average_coords)

            with open(itp_file,"r") as itpf:
                lines=itpf.readlines()
                for line in lines:
                    if re.search(" "+res+" ",line) and re.search(" "+atom+" ",line) and re.search(" "+index+" ",line):
                        atom_type=line.split()[1]
                        atom_charge=line.split()[6]
            fw0.write(" {0}   {3:0.3f} {4:0.3f} {5:0.3f} \n".format(atom_name,atom_type,atom_charge,\
                                                    float(atom_xyz[0]),float(atom_xyz[1]),float(atom_xyz[2])))
        # Write connectivity
        fw0.write('\n\n')
## optimization ##

# dqs submit script
def make_gaussian_inputs_o3_make_dqs_aurum(filename_counter,w_calculate,n_iteration,write_folder):

    if w_calculate[0] != 1:
        c_vib_v='#'
    else:
        c_vib_v=''
    if w_calculate[1] != 1:
        c_shifts_v='#'
    else:
        c_shifts_v=''
    if w_calculate[2] != 1:
        c_spinspin_v='#'
    else:
        c_spinspin_v=''

    final_g_dqs_file=f"""#!/bin/bash
{config.cluster_bash_header['gaussian_o3']}

# optimization
$GAUSS_EXEDIR/g16 <g_opt0.inp >g_opt0.inp.log
rm g_opt.chk

# get final structure
python g_structure_gro_extraction.py g_opt0.inp.log structure_final.gro structure_initial.gro

# shifts
{c_shifts_v}python g_inp_file_preparation_pcm.py g_opt0.inp.log g_nmr_shifts.inp g_nmr_shifts_sample.inp
{c_shifts_v}$GAUSS_EXEDIR/g16 <g_nmr_shifts.inp >g_nmr_shifts.inp.log

# Raman/ROA
{c_vib_v}python g_inp_file_preparation_oniom.py g_opt0.inp.log g_final_ROA_calc.inp g_raman_sample0.inp
{c_vib_v}$GAUSS_EXEDIR/g16 <g_final_ROA_calc.inp >g_final_ROA_calc.inp.log
rm raman_calc_chk.chk

# Jij couplings
{c_spinspin_v}python g_inp_file_preparation_pcm.py g_opt0.inp.log g_nmr_spinspin.inp g_nmr_spinspin_sample.inp
{c_spinspin_v}$GAUSS_EXEDIR/g16 <g_nmr_spinspin.inp >g_nmr_spinspin.inp.log
#

"""


    with open("{2}/f{0}_{1:05d}/qsub_ROA_calc.inp.dqs".format(n_iteration,filename_counter,write_folder),"w+") as frame_qsub_ROA_calc_dqs:
        frame_qsub_ROA_calc_dqs.write(final_g_dqs_file)

###########################################
### Option 3 - very low quality - END #####
###########################################


def prepare_qm_input_files(molecule_features,n_iteration,w_calculate,qm_m_nproc,**prepare_qm_input_files_kwargs):
    
    # set method
    try:
        method = prepare_qm_input_files_kwargs['method']
    except:
        print("Problem with setting qm generate input files. Defaulting to m1_raman_low.")
        method = 'm1_raman_low'
        
    # set oniom vdw parameters
    oniom_vdw_parameters = set_oniom_vdw_parameters(**prepare_qm_input_files_kwargs)
    
    # what to calculate
    c_vib,c_shifts,c_spinspin = w_calculate
    
    # make initial folders if needed
    os.makedirs("figures",exist_ok=True)
    os.makedirs("tmp_files",exist_ok=True)

    xtc_file='new_iteration_{0}/MD_frames/frames_opt_cat.xtc'.format(n_iteration)
    itp_file=glob.glob('needed_files/md_inp_files/*itp')[0]
    pdb_file = glob.glob('needed_files/md_inp_files/*pdb')[0]
    
    try:
        shutil.rmtree("new_iteration_{0}/input_files".format(n_iteration))
    except OSError:
        pass
    os.makedirs("new_iteration_{0}/input_files".format(n_iteration))
    
    t = mdtraj.load(xtc_file, top = pdb_file,standard_names=False)

    #######################################################################
    filename_counter=0
   
    for frame in t[::1]:
        # Make directory
        os.makedirs("new_iteration_{0}/input_files/f{0}_{1:05d}".format(n_iteration,filename_counter))

        # Write what is gonna be calculated
        calc_name = "new_iteration_{0}/input_files/f{0}_{1:05d}/calculate.dat".format(n_iteration,filename_counter)
        with open(calc_name,'w') as fw:
            fw.write('vibrational_data {0}\n'.format(c_vib))
            fw.write('chemical_shifts {0}\n'.format(c_shifts))
            fw.write('spinspin_couplings {0}\n'.format(c_spinspin))
                
##################################################################################################                    
        # call the class for the frame
        frame_features_i = frame_features(frame,molecule_features)
##################################################################################################                    
      
        write_folder = 'new_iteration_{0}/input_files'.format(n_iteration)
        
        # option 1 - Raman low quality
        if method == 'm1_raman_low':       
            make_gaussian_inputs_o1_opt(filename_counter,frame, frame_features_i, oniom_vdw_parameters,itp_file,n_iteration,qm_m_nproc,write_folder)
            make_gaussian_inputs_o1_raman_roa(filename_counter,frame, frame_features_i, oniom_vdw_parameters,itp_file,n_iteration,qm_m_nproc,write_folder)
            make_gaussian_inputs_o1_nmr_shifts(filename_counter,frame, frame_features_i, oniom_vdw_parameters,itp_file,n_iteration,qm_m_nproc,write_folder)
            make_gaussian_inputs_o1_nmr_spinspin_couplings(filename_counter,frame,frame_features_i,n_iteration,qm_m_nproc,write_folder)
            make_gaussian_inputs_o1_make_dqs_aurum(filename_counter,w_calculate,n_iteration,write_folder)
            make_python_scripts(n_iteration,filename_counter,molecule_features,write_folder)           
            
        # option 2 - Raman high quality
        elif method == 'm2_raman_high':

            # optimization, shifts, and spinspin couplings are the same as in option 1
            make_gaussian_inputs_o1_opt(filename_counter,frame, frame_features_i, oniom_vdw_parameters,itp_file,n_iteration,qm_m_nproc,write_folder)
            make_gaussian_inputs_o2_raman_roa(filename_counter,frame, frame_features_i, oniom_vdw_parameters,itp_file,n_iteration,qm_m_nproc,write_folder)
            make_gaussian_inputs_o1_nmr_shifts(filename_counter,frame, frame_features_i, oniom_vdw_parameters,itp_file,n_iteration,qm_m_nproc,write_folder)
            make_gaussian_inputs_o1_nmr_spinspin_couplings(filename_counter,frame,frame_features_i,n_iteration,qm_m_nproc,write_folder)
            make_gaussian_inputs_o2_make_dqs_aurum(filename_counter,w_calculate,n_iteration,write_folder)
            make_python_scripts(n_iteration,filename_counter,molecule_features,write_folder)           

        # file counter
        filename_counter+=1

    print("DONE")




