import numpy as np
import os
import math as m
import glob
import mdtraj
import re
import sys

class class_molecule_features:
    """
Class for loading structure in gromacs format *.gro and extracting all unique dihedral angles or puckering coordinates.
Moreover it will write the initial plumed.dat file.
    """
    def __init__(self,name,itp_file,**kwargs):
        self.name=name

        # def functions used here

        def f5(seq, idfun=None):
            if idfun is None:
                def idfun(x): return x
            seen = {}
            result = []
            for item in seq:
                marker = idfun(item)
                if marker in seen: continue
                seen[marker] = 1
                result.append(item)
            return result


        def get_charges(itp_file):
            charges_array=[]
            read_f=0
            with open(itp_file) as fr:
                sum=0
                for line in fr:
                    if line =='[ atoms ]\n':
                        read_f=1
                    if read_f==1 and line =='[ bonds ]\n':
                        read_f=0
                    try:
                        if read_f==1 and line.split()[0]!=';':
                            sum+=float(line.split()[6])
                            charges_array.append(float(line.split()[6]))
                    except IndexError:
                        pass
            # sort cases
            # 0
            if sum>-0.25 and sum<0.25:
                charge=0
            # 1
            elif sum>0.5 and sum<1.25:
                charge=1
            # 2
            elif sum>1.25 and sum<2.1:
                charge=2
            # -1
            elif sum>-1.25 and sum<-0.5:
                charge=-1
            # -2
            elif sum>-2.1 and sum<-1.25:
                charge=-2
            else:
                print('The charge is quite big, is it correct?')
                charge=round(sum)
                print("CHECK CHARGE OF YOUR MOLECULE PROPERLY - IT MAY BE WRONG")
            print('Charge of you molecule is: {0}'.format(charge))
            return charge,charges_array

        ################ start ####################

        #############################
        # get xyz, elements,indices #
        #############################

        # load it using mdtraj
        t_name=glob.glob(name)[0]
        t = mdtraj.load(t_name)

        # get mol idx/water idx
        try:
            sel_mol_idx = t.top.select(kwargs['select_central_mol'])
            print("Central mol selection: ",kwargs['select_central_mol'])
            print("Indices selected: ",sel_mol_idx)
        except:
            sel_mol_idx = t.top.select("not water")
            print("No central mol selection")
            print("Indices selected: ",sel_mol_idx)
        self.mol_idx = sel_mol_idx
        self.water_idx = t.top.select("water")


        # get itp file
        itp_file = glob.glob(itp_file)[0]

        # get charge of your molecule
        self.charge,self.individual_charges = get_charges(itp_file)

        # get xyz
        t.xyz
        molecule_idx = t.top.select("not (water or name NA or name CL)")
        x=np.squeeze(t.xyz)[molecule_idx]
        self.xyz=x
        self.mol_xyz=x

        # get elements
        elements = np.array([i.element.symbol for i in t.topology.atoms])[molecule_idx]
        self.elements=elements

        # atom indices
        self.idx=np.arange(len(self.xyz))

        # n atoms
        natoms=np.max(molecule_idx)+1
        self.natoms=natoms

        # Atom list:
        al=[i.element.symbol for i in t.topology.atoms][:natoms]
        self.atom_list = al

        # Multiplicity of the system - always gonna be 1
        self.multiplicity = 1

        ##########################
        # stereochemical centers #
        ##########################

        def dihedral2(p):
            b = p[:-1] - p[1:]
            b[0] *= -1
            v = np.array( [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] )
            # Normalize vectors
            v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
            b1 = b[1] / np.linalg.norm(b[1])
            x = np.dot(v[0], v[1])
            m = np.cross(v[0], b1)
            y = np.dot(m, v[1])
            return np.degrees(np.arctan2( y, x ))

        # load stereochemical features
        stereochemical_centers=[]
        for idx_i,i in enumerate(self.xyz):
            al=[]
            for idx_j,j in enumerate(self.xyz):
                dist=(np.linalg.norm(i-j))
                if dist<0.165 and dist !=0:
                    #print("{0} {1}".format(idx_j,dist))
                    al.append(idx_j)
            if len(al)==4:
                dih="{0:.3f}".format(dihedral2(self.xyz[al])/180*np.pi)
                stereochemical_centers.append([[i+1 for i in al],dih])
        self.stereochemical_centers=stereochemical_centers

        ###################
        # calculate bonds #
        ###################
        bonds_count=[]
        bonds=[]

        for idx_i,i in enumerate(x):

            bonds_c=0
            for idx_j,j in enumerate(x):
                dist=np.linalg.norm(i-j)
                if elements[idx_i] == 'H' or  elements[idx_j] == 'H':
                    if dist <0.12 and dist >0:
                        bonds.append([idx_i,idx_j])
                        bonds_c+=1
                else:
                    if dist <0.165 and dist >0:
                        bonds.append([idx_i,idx_j])
                        bonds_c+=1
            bonds_count.append(bonds_c)
        bonds=np.array(bonds)
        self.bonds=bonds
        #self.bonds_count=bonds_count

        ##################################
        # get all unique dihedral angles #
        ##################################

        dih_list=[]
        bonds12=[]
        bonds13=[]


        for i in np.arange(len(elements)):
            list2=(np.where(bonds==i))[0]
            for l2 in list2:
                lx2=[bonds[l2][bonds[l2]!=i][0]]
                for lx2a in lx2:
                    list3=(np.where(bonds==lx2a))[0]
                    for l3 in list3:
                        lx3=[bonds[l3][bonds[l3]!=lx2a][0]]
                        b12=[i,lx3[0]]
                        bonds12.append(b12)
                        for lx3a in lx3:
                            list4=(np.where(bonds==lx3a))[0]
                            for l4 in list4:
                                lx4=[bonds[l4][bonds[l4]!=lx3a][0]]
                                dih = [i,lx2[0],lx3[0],lx4[0]]
                                b13 = [i,lx4[0]]
                                if len(f5(dih)) == 4:
                                    dih_list.append(dih)
                                    bonds13.append(b13)


        # Set all two/three bonds away atoms ( delete 1-2-1-2 dihedrals)
        ub2 = []
        for i in bonds12:
            if len (set(i)) == 2:
                ub2.append(list(set(i)))
        ub2 = np.array([sorted(list(i)) for i in list(dict.fromkeys({tuple(row) for row in ub2}))])+1
        ub3 = np.array([sorted(list(i)) for i in list(dict.fromkeys({tuple(row) for row in bonds13}))])+1

        self.bonds12 = ub2
        self.bonds13 = ub3


        #print(dih_list)
        #dih_set=[list(i) for i in list(dict.fromkeys({tuple(row) for row in dih_list}))]
        #print(dih_set)


        # delete redundant ones - OLD
        #dih_set=np.vstack({tuple(row) for row in dih_list})
        dih_set=np.array([list(i) for i in list(dict.fromkeys({tuple(row) for row in dih_list}))])
        # order the dihedral angles
        new_dih_set=[]
        for i in dih_set:
            if i[1] < i[2]:
                new_dih_set.append(list(i))
            else:
                new_dih_set.append([i[3],i[2],i[1],i[0]])
        # filter the duplicates
        #new_dih_setX=np.vstack({tuple(row) for row in new_dih_set})
        new_dih_setX=np.array([list(i) for i in list(dict.fromkeys({tuple(row) for row in new_dih_set}))])


        # filter so one bond is described only by one dihedral
        od_all=[]
        for idxi,i in enumerate(new_dih_setX):
            od=[]
            for idxj,j in enumerate(new_dih_setX):
                if idxi>idxj:
                    if i[1] == j[1] and i[2] == j[2]:
                        od.append(idxj)
            od_all.append(od)


        # now the selction array - if dih is unique (is empty) then keep it, otehrwise throw it away
        sarray=[]
        for i in od_all:
            if i == []:
                sarray.append(True)
            else:
                sarray.append(False)

        final_dih_set=new_dih_setX[np.array(sarray)]

        # final set of dihedral angles starting with index 1
        final_dihedral_a=np.array(final_dih_set+np.ones(np.shape(final_dih_set)),dtype=int)
        self.dihedrals=final_dihedral_a
        self.all_dihedrals=np.array(np.array(dih_list) + np.ones(np.shape(dih_list)),dtype=int)


        ########################
        # Amide bond detection #
        ########################


        amide_bond = []
        for i in self.all_dihedrals:
            if  elements[i[0]-1] =='H' and elements[i[1]-1] =='N' and\
                elements[i[2]-1] =='C' and elements[i[3]-1] =='O':
                amide_bond.append(i)
        # now filter them
        amide_bond = np.array([list(x) for x in set(tuple(x) for x in amide_bond)])

        if len(amide_bond) == 0:
            self.amide_bond = []
        else:
            self.amide_bond = amide_bond

        ##########################
        # polar groups detection #
        ##########################

        # 1) amide atoms + 1
        if amide_bond!=[]:
            bonded_atoms=[]
            for amide_bond_i in amide_bond:
                for ab_atom in amide_bond_i:
                    bf=bonds[np.where(ab_atom-1==bonds)[0]]
                    bonded_atoms.extend(bf)
            pg_amide_bond=np.unique([[i[0],i[1]] for i in bonded_atoms])+1
            self.pg_amide_bond=pg_amide_bond
        else:
            self.pg_amide_bond=[]
        # 2) all nitrogen atoms + 1
        bonded_atoms=[]
        nitrogen_atoms=np.where(elements=='N')
        if nitrogen_atoms[0].size > 0:
            for i in nitrogen_atoms:
                bf=bonds[np.where(i==bonds)[0]]
                bonded_atoms.extend(bf)
            pg_nitrogen=np.unique([[i[0],i[1]] for i in bonded_atoms])+1
            self.pg_nitrogen=pg_nitrogen
        else:
            self.pg_nitrogen=[]

        # 3) all sulphur atoms + 1
        bonded_atoms=[]
        sulphur_atoms=np.where(elements=='S')
        if sulphur_atoms[0].size > 0:
            print('y')
            for i in sulphur_atoms:
                bf=bonds[np.where(i==bonds)[0]]
                bonded_atoms.extend(bf)
            pg_sulphur=np.unique([[i[0],i[1]] for i in bonded_atoms])+1
            self.pg_sulphur=pg_sulphur
        else:
            self.pg_sulphur=[]

        # 4) all caboxylates

        # looking for carbon atom with 3 bonds only
        pg_carboxylate=[]
        for i in np.where(elements=='C')[0]:
            bf=bonds[np.where(i==bonds)[0]]
            unique_atoms=np.unique([[i[0],i[1]] for i in bf])
            if len(unique_atoms) == 4: # because we count also the central atom
                sl=sorted(elements[unique_atoms])
                if sl[0] =='C' and sl[1] =='C' and sl[2] =='O' and sl[3] =='O':
                    pg_carboxylate.extend(unique_atoms+1)
        self.pg_carboxylate=pg_carboxylate

        # Merge the polar groups together
        pg_all=list(self.pg_amide_bond)+list(self.pg_nitrogen)+list(self.pg_sulphur)+list(self.pg_carboxylate)
        polar_atoms=np.unique(np.array(pg_all))
        self.polar_atoms=polar_atoms

        ######################################################
        # puckering detection - 6 membered rings - pyranoses #
        ######################################################

        dih_list=[]
        for i in np.arange(len(elements)):
            list2=(np.where(bonds==i))[0]
            for l2 in list2:
                lx2=[bonds[l2][bonds[l2]!=i][0]]
                for lx2a in lx2:
                    list3=(np.where(bonds==lx2a))[0]
                    for l3 in list3:
                        lx3=[bonds[l3][bonds[l3]!=lx2a][0]]
                        for lx3a in lx3:
                            list4=(np.where(bonds==lx3a))[0]
                            for l4 in list4:
                                lx4=[bonds[l4][bonds[l4]!=lx3a][0]]
                                for lx4a in lx4:
                                    list5=(np.where(bonds==lx4a))[0]
                                    for l5 in list5:
                                        lx5=[bonds[l5][bonds[l5]!=lx4a][0]]
                                        for lx5a in lx5:
                                            list6=(np.where(bonds==lx5a))[0]
                                            for l6 in list6:
                                                lx6=[bonds[l6][bonds[l6]!=lx5a][0]]
                                                for lx6a in lx6:
                                                    list7=(np.where(bonds==lx6a))[0]
                                                    for l7 in list7:
                                                        lx7=[bonds[l7][bonds[l7]!=lx6a][0]]
                                                        dih=[i,lx2[0],lx3[0],lx4[0],lx5[0],lx6[0],lx7[0]]
                                                        if len(f5(dih)) ==6:
                                                            dih_list.append(dih)

        possible_dih=[]
        #dih_set=np.vstack({tuple(row) for row in dih_list})
        dih_set=np.array([list(i) for i in list(dict.fromkeys({tuple(row) for row in dih_list}))])
        for i in dih_set:
            if i[0]== i[6]:
                possible_dih.append((i[:6]))

        if possible_dih != []:
            # filter more redundant ones
            od_all=[]
            for idxi,i in enumerate(possible_dih):
                od=[]
                for idxj,j in enumerate(possible_dih):
                    if idxi>idxj:
                        if set(i)==set(j):
                            od.append(idxj)
                od_all.append(od)


            slist=np.array(np.unique(np.concatenate(od_all,axis=0)),dtype=int)
            sarray=[]
            for i in np.arange(len(possible_dih)):
                if i in slist:
                    sarray.append(False)
                else:
                    sarray.append(True)

            # your almost final puckering set
            final_dih_set=np.array(possible_dih)[np.array(sarray)]

            puck_final_list=[]
            for idx,i in enumerate(final_dih_set):
                puck_elements=np.array(elements)[np.array(i)]
                puck_bonds=np.array(bonds_count)[np.array(i)]
                pyranose=1
                for j in zip(puck_elements,puck_bonds):
                    # Loop through 6 atoms, if all atoms are either 4 bondeed carbons, or 2 bonded oxygens/nitrogens
                    if (j[0] == 'C' and j[1] == 4) or (j[0] == 'O' and j[1] == 2) or (j[0] == 'N' and j[1] == 2):
                        pass
                    else:
                        pyranose=1

                # your final puckering set
                if  pyranose == 1:
                    puck_final_list.append(i)

            # final set of puckering pyranose atoms that start with index 1
            puck_final_list=np.array(puck_final_list+np.ones(np.shape(puck_final_list)),dtype=int)
            self.puckering=puck_final_list
        else:
            self.puckering=[]

        ######################################################
        # puckering detection - 5 membered rings - furanoses #
        ######################################################

        dih_list=[]
        for i in np.arange(len(elements)):
            list2=(np.where(bonds==i))[0]
            for l2 in list2:
                lx2=[bonds[l2][bonds[l2]!=i][0]]
                for lx2a in lx2:
                    list3=(np.where(bonds==lx2a))[0]
                    for l3 in list3:
                        lx3=[bonds[l3][bonds[l3]!=lx2a][0]]
                        for lx3a in lx3:
                            list4=(np.where(bonds==lx3a))[0]
                            for l4 in list4:
                                lx4=[bonds[l4][bonds[l4]!=lx3a][0]]
                                for lx4a in lx4:
                                    list5=(np.where(bonds==lx4a))[0]
                                    for l5 in list5:
                                        lx5=[bonds[l5][bonds[l5]!=lx4a][0]]
                                        for lx5a in lx5:
                                            list6=(np.where(bonds==lx5a))[0]
                                            for l6 in list6:
                                                lx6=[bonds[l6][bonds[l6]!=lx5a][0]]
                                                dih=[i,lx2[0],lx3[0],lx4[0],lx5[0],lx6[0]]
                                                if len(f5(dih)) == 5:
                                                    dih_list.append(dih)

        possible_dih=[]
        #dih_set=np.vstack({tuple(row) for row in dih_list})
        dih_set=np.array([list(i) for i in list(dict.fromkeys({tuple(row) for row in dih_list}))])
        for i in dih_set:
            if i[0]== i[5]:
                possible_dih.append((i[:5]))

        if possible_dih != []:
            # filter more redundant ones
            od_all=[]
            for idxi,i in enumerate(possible_dih):
                od=[]
                for idxj,j in enumerate(possible_dih):
                    if idxi>idxj:
                        if set(i)==set(j):
                            od.append(idxj)
                od_all.append(od)


            slist=np.array(np.unique(np.concatenate(od_all,axis=0)),dtype=int)
            sarray=[]
            for i in np.arange(len(possible_dih)):
                if i in slist:
                    sarray.append(False)
                else:
                    sarray.append(True)

            # your almost final puckering set
            final_dih_set=np.array(possible_dih)[np.array(sarray)]

            puck_final_list=[]
            for idx,i in enumerate(final_dih_set):
                puck_elements=np.array(elements)[np.array(i)]
                puck_bonds=np.array(bonds_count)[np.array(i)]
                furanose=1
                for j in zip(puck_elements,puck_bonds):
                    # Loop through 6 atoms, if all atoms are either 4 bondeed carbons, or 2 bonded oxygens, or 3/4 bonded nitrogens

                    if (j[0] == 'C' and j[1] == 4) or (j[0] == 'O' and j[1] == 2) or (j[0] == 'N' and j[1] == 4) or (j[0] == 'N' and j[1] == 3):
                        pass
                    else:
                        furanose=1

                # your final puckering set
                if  furanose == 1:
                    puck_final_list.append(i)

            # final set of puckering pyranose atoms that start with index 1
            puck_final_list=np.array(puck_final_list+np.ones(np.shape(puck_final_list)),dtype=int)
            self.puckering5=puck_final_list
        else:
            self.puckering5=[]

        # Find unique improper dihedrals to restrain them
        def all_numbers(s):
            all_numbers=1
            for i in s:
                try:
                    int(i)
                except ValueError:
                    all_numbers=0
                    break
            if all_numbers == 1:
                return True
            else:
                return False
        unique_improper_dihedrals=[]
        with open(itp_file,'r') as fr:
            for line in fr:
                if len(line.split()[:5]) == 5:
                    if all_numbers(line.split()[:5]) == True and int(line.split()[4]) == (1 or 2):
                        uid=[[int(i) for i in line.split()[:4]],np.pi]
                        unique_improper_dihedrals.append(uid)
        self.unique_improper_dihedrals=unique_improper_dihedrals


        # make array for features
        features_def_list=[]
        for idx,i in enumerate(self.dihedrals):
            features_def_list.append('d{0}'.format(idx))
        for idx,i in enumerate(self.puckering):
            features_def_list.append('puck{0}.phi'.format(idx))
            features_def_list.append('puck{0}.theta'.format(idx))
        for idx,i in enumerate(self.puckering5):
            features_def_list.append('puck5f{0}.phs'.format(idx))

        self.features_def_list = features_def_list

        # make array for features - numbered
        features_def_list_idx=[]
        for i in self.features_def_list:
            if i[0] == 'd' and i[1].isdigit():
                features_def_list_idx.append(1)
            elif 'puck' in i and 'phi' in i:
                features_def_list_idx.append(21)
            elif 'puck' in i and 'theta' in i:
                features_def_list_idx.append(22)
            elif 'puck5f' in i and 'phs' in i:
                features_def_list_idx.append(31)
            else:
                print("Dont recognize type of the variable.")
        self.features_def_list_idx = features_def_list_idx


    # Lastly check if there is a plumed_redefined.dat file, if yes, then use it
        try:
            with open('needed_files/plumed_redefined.dat','r') as fr:
                for line in fr:
                    # dihedrals
                    if line.split()[0] == 'TORSION':
                        s = line.split()[1][6:]
                        s = [int(i) for i in s.split(',')]
                        self.dihedrals[int(line.split()[-1][7:])] = s
                    # pyranose puck
                    if (line.split()[0] == 'PUCKERING') and (len([int(i) for i in line.split()[1][6:].split(',')]) == 6):
                        s = line.split()[1][6:]
                        s = [int(i) for i in s.split(',')]
                        self.puckering[int(line.split()[-1][10:])] = s
                    # furanose puck
                    if (line.split()[0] == 'PUCKERING') and (len([int(i) for i in line.split()[1][6:].split(',')]) == 5):
                        s = line.split()[1][6:]
                        s = [int(i) for i in s.split(',')]
                        self.puckering5[int(line.split()[-1][12:])] = s
            print("Found plumed_redefined.dat file. Using these definition for features instead.")
        except:
            pass

                            

    def features(self):
        return self.dihedrals, self.puckering, self.puckering5
        #self.puckering=puck_final_list

    def write_plumed_features(self):
        str_to_write=''

        ### dihedral angles ###
        dihedral_a=self.dihedrals
        # dihedral angles
        idx_i=0
        # write first line
        i=dihedral_a[0]
        dw="{0},{1},{2},{3}".format(i[0],i[1],i[2],i[3])
        str_to_write+="TORSION ATOMS={0} LABEL=d{1}\n".format(dw,idx_i)
        idx_i+=1
        # write the rest
        for i in dihedral_a[1:]:
            dw="{0},{1},{2},{3}".format(i[0],i[1],i[2],i[3])
            str_to_write+="TORSION ATOMS={0} LABEL=d{1}\n".format(dw,idx_i)
            idx_i+=1

        ### pyranose puckering ###
        puckering_a=self.puckering
        idx_i=0
        # write puckering pyranose feature
        for i in puckering_a:
            dw="{0},{1},{2},{3},{4},{5}".format(i[0],i[1],i[2],i[3],i[4],i[5])
            str_to_write+="PUCKERING ATOMS={0} LABEL=puck{1}\n".format(dw,idx_i)
            idx_i+=1

        ### furanose puckering ###
        puckering_a=self.puckering5
        idx_i=0
        # write puckering furanose feature
        for i in puckering_a:
            dw="{0},{1},{2},{3},{4}".format(i[0],i[1],i[2],i[3],i[4])
            str_to_write+="PUCKERING ATOMS={0} LABEL=puck5f{1}\n".format(dw,idx_i)
            idx_i+=1

        ### return it ###
        return str_to_write

    def write_plumed_stereochemical_centers(self):
        str_to_write=''
        if self.stereochemical_centers == []:
            pass
        else:
            for idx_i,i in enumerate(self.stereochemical_centers):
                dw="{0},{1},{2},{3}".format(*i[0])
                str_to_write+="TORSION ATOMS={0} LABEL=sc{1}\n".format(dw,idx_i)
        return str_to_write

    def write_plumed_improper_dihedrals(self):
        str_to_write=''
        if self.unique_improper_dihedrals == []:
            pass
        else:
            for idx_i,i in enumerate(self.unique_improper_dihedrals):
                dw="{0},{1},{2},{3}".format(*i)
                str_to_write+="TORSION ATOMS={0} LABEL=improper_dih{1}\n".format(dw,idx_i)                            
        return str_to_write

    def write_plumed_amide_bonds(self):
        str_to_write=''
        if self.amide_bond == []:
            pass
        else:
            for idx_i,i in enumerate(self.amide_bond):
                dw="{0},{1},{2},{3}".format(*i)
                str_to_write+="TORSION ATOMS={0} LABEL=amide_bond{1}\n".format(dw,idx_i)                            
        return str_to_write

    def write_plumed_stereochemical_centers_restraints(self):
        steoreochemistry_kappa = 1000
        str_to_write=''
        if self.stereochemical_centers != []:
            str_to_write += 'RESTRAINT ARG='
            for idx,i in enumerate(self.stereochemical_centers[:-1]):
                str_to_write += 'sc{0},'.format(idx)
 
            str_to_write += 'sc{0}'.format(len(self.stereochemical_centers)-1)
            str_to_write += ' AT='
            for idx,i in enumerate(self.stereochemical_centers[:-1]):
                str_to_write += '{0},'.format(i[1])
            str_to_write += '{0}'.format(self.stereochemical_centers[len(self.stereochemical_centers)-1][1])
            str_to_write += ' KAPPA='
            for idx,i in enumerate(self.stereochemical_centers[:-1]):
                str_to_write += '{0},'.format(steoreochemistry_kappa)
            str_to_write += '{0}'.format(steoreochemistry_kappa)
            str_to_write += ' LABEL=stereochemical_restraint\n'
        return str_to_write

    def write_plumed_stereochemical_centers_restraints_print(self,write_frequency,plumed_output_folder):
        str_to_write=''
        if self.stereochemical_centers != []:
            str_to_write += 'PRINT ARG='
            for idx,i in enumerate(self.stereochemical_centers):
                str_to_write += 'sc{0},'.format(idx)
            str_to_write += 'stereochemical_restraint.bias STRIDE={0} FILE={1}plumed_output_stereochemical_centers_bias.dat\n\n'.format(write_frequency,plumed_output_folder)
        return str_to_write

    def write_plumed_improper_dihedrals_restraints(self):
        kappa = 1000
        str_to_write=''
        if self.unique_improper_dihedrals != []:
            str_to_write += 'RESTRAINT ARG='
            for idx,i in enumerate(self.unique_improper_dihedrals[:-1]):
                str_to_write += 'improper_dih{0},'.format(idx)
 
            str_to_write += 'improper_dih{0}'.format(len(self.unique_improper_dihedrals)-1)
            str_to_write += ' AT='
            for idx,i in enumerate(self.unique_improper_dihedrals[:-1]):
                str_to_write += '{0},'.format(i[1])
            str_to_write += '{0}'.format(self.unique_improper_dihedrals[len(self.unique_improper_dihedrals)-1][1])
            str_to_write += ' KAPPA='
            for idx,i in enumerate(self.unique_improper_dihedrals[:-1]):
                str_to_write += '{0},'.format(kappa)
            str_to_write += '{0}'.format(kappa)
            str_to_write += ' LABEL=improper_dih_restraint\n'
        return str_to_write

    def write_plumed_improper_dihedrals_restraints_print(self,write_frequency,plumed_output_folder):
        str_to_write=''
        if self.unique_improper_dihedrals != []:
            str_to_write += 'PRINT ARG='
            for idx,i in enumerate(self.unique_improper_dihedrals):
                str_to_write += 'improper_dih{0},'.format(idx)
            str_to_write += 'improper_dih.bias STRIDE={0} FILE={1}plumed_output_improper_dih_bias.dat\n'.format(write_frequency,plumed_output_folder)
        return str_to_write

    def write_plumed_amide_bond_restraints(self,kwargs_dict):
        try:
            freeze_amide_bonds = kwargs_dict['freeze_amide_bonds']
        except:
            freeze_amide_bonds = True
            print("Freeze_amide_bond is not defined/ill defined, setting to True")
        if freeze_amide_bonds == True:
            kappa = 1000
        else:
            kappa = 0

        str_to_write=''
        if self.amide_bond != []:
            str_to_write += 'RESTRAINT ARG='
            for idx,i in enumerate(self.amide_bond[:-1]):
                str_to_write += 'amide_bond{0},'.format(idx)
 
            str_to_write += 'amide_bond{0}'.format(len(self.amide_bond)-1)
            str_to_write += ' AT='
            for idx,i in enumerate(self.amide_bond[:-1]):
                str_to_write += '{0},'.format('3.1416')
            str_to_write += '{0}'.format('3.1416')
            str_to_write += ' KAPPA='
            for idx,i in enumerate(self.amide_bond[:-1]):
                str_to_write += '{0},'.format(kappa)
            str_to_write += '{0}'.format(kappa)
            str_to_write += ' LABEL=amide_bond_restraint\n'
        return str_to_write

    def write_plumed_amide_bond_restraints_print(self,write_frequency,plumed_output_folder):
        str_to_write=''
        if self.amide_bond != []:
            str_to_write += 'PRINT ARG='
            for idx,i in enumerate(self.amide_bond):
                str_to_write += 'amide_bond{0},'.format(idx)
            str_to_write += 'amide_bond_restraint.bias STRIDE={0} FILE={1}plumed_output_amide_bond_bias.dat\n'.format(write_frequency,plumed_output_folder)
        return str_to_write

    def write_plumed_features_restraints(self,restrain_features_values):
        str_to_write=''
        features_kappa = 100
         # restrain them
        str_to_write += 'RESTRAINT ARG='
        for i in self.features_def_list[:-1]:
            str_to_write += i+','
        str_to_write += self.features_def_list[-1]
        str_to_write += ' AT='
        for i in restrain_features_values[:-1]:
            str_to_write += str(i)+','
        str_to_write += str(restrain_features_values[-1])
        str_to_write += ' KAPPA='
        for i in restrain_features_values[:-1]:
            str_to_write += '{0},'.format(features_kappa)
        str_to_write += '{0}'.format(features_kappa)
        str_to_write += '\n'
        return str_to_write

    def write_plumed_features_print(self,write_frequency,plumed_output_folder):
        str_to_write='PRINT STRIDE={0} ARG='.format(write_frequency)
        for j in self.features_def_list[:-1]:
            str_to_write += "{0},".format(j)
        str_to_write += self.features_def_list[-1]
        str_to_write += ' FILE={0}plumed_output_features.dat\n'.format(plumed_output_folder)
        return str_to_write

    def write_print_stereochemical_restraints(self,write_frequency,plumed_output_folder):
        str_to_write = ''
        if self.stereochemical_centers != []:
            str_to_write='PRINT STRIDE={0} ARG='.format(write_frequency)
            for j in range(len(self.stereochemical_centers)):
                str_to_write += "sc{0},".format(j)
            str_to_write += "sc{0}".format(len(self.stereochemical_centers))
            str_to_write += ' FILE={0}plumed_output_stereochemical_restraints.dat\n'.format(plumed_output_folder)
        return str_to_write

    def write_print_improper_dihedrals(self,write_frequency,plumed_output_folder):
        str_to_write = ''
        if self.unique_improper_dihedrals != []:
            str_to_write='PRINT STRIDE={0} ARG='.format(write_frequency)
            for j in range(len(self.unique_improper_dihedrals)):
                str_to_write += "improper_dih{0},".format(j)
            str_to_write += "improper_dih{0}".format(len(self.unique_improper_dihedrals))
            str_to_write += ' FILE={0}plumed_output_improper_dih.dat\n'.format(plumed_output_folder)
        return str_to_write

    def write_print_amide_bond(self,write_frequency,plumed_output_folder):
        str_to_write = ''
        if self.amide_bond != []:
            str_to_write='PRINT STRIDE={0} ARG='.format(write_frequency)
            for j in range(len(self.amide_bond)):
                str_to_write += "amide_bond{0},".format(j)
            str_to_write += "amide_bond{0}".format(len(self.unique_improper_dihedrals))
            str_to_write += ' FILE={0}plumed_output_amide_bond.dat\n'.format(plumed_output_folder)
        return str_to_write

    # To write initial plumed file
    def write_plumed(self,plumed_file):
        with open (plumed_file,'w') as fw:
            fw.write(self.write_plumed_features())
        return None

    # To write plumed print file
    def write_plumed_print_features(self,plumed_file_name,plumed_output_folder,write_frequency):
        with open(plumed_file_name,'w') as fw:
            fw.write(self.write_plumed_features())
            fw.write(self.write_plumed_features_print(write_frequency,plumed_output_folder))
            
    def write_additional_plumed_print_features(self,plumed_file_name,plumed_output_folder):
        try:
            os.system('cp needed_files/plumed_additional_features.dat {0}'.format(plumed_file_name))
            os.system("sed -i 's|FILEDEST|{0}|g' {1} ".format(plumed_output_folder,plumed_file_name))
        except:
            pass

    def write_plumed_file_MD0(self,plumed_file_name,plumed_output_folder,write_frequency,kwargs_dict):
        with open (plumed_file_name,'w') as fw:
            fw.write(self.write_plumed_features())
            fw.write(self.write_plumed_stereochemical_centers())
            fw.write(self.write_plumed_improper_dihedrals())
            fw.write(self.write_plumed_amide_bonds())

            fw.write(self.write_plumed_stereochemical_centers_restraints())
            fw.write(self.write_plumed_improper_dihedrals_restraints())
            fw.write(self.write_plumed_amide_bond_restraints(kwargs_dict))

            fw.write(self.write_plumed_stereochemical_centers_restraints_print(write_frequency,plumed_output_folder))
            fw.write(self.write_plumed_improper_dihedrals_restraints_print(write_frequency,plumed_output_folder))
            fw.write(self.write_plumed_amide_bond_restraints_print(write_frequency,plumed_output_folder))
            fw.write(self.write_plumed_features_print(write_frequency,plumed_output_folder))

    def write_plumed_file_MD0_check_features(self,plumed_file_name,plumed_output_folder,write_frequency):
        with open (plumed_file_name,'w') as fw:
            fw.write(self.write_plumed_features())
            fw.write(self.write_plumed_stereochemical_centers())
            fw.write(self.write_plumed_improper_dihedrals())
            fw.write(self.write_plumed_amide_bonds())

            fw.write(self.write_plumed_features_print(write_frequency,plumed_output_folder))

    def write_plumed_file_MD_opt(self,plumed_file_name,plumed_output_folder,write_frequency,kwargs_dict,restrain_features_values):
        with open (plumed_file_name,'w') as fw:
            fw.write(self.write_plumed_features())
            fw.write(self.write_plumed_stereochemical_centers())
            fw.write(self.write_plumed_improper_dihedrals())
            fw.write(self.write_plumed_amide_bonds())

            fw.write(self.write_plumed_stereochemical_centers_restraints())
            fw.write(self.write_plumed_improper_dihedrals_restraints())
            fw.write(self.write_plumed_amide_bond_restraints(kwargs_dict))
            fw.write(self.write_plumed_features_restraints(restrain_features_values))

            fw.write(self.write_plumed_stereochemical_centers_restraints_print(write_frequency,plumed_output_folder))
            fw.write(self.write_plumed_improper_dihedrals_restraints_print(write_frequency,plumed_output_folder))
            fw.write(self.write_plumed_amide_bond_restraints_print(write_frequency,plumed_output_folder))
            fw.write(self.write_plumed_features_print(write_frequency,plumed_output_folder))

    def make_index_file(self):
        # Prepare index file
        try:
            os.remove('needed_files/md_inp_files/index.ndx')
        except FileNotFoundError:
            pass
     
        grofile=glob.glob('needed_files/md_inp_files/*.gro')[0]
        os.system("printf \"del 1-30 \n r SOL \n name 1 SOLUTION \n 0 & ! 1 \n name 2 MOLECULE \n q \n\"|gmx make_ndx -f {0} -o needed_files/md_inp_files/index.ndx".format(grofile))
     
     
     
     

    def write_gromacs_mdp_files(self):

        mdp1=''';define                  = -DPOSRES
integrator              = steep
emtol                   = 1000.0
nsteps                  = 1000
nstlist                 = 10
cutoff-scheme           = Verlet
rlist                   = 1.2
vdwtype                 = Cut-off
vdw-modifier            = Potential-shift-Verlet
rvdw_switch             = 1.0
rvdw                    = 1.2
coulombtype             = pme
rcoulomb                = 1.2
coulomb-modifier        = Potential-shift-Verlet
constraints             = none
constraint_algorithm    = LINCS
nstxout                 = 1


'''

        mdp2=''';define                   = -DPOSRES
integrator              = md
dt                      = 0.0002
nsteps                  = 500000000
nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstcalcenergy           = 0
nstlog                  = 0
nstenergy               = 0
nstxout-compressed      = 0
;
cutoff-scheme           = Verlet
nstlist                 = 20    ; With Verlet list this is the minimum used value (mdrun might increase it).
rlist                   = 1.0   ; Ignored when using Verlet cutoff scheme
coulombtype             = pme
coulomb-modifier        = Potential-shift-Verlet
rcoulomb                = 1.0
vdwtype                 = Cut-off
vdw-modifier            = Potential-shift-Verlet
rvdw_switch             = 1.0
rvdw                    = 1.0
DispCorr                = EnerPres
;
tcoupl                  = v-rescale
tc_grps                 = MOLECULE   SOLUTION
tau_t                   = 1.0     1.0
ref_t                   = 300.00 300.00
;
;pcoupl                  = berendsen
pcoupl                  = no
pcoupltype              = isotropic
tau_p                   = 5.0
compressibility         = 4.5e-5
ref_p                   = 1.0
;
constraints             = all-bonds
constraint_algorithm    = LINCS
continuation            = no
gen-vel                 = yes
gen-temp                = 100
;
nstcomm                 = 100
comm_mode               = linear
comm_grps               = MOLECULE SOLUTION
;
refcoord_scaling        = com



'''

        mdp3='''integrator              = md
dt                      = 0.0002
nsteps                  = 500000000
nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstcalcenergy           = 5000
nstlog                  = 5000
nstenergy               = 5000
nstxout-compressed      = 5000

;
cutoff-scheme           = Verlet
verlet-buffer-tolerance = 0.0001
nstlist                 = 20    ; With Verlet list this is the minimum used value (mdrun might increase it).
rlist                   = 1.0   ; Ignored when using Verlet cutoff scheme
coulombtype             = pme
coulomb-modifier        = Potential-shift-Verlet
rcoulomb                = 1.0
vdwtype                 = Cut-off
vdw-modifier            = Potential-shift-Verlet
rvdw_switch             = 1.0
rvdw                    = 1.0
DispCorr                = EnerPres
;
tcoupl                  = v-rescale
tc_grps                 = MOLECULE   SOLUTION
tau_t                   = 0.001     0.001
ref_t                   = 8000.00 300.00
;
;pcoupl                  = berendsen
pcoupl                  = no
pcoupltype              = isotropic
tau_p                   = 5.0
compressibility         = 4.5e-5
ref_p                   = 1.0
;
constraints             = h-bonds
;constraints             = all-bonds
shake-tol               = 0.01
lincs-warnangle         = 90
constraint_algorithm    = LINCS
continuation            = no
gen-vel                 = yes
gen-temp                = 300.00
;
nstcomm                 = 100
comm_mode               = linear
comm_grps               = MOLECULE SOLUTION
;
refcoord_scaling        = com


'''

        mdp4='''integrator              = md
dt                      = 0.0002
nsteps                  = 500000000
nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstcalcenergy           = 200
nstlog                  = 200
nstenergy               = 200
nstxout-compressed      = 200

;
cutoff-scheme           = Verlet
verlet-buffer-tolerance = 0.0001
nstlist                 = 20    ; With Verlet list this is the minimum used value (mdrun might increase it).
rlist                   = 1.0   ; Ignored when using Verlet cutoff scheme
coulombtype             = pme
coulomb-modifier        = Potential-shift-Verlet
rcoulomb                = 1.0
vdwtype                 = Cut-off
vdw-modifier            = Potential-shift-Verlet
rvdw_switch             = 1.0
rvdw                    = 1.0
DispCorr                = EnerPres
;
tcoupl                  = v-rescale
tc_grps                 = MOLECULE   SOLUTION
tau_t                   = 0.001     0.001
ref_t                   = 300.00 300.00
;
;pcoupl                  = berendsen
pcoupl                  = no
pcoupltype              = isotropic
tau_p                   = 5.0
compressibility         = 4.5e-5
ref_p                   = 1.0
;
constraints             = all-bonds
constraint_algorithm    = LINCS
shake-tol               = 0.01
lincs-warnangle         = 90
continuation            = no
gen-vel                 = yes
gen-temp                = 100
;
nstcomm                 = 100
comm_mode               = linear
comm_grps               = MOLECULE SOLUTION
;
refcoord_scaling        = com



'''

        with open('needed_files/md_inp_files/md_min.mdp','w') as fw:
            fw.write(mdp1)
        with open('needed_files/md_inp_files/md_opt.mdp','w') as fw:
            fw.write(mdp2)
        with open('needed_files/md_inp_files/md_prod_it0.mdp','w') as fw:
            fw.write(mdp3)
        with open('needed_files/md_inp_files/md_prod_itX.mdp','w') as fw:
            fw.write(mdp4)
 

    def gro_to_pdb(self):
        if glob.glob('needed_files/md_inp_files/structure.pdb') == []:
            gro_file = glob.glob('needed_files/md_inp_files/*.gro')[0]
            os.system("printf \"0\n\"|gmx trjconv -f {0} -s {0} -o needed_files/md_inp_files/structure.pdb ".format(gro_file))
    





