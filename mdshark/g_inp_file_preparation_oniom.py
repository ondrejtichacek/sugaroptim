import pickle
from sys import argv
sc,f1_goptfile,f2_gfinac_new,f3_gfinalc_sample = argv


atom_names=[]
oniom_levels=[]
xyz=[]

with open('molecule_features.pickle', 'rb') as f:
    molecule_features = pickle.load(f)

charge = str(molecule_features['charge'])
multiplicity = str(molecule_features['multiplicity'])

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
        lst=line.split()
        if lst==[charge,multiplicity,charge,multiplicity,charge,multiplicity]:
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
