from sys import argv
sc,f1_goptfile,f2_gfinac_new,f3_gfinalc_sample = argv


xyz=[]

n_atoms = None
xyz_last_line = None

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

if n_atoms is None or xyz_last_line is None:
    raise(ValueError("NAtoms and Coordinates not found in the log file, did the simulation finish properly?"))

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
        str2w='{0:8.3f}{1:8.3f}{2:8.3f} \n'.format(xyz[idx][0],xyz[idx][1],xyz[idx][2])
        str_w=str1+str2w
        f_o.write(str_w)
    f_o.write(str(f_sample_l[-1]))