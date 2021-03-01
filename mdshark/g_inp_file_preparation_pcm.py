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