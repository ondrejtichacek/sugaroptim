import numpy as np
from scipy.interpolate import interp1d
import glob
import os
import scipy.ndimage

# include python modules
import sys
sys.path.append(os.getcwd()+'/../0_python_modules/python_modules/')
import m_molecular_features

class experiment:
    def __init__(self,experimental_data_av):
        # load exp data first
        exp_data_path='needed_files/exp_data'
        # interpolate the experimetnal data on (minv,maxv) frequency scale
        if experimental_data_av[0] == 1:
            freq_ram,ram = np.loadtxt(exp_data_path+'/raman_icpu.dat',usecols=(0,1), unpack=True)
            freq_roa,roa = np.loadtxt(exp_data_path+'/roa_icpu.dat',usecols=(0,1), unpack=True)
 
            # set global freq range
            global minv
            global maxv
            minv=int(np.max([np.min(freq_ram),np.min(freq_roa)]))+1
            maxv=int(np.min([np.max(freq_ram),np.max(freq_roa)]))-1
 
            # interpolate
            freq_i=np.arange(minv,maxv+1)
            ram_i=interp1d(freq_ram,ram)(freq_i)
            roa_i=interp1d(freq_roa,roa)(freq_i)
 
            # smear them and use every second data point
            freq_i=freq_i[::2]
            ram_i=scipy.ndimage.filters.gaussian_filter(ram_i,sigma=1)[::2]
            roa_i=scipy.ndimage.filters.gaussian_filter(roa_i,sigma=1)[::2]
 
            # make it as a variable of class
            self.freq=freq_i
            self.ram=ram_i
            self.roa=roa_i
        else:
            self.freq = [1,2,3]
            self.ram = [1,2,3]
            self.roa = [1,2,3]

        # load nmr H data
        if experimental_data_av[2] == 1:
            atom_sch,sch=np.loadtxt(exp_data_path+'/sch.dat',usecols=(0,1), unpack=True)
        else:
            atom_sch,sch = [[],[]]

        # load nmr C data
        if experimental_data_av[3] == 1:
            atom_scc,scc=np.loadtxt(exp_data_path+'/scc.dat',usecols=(0,1), unpack=True)
        else:
            atom_scc,scc = [[],[]]
            
        # load nmr Jij data
        if experimental_data_av[4] == 1:
            jij,a1,a2=np.loadtxt(exp_data_path+'/jij.dat',usecols=(0,1,2), unpack=True)
        else:
            jij,a1,a2 = [[],[],[]]

        # make it as a variable of class
        self.sch=sch
        self.sch_idx=atom_sch
        self.scc=scc
        self.scc_idx=atom_scc
        self.jij=jij
        self.jij_idx1=a1
        self.jij_idx2=a2


class simulation:
    def __init__(self,path,molecule_features,experimental_data_av):

        # expand - what experimental data do we have?
        ed_raman,ed_roa,ed_shifts_H,ed_shifts_C,ed_spinspin = experimental_data_av

        # load exp data
        self.exp=experiment(experimental_data_av)

        def load_sim_roa_raman(path):

            # get the values
            freq,ram = np.loadtxt(path+'/raman_icpu.dat',usecols=(0,1), unpack=True)
            freq,roa = np.loadtxt(path+'/roa_icpu.dat',usecols=(0,1), unpack=True)
            
            # interpolate the experimetnal data on (minv,maxv) frequency scale
            freq_i=np.arange(minv,maxv+1)
            ram_i=interp1d(freq,ram)(freq_i)
            roa_i=interp1d(freq,roa)(freq_i)

            freq_i=freq_i[::2]
            ram_i=ram_i[::2]
            roa_i=roa_i[::2]

            return freq_i,ram_i,roa_i


        #####################
        # Now load sim data #
        #####################
        sim_data_path=path
        ind_folders = sorted(glob.glob(path+'/input_files/f*'))

        # #Check how many files do we have so we can prepare arrays
        fwc=0
        for i in ind_folders:
            fwc+=1
        self.n_files=fwc
    

        # shifts
        try:        
            # get atom lists for NMR -  which shifts correspond to which atoms
            for i in ind_folders:
                atoml_sch=np.loadtxt(i+'/sch.dat',usecols=0, unpack=True)
                atoml_scc=np.loadtxt(i+'/scc.dat',usecols=0, unpack=True)


            # make sim lists of nmr
            self.sch_idx=atoml_sch
            self.scc_idx=atoml_scc

            scc_array=np.empty([fwc,len(atoml_scc)])
            sch_array=np.empty([fwc,len(atoml_sch)])
        except:
            # make sim lists of nmr
            self.sch_idx=np.array([])
            self.scc_idx=np.array([])

            scc_array=np.empty([fwc,1])
            sch_array=np.empty([fwc,1])

        # spin-spin couplings
        try:
            # get atom lists for NMR -  which shifts correspond to which atoms
            for i in ind_folders:
                atoml_jij1=np.loadtxt(i+'/jij.dat',usecols=1, unpack=True)
                atoml_jij2=np.loadtxt(i+'/jij.dat',usecols=2, unpack=True)

            # make sim lists of nmr
            self.jij_idx1=atoml_jij1
            self.jij_idx2=atoml_jij2

            jij_array=np.empty([fwc,len(atoml_jij1)])
        except:
            # make sim lists of nmr
            self.jij_idx1=np.array([])
            self.jij_idx2=np.array([])

            jij_array=np.empty([fwc,1])


        # make arrays
        file_array = ["" for x in range(fwc)]
        freq_range=1*self.exp.freq
        data_array=np.zeros([fwc,3,np.shape(freq_range)[0]])
        freq_array=np.zeros([fwc,np.shape(freq_range)[0]])
        ram_array=np.zeros([fwc,np.shape(freq_range)[0]])
        roa_array=np.zeros([fwc,np.shape(freq_range)[0]])


        fa_features = [[] for x in range(fwc)]
        
        # additional features
        fa_additional_features = []

        # set default raman/roa scaling function
        self.scaling_function = [1,1,10,1000]

        # load arrays
        fwc=0
        for i in ind_folders:
            # file name
            file_array[fwc]=i
            
            # scaling fnc
            try:
                self.scaling_function = np.loadtxt(i+'/raman_scaling_function.dat')
            except:
                self.scaling_function = np.array([1,1,10,1000])
      
            try:
                # freq/ram/roa
                freq_array[fwc],ram_array[fwc],roa_array[fwc]=load_sim_roa_raman(i)
                f,ram,roa=load_sim_roa_raman(i)

                # Raman/ROA/IR/VCD data
                data_array[fwc][0]=f
                data_array[fwc][1]=ram
                data_array[fwc][2]=roa
            except:
                pass

            # NMR data
            try:                
                sch_array[fwc]=np.loadtxt(i+'/sch.dat',usecols=1, unpack=True)
            except:
                pass
            try:
                scc_array[fwc]=np.loadtxt(i+'/scc.dat',usecols=1, unpack=True)
            except:
                pass
            try:
                jij_array[fwc]=np.loadtxt(i+'/jij.dat',usecols=0, unpack=True)
            except:
                pass

            # Load features
            fa_features[fwc]=list(np.loadtxt(i+'/plumed_output_features.dat',unpack=True)[1:])
            
            # Try to load additional plumed features
            try:
                fa_additional_features.append(list(np.loadtxt(i+'/plumed_additional_output_features.dat',unpack=True)[1:]))
            except:
                pass
            
            # fwc count
            fwc+=1


#         if ed_shifts_H !=0 or ed_shifts_C != 0:
#             # Select only nmr data for which we have exp data
#             al_sch_sel=[]
#             for i in self.exp.sch_idx:
#                 al_sch_sel.append(np.where(self.sch_idx==i)[0][0])
#             self.sch=np.transpose(np.array([list(sch_array[:,i]) for i in al_sch_sel]))

#             al_scc_sel=[]
#             for i in self.exp.scc_idx:
#                 al_scc_sel.append(np.where(self.scc_idx==i)[0][0])
#             self.scc=np.transpose(np.array([list(scc_array[:,i]) for i in al_scc_sel]))
#         else:
#             self.sch=np.array([])
#             self.scc=np.array([])

#         if ed_spinspin !=0:
#             sel_jij=[]
#             sel_jij_idx=[]
#             for i in zip(self.exp.jij_idx1,self.exp.jij_idx2):
#                 sel_jij.append(list(np.sort(i)))
#             for i in sel_jij:
#                 for idx,j in enumerate([list(zipl) for zipl in zip(self.jij_idx1,self.jij_idx2)]):
#                     if j==i:
#                         sel_jij_idx.append(idx)

#             self.jij=np.transpose(np.array([list(jij_array[:,i]) for i in sel_jij_idx]))
#         else:
#             self.jij=np.array([])

        # Select only nmr data for which we have exp data
        al_sch_sel=[]
        for i in self.exp.sch_idx:
            al_sch_sel.append(np.where(self.sch_idx==i)[0][0])
        self.sch=np.transpose(np.array([list(sch_array[:,i]) for i in al_sch_sel]))

        al_scc_sel=[]
        for i in self.exp.scc_idx:
            al_scc_sel.append(np.where(self.scc_idx==i)[0][0])
        self.scc=np.transpose(np.array([list(scc_array[:,i]) for i in al_scc_sel]))

        sel_jij=[]
        sel_jij_idx=[]
        for i in zip(self.exp.jij_idx1,self.exp.jij_idx2):
            sel_jij.append(list(np.sort(i)))
        for i in sel_jij:
            for idx,j in enumerate([list(zipl) for zipl in zip(self.jij_idx1,self.jij_idx2)]):
                if j==i:
                    sel_jij_idx.append(idx)

        self.jij=np.transpose(np.array([list(jij_array[:,i]) for i in sel_jij_idx]))

        # Assign the rest
        self.file_array=file_array
        self.freq=freq_array
        self.freq_sim=freq_array
        self.ram=ram_array
        self.roa=roa_array
        self.data_array=data_array
        self.features=np.array(fa_features)
        self.additional_features = np.array(fa_additional_features)

        # Maybe delete afterwards?
        self.features_def_list = molecule_features.features_def_list
        self.features_def_list_idx = molecule_features.features_def_list_idx
        self.nfeatures=len(molecule_features.features_def_list)


# Load all data
class data:
    def __init__(self,path,iteration,molecule_features,experimental_data_av,**kwargs):

        # expand - what experimental data do we have?
        ed_raman,ed_roa,ed_shifts_H,ed_shifts_C,ed_spinspin = experimental_data_av

        ###################################################################################################
        # Define functions first
        self.iteration = iteration
        # NMR filtering scripts
        # Filter NMR data so they are statistically significant
        def pd(a,b):
            # periodic distance
            return np.amin([np.abs(a-b),2*np.pi-np.abs(a-b)])

        def filter_Jij_1(Jij):
            # Check Jij  data points for a too small spread of calculated values ((10,90) % interval is > 1.5 Hz)
            threshold = 1.0
            sa = (np.percentile(Jij,90,axis=0)-np.percentile(Jij,10,axis=0)) > threshold
            return sa

        def filter_sch_2(sch_sim,sch_exp):
            # Check H shifts data points for large deviances sim-exp
            threshold = 0.50
            sa = np.abs(np.average(sch_sim,axis=0)-sch_exp) < threshold
            return sa

        def filter_sch_1(sch_sim):
            # Check H shifts data points for a too small spread of calculated values ((10,90) % interval is > thr)
            threshold = 0.35
            sa = (np.percentile(sch_sim,90,axis=0)-np.percentile(sch_sim,10,axis=0)) > threshold
            return sa

        def filter_scc_2(scc_sim,scc_exp):
            # Check C shifts data points for large deviances sim-exp
            threshold = 6.0
            sa = np.abs(np.average(scc_sim,axis=0)-scc_exp) < threshold
            return sa

        def filter_scc_1(scc_sim):
            # Check C shifts data points for a too small spread of calculated values ((10,90) % interval is > thr)
            threshold = 3.0
            sa = (np.percentile(scc_sim,90,axis=0)-np.percentile(scc_sim,10,axis=0)) > threshold
            return sa

        ###############################################################################################
        # Load whole molecule data description - stereochemical centers, features, atributes and so on
        p_dir=os.getcwd()
        self.molecule_features = molecule_features

        # Now load the data
        self.exp=experiment(experimental_data_av)
        self.sim=simulation(path,molecule_features,experimental_data_av)


        if ed_spinspin !=0:
            # Check Jij exp x sim -> many times the sign is wrongly assigned !!!
            # Therefore compare to <sim> and change accordingly
            a=np.average(self.sim.jij,axis=0)>=0
            b=self.exp.jij>=0
            jij_sign_check = a&b
            for idx,i in enumerate(zip(a,b)):
                if i[0]==i[1]:
                    pass
                else:
                    print("Warning, Jij shift does not match in sign (exp x sim)")
                    print('Jij: {0} {1}, exp: {2} <sim>: {3}'.\
                          format(int(self.exp.jij_idx1[idx]),int(self.exp.jij_idx2[idx]),\
                          round(self.exp.jij[idx],1),round(np.average(self.sim.jij,axis=0)[idx],1)))
                    #change
                    self.exp.jij[idx]=-1*self.exp.jij[idx]
                    print("Changing exp data sign to match simulation data.")

        # Assign original values - not for optimization:
        self.sim.nf_jij=self.sim.jij.copy()
        self.exp.nf_jij=self.exp.jij.copy()
        self.exp.nf_jij_idx1=self.exp.jij_idx1.copy()
        self.exp.nf_jij_idx2=self.exp.jij_idx2.copy()

        self.sim.nf_sch=self.sim.sch.copy()
        self.exp.nf_sch=self.exp.sch.copy()
        self.exp.nf_sch_idx=self.exp.sch_idx.copy()
        self.sim.nf_scc=self.sim.scc.copy()
        self.exp.nf_scc=self.exp.scc.copy()
        self.exp.nf_scc_idx=self.exp.scc_idx.copy()

        try:
            do_not_filter = kwargs['do_not_filter']
        except:
            do_not_filter = False
        if do_not_filter == False:

            if ed_spinspin !=0:
                # Filter values so that they are statistically significant - for optimization
                # Filter Jij data - 1 - check whether the spread is significant (>than threshold)
                Jij_sa = filter_Jij_1(self.sim.jij)
                if np.sum(Jij_sa) != len(Jij_sa):
                    print('Warning, following Jij values (pairs) have too small spread of sim values - omitting them in the optimization.')
                    print([i for i in zip(self.exp.jij_idx1[~Jij_sa],self.exp.jij_idx2[~Jij_sa])])
 
                self.sim.jij=self.sim.jij[:,Jij_sa]
                self.exp.jij=self.exp.jij[Jij_sa]
                self.exp.jij_idx1=self.exp.jij_idx1[Jij_sa]
                self.exp.jij_idx2=self.exp.jij_idx2[Jij_sa]
 
 
            if ed_shifts_H !=0:
                # Filter sch data
                sch_sa_1 = filter_sch_1(self.sim.sch)
                if np.sum(sch_sa_1) != len(sch_sa_1):
                    print('Warning, following H chemical shifts have too small spread of sim values - omitting them in the optimization.')
                    print([int(i) for i in self.exp.sch_idx[~sch_sa_1]])
 
                sch_sa_2 = filter_sch_2(self.sim.sch,self.exp.sch)
                if np.sum(sch_sa_2) != len(sch_sa_2):
                    print('Warning, following H chemical shifts (atoms) \
                        average values are too far from experimental data!\
                        Ommiting in optimization for now.')
                    print([int(i) for i in self.exp.sch_idx[~sch_sa_2]])
 
                sch_sa = sch_sa_1 & sch_sa_2
 
                self.sim.sch=self.sim.sch[:,sch_sa]
                self.exp.sch=self.exp.sch[sch_sa]
                self.exp.sch_idx=self.exp.sch_idx[sch_sa]
 
            if ed_shifts_C !=0:
                # Filter sch data
                scc_sa_1 = filter_scc_1(self.sim.scc)
                if np.sum(scc_sa_1) != len(scc_sa_1):
                    print('Warning, following C chemical shifts have too small spread of sim values - omitting them in the optimization.')
                    print([int(i) for i in self.exp.scc_idx[~scc_sa_1]])
 
                scc_sa_2 = filter_scc_2(self.sim.scc,self.exp.scc)
                if np.sum(scc_sa_2) != len(scc_sa_2):
                    print('Warning, following C chemical shifts (atoms) \
                        average values are too far from experimental data!\
                        Ommiting in optimization for now.')
                    print([int(i) for i in self.exp.scc_idx[~scc_sa_2]])
 
                scc_sa = scc_sa_1 & scc_sa_2
 
                self.sim.scc=self.sim.scc[:,scc_sa]
                self.exp.scc=self.exp.scc[scc_sa]
                self.exp.scc_idx=self.exp.scc_idx[scc_sa]


        # put them together - optimization
        self.sim_data_array=[self.sim.freq,self.sim.ram,self.sim.roa,\
                             self.sim.sch,self.sim.scc,self.sim.jij]
        self.exp_data_array=[self.exp.freq,self.exp.ram,self.exp.roa,\
                             self.exp.sch,self.exp.scc,self.exp.jij]

        # put them together - non filtered
        self.nf_sim_data_array=[self.sim.freq,self.sim.ram,self.sim.roa,\
                             self.sim.nf_sch,self.sim.nf_scc,self.sim.nf_jij]
        self.nf_exp_data_array=[self.exp.freq,self.exp.ram,self.exp.roa,\
                             self.exp.nf_sch,self.exp.nf_scc,self.exp.nf_jij]

        # Make Jij_deff_list - what are 13 and what 14

        Jij_atom_list = [i for i in zip(self.exp.nf_jij_idx1,self.exp.nf_jij_idx2)]
        Jij_deff_list = np.zeros(len(Jij_atom_list))
        for idx,i in  enumerate(Jij_atom_list):
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
        self.exp.Jij_deff_list = Jij_deff_list




        print("{0} files loaded.".format(self.sim.n_files))
