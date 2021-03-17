import numpy as np
import os


def extract_nmr_reg(all_data, weights):
    os.makedirs('tmp_files', exist_ok=True)

    # Cheap method
    intercept_H, slope_H = [31.180751027466613, -0.9746151759790808]
    intercept_C, slope_C = [190.0772750438132, -1.0725870103021375]
    Jij_intercept, Jij_slope = [0, 1]

    #################
    data = intercept_H+slope_H*all_data.nf_sim_data_array[3]
    sims = np.dot(data.transpose(), (weights/np.sum(weights)))

    with open('tmp_files/sch_data_regression.dat', 'w') as fw:
        for i in zip(all_data.exp.nf_sch, sims):
            #print("{0:3f}  {1:3f}\n".format(i[0],i[1]))
            fw.write("{0:3f}  {1:3f}\n".format(i[0], i[1]))

    #################
    data = intercept_C+slope_C*all_data.nf_sim_data_array[4]
    sims = np.dot(data.transpose(), (weights/np.sum(weights)))

    with open('tmp_files/scc_data_regression.dat', 'w') as fw:
        for i in zip(all_data.exp.nf_scc, sims):
            #print("{0:3f}  {1:3f}\n".format(i[0],i[1]))
            fw.write("{0:3f}  {1:3f}\n".format(i[0], i[1]))
    #################
    data = Jij_intercept+Jij_slope*all_data.nf_sim_data_array[5]
    sims = np.dot(data.transpose(), (weights/np.sum(weights)))

    with open('tmp_files/jij_data_regression.dat', 'w') as fw:
        for i in zip(all_data.exp.nf_jij, sims):
            #print("{0:3f}  {1:3f}\n".format(i[0],i[1]))
            fw.write("{0:3f}  {1:3f}\n".format(i[0], i[1]))
    # Jij 3
    data = Jij_intercept+Jij_slope*all_data.nf_sim_data_array[5]
    sims = np.dot(data.transpose(), (weights/np.sum(weights)))
    sa = [all_data.exp.Jij_deff_list == 3]

    with open('tmp_files/jij3_data_regression.dat', 'w') as fw:
        for i in zip(all_data.exp.nf_jij[sa], sims[sa]):
            #print("{0:3f}  {1:3f}\n".format(i[0],i[1]))
            fw.write("{0:3f}  {1:3f}\n".format(i[0], i[1]))
    # Jij 4
    data = Jij_intercept+Jij_slope*all_data.nf_sim_data_array[5]
    sims = np.dot(data.transpose(), (weights/np.sum(weights)))
    sa = [all_data.exp.Jij_deff_list == 4]

    with open('tmp_files/jij4_data_regression.dat', 'w') as fw:
        for i in zip(all_data.exp.nf_jij[sa], sims[sa]):
            #print("{0:3f}  {1:3f}\n".format(i[0],i[1]))
            fw.write("{0:3f}  {1:3f}\n".format(i[0], i[1]))
