#import mpl_toolkits.basemap as bm
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Define function which is minimized (Raman) to obtain best overlap
def minimize_sf(sf, model_int, experimental_intensities):
    return (
        np.sum((np.absolute((model_int*sf-experimental_intensities)))**2)
        / np.sum(np.absolute(experimental_intensities))
    )


def S_fg_calculate(model_int, experimental_intensities):
    # For comparing Raman/ROA spectra - return a number between -1,1; 1 beeing the best
    S_fg = np.sum(
        model_int*experimental_intensities) /\
        ((np.sum((model_int)**2)*np.sum((experimental_intensities)**2))**0.5)
    return S_fg


def shift_raman_roa_data(params, freq, ram, roa, exp_ram, exp_roa):
    par_a, par_b, par_c, par_d = params

    ftt = 1

    freq = freq[::ftt]
    ram = ram[::ftt]
    roa = roa[::ftt]
    exp_ram = exp_ram[::ftt]
    exp_roa = exp_roa[::ftt]

    # shift
    sf_array = (par_b*1/(1+np.exp((-freq+par_d)/par_c)) +
                par_a*(1-1/(1+np.exp((-freq+par_d)/par_c))))
    freq_new_tmp = freq*sf_array

    minv = np.min(freq)
    maxv = np.max(freq)

    freq_new = np.zeros(len(freq))
    ram_new = np.zeros(len(freq))
    roa_new = np.zeros(len(freq))
    for j, freq_v in enumerate(np.linspace(minv, maxv, len(freq))):
        ram_new[j] = calculate_point(freq_v, freq_new_tmp, ram)
        roa_new[j] = calculate_point(freq_v, freq_new_tmp, roa)
        freq_new[j] = freq_v

    return ram_new, roa_new

# Calculate gaussian smearing


def calculate_point(x_my, x_array, intensities):
    sigma = 3
    x = np.abs(x_array-x_my)
    point_sum = np.sum(intensities*(1/sigma)*np.exp(-(x**2)/sigma/sigma/2))
    return point_sum


class spectra:
    def __init__(self, weights, all_data, experimental_data_av, **kwargs):

        # expand - what experimental data do we have?
        self.experimental_data_av = experimental_data_av
        ed_raman, ed_roa, ed_shifts_H, ed_shifts_C, ed_spinspin = experimental_data_av

        all_data_transf = all_data
        # Shift Raman/roa
        if ed_raman == 1:
            try:
                sf_raman_params = np.loadtxt(
                    'tmp_files/raman_roa_optimized_scaling_function_parameters.dat')
                if np.shape(sf_raman_params) != (4,):
                    sf_raman_params = sf_raman_params[-1]
            except:
                sf_raman_params = np.array([1, 1, 15, 1000])
            for idx, i in enumerate(all_data_transf.nf_sim_data_array[1]):
                all_data_transf.nf_sim_data_array[1][idx], all_data_transf.nf_sim_data_array[2][idx] = shift_raman_roa_data(
                    sf_raman_params, all_data.nf_exp_data_array[0], all_data.nf_sim_data_array[1][idx], all_data.nf_sim_data_array[2][idx], all_data.nf_exp_data_array[1], all_data.nf_exp_data_array[2])

        self.all_data = all_data_transf

        self.weights = weights

        # assign x axes
        self.freq = all_data.exp.freq
        self.sch_idx = all_data.exp.nf_sch_idx
        self.scc_idx = all_data.exp.nf_scc_idx
        self.jij_idx1 = all_data.exp.nf_jij_idx1
        self.jij_idx2 = all_data.exp.nf_jij_idx2

        # assign rest
        self.exp_sch = all_data.exp.nf_sch
        self.exp_scc = all_data.exp.nf_scc
        self.exp_jij = all_data.exp.nf_jij

        def get_spectra(params, sim_data, exp_data):
            # Load data
            sim_raman_sp = np.dot(
                sim_data[1].transpose(), (params/np.sum(params)))
            sim_roa_sp = np.dot(
                sim_data[2].transpose(), (params/np.sum(params)))

            # Load data NMR
            if sim_data[3].size != 0:
                sim_sch = np.dot(
                    sim_data[3].transpose(), (params/np.sum(params)))
            else:
                sim_sch = np.array([])
            if sim_data[4].size != 0:
                sim_scc = np.dot(
                    sim_data[4].transpose(), (params/np.sum(params)))
            else:
                sim_scc = np.array([])
            if sim_data[5].size != 0:
                sim_jij = np.dot(
                    sim_data[5].transpose(), (params/np.sum(params)))
            else:
                sim_jij = np.array([])

            # normalize exp ram and roa

            ram_exp = np.array(exp_data[1]) / \
                (np.max(np.absolute((exp_data[1])))/10)
            roa_exp = np.array(exp_data[2]) / \
                (np.max(np.absolute((exp_data[2])))/10)

            # 1) overlap them
            if np.sum(sim_raman_sp) != 0:

                f_ram = np.sum(ram_exp)/np.sum(sim_raman_sp)
                res = minimize(minimize_sf, f_ram, method='nelder-mead',
                               options={'maxiter': 10}, args=(sim_raman_sp, ram_exp))
                sc_f1_raman = float(res.x[0])

                f_roa = np.sum(np.absolute(roa_exp)) / \
                    np.sum(np.absolute(sim_roa_sp))
                res = minimize(minimize_sf, f_roa, method='nelder-mead',
                               options={'maxiter': 10}, args=(sim_roa_sp, roa_exp))
                sc_f1_roa = float(res.x[0])
            else:
                sc_f1_raman, sc_f1_roa = [1, 1]

            return sim_raman_sp*sc_f1_raman, \
                sim_roa_sp*sc_f1_roa,\
                sim_sch, sim_scc, sim_jij

        # get sim data
        self.sim_ram, self.sim_roa, self.sim_sch, self.sim_scc, self.sim_jij =\
            get_spectra(weights, all_data_transf.nf_sim_data_array,
                        all_data.nf_exp_data_array)

        # get scaled ram/roa
        self.exp_ram = np.array(
            all_data.exp_data_array[1])/(np.max(np.absolute((all_data.exp_data_array[1])))/10)
        self.exp_roa = np.array(
            all_data.exp_data_array[2])/(np.max(np.absolute((all_data.exp_data_array[2])))/10)

        # calcualte individual errors
        difference = 0
        ram_diff, roa_diff, sch_diff, scc_diff, jij_diff = [0, 0, 0, 0, 0]

        c_ram_0, c_roa_0, c_sch_0, c_scc_0, c_jij_0 = [
            500.0, 71.75, 177.1, 16.83, 18.76]
        c_ram_exp, c_roa_exp, c_sch_exp, c_scc_exp, c_jij_exp = [1, 1, 1, 1, 1]

        S_fg_ram = S_fg_calculate(self.sim_ram, self.exp_ram)
        ram_diff = c_ram_0*((1 - S_fg_ram)**c_ram_exp)
        difference += ram_diff

        S_fg_roa = S_fg_calculate(self.sim_roa, self.exp_roa)
        ram_diff = c_roa_0*((1 - S_fg_roa)**c_roa_exp)
        difference += ram_diff

        # H NMR chemical shifts
        sch_diff = c_sch_0 * \
            np.sum((np.abs(self.sim_sch-self.exp_sch))
                   ** c_sch_exp)/len(self.sim_sch)
        difference += sch_diff

        # C NMR chemical shifts
        scc_diff = c_scc_0 * \
            np.sum((np.abs(self.sim_scc-self.exp_scc))
                   ** c_scc_exp)/len(self.sim_scc)
        difference += scc_diff

        # Jij spin-spin coupling constants
        jij_diff = c_jij_0 * \
            np.sum((np.abs(self.sim_jij-self.exp_jij))
                   ** c_jij_exp)/len(self.sim_jij)
        difference += jij_diff

        self.individual_errors = np.array(
            [ram_diff, roa_diff, sch_diff, scc_diff, jij_diff])
        self.kwargs = kwargs

    def plot(self):

        # expand - what experimental data do we have?
        ed_raman, ed_roa, ed_shifts_H, ed_shifts_C, ed_spinspin = self.experimental_data_av

        # Make nice plots
        import matplotlib as mpl
        from matplotlib import rc

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [
            'Times New Roman'] + plt.rcParams['font.serif']

        figsh = 7.5  # figure height
        figsw = 13  # figure width
        #
        SMALL_SIZE = 24
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 24

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        mpl.rcParams['axes.linewidth'] = 2.0  # set the value globally

        # coloring
        spectra_c1 = 'dodgerblue'
        spectra_c2 = 'r'

        ##########################################################################

        # Subplots Vibrational spectroscopy

        # set Raman/ROA frequency x values
        minv = np.min(self.freq)
        maxv = np.max(self.freq)

        f, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [
                             1, 1]}, figsize=(figsw, figsh))
        plt.subplots_adjust(hspace=0)

        # raman
        ind_sp_scaling = 1.0
        ax[0].plot([minv, maxv], [0, 0], 'k', linewidth=4.0)
        for i in range(len(self.all_data.sim_data_array[0])):
            ax[0].plot(self.freq, self.all_data.sim_data_array[1]
                       [i]/ind_sp_scaling, color='k', linewidth=0.15)
        g1, = ax[0].plot(self.freq, self.exp_ram, linewidth=4.0,
                         label='exp', color=spectra_c1)
        g2, = ax[0].plot(self.freq, self.sim_ram, linewidth=4.0,
                         label='sim', color=spectra_c2)
        ax[0].legend(handles=[g1, g2], loc=1,
                     fancybox=True).get_frame().set_alpha(0)
        wtp = '$\mathregular{S_{fg}=}$' + \
            str('{0:.3f}'.format(S_fg_calculate(self.exp_ram, self.sim_ram)))
        ax[0].text(0.985, 0.05, wtp, transform=ax[0].transAxes,
                   verticalalignment='bottom', horizontalalignment='right')
        ax[0].set_xlim([minv, maxv])
        ax[0].set_ylim([-0.5, 12])
        ax[0].set_yticks([0, 4, 8])
        ax[0].tick_params(axis='x', which='both', bottom=False,
                          top=False, labelbottom=False)
        ax[0].set_ylabel('$\mathregular{I_R+I_L [a.u.]}$')
        ax[0].xaxis.grid()  # vertical lines

        # ROA
        ind_sp_scaling = 3.0
        ax[1].plot([minv, maxv], [0, 0], 'k', linewidth=4.0)
        for i in range(len(self.all_data.sim_data_array[0])):
            ax[1].plot(self.freq, self.all_data.sim_data_array[2]
                       [i]/ind_sp_scaling, color='k', linewidth=0.15)
        g1, = ax[1].plot(self.freq, self.exp_roa, linewidth=4.0,
                         label='exp', color=spectra_c1)
        g2, = ax[1].plot(self.freq, self.sim_roa, linewidth=4.0,
                         label='sim', color=spectra_c2)
        wtp = '$\mathregular{S_{fg}=}$' + \
            str('{0:.3f}'.format(S_fg_calculate(self.exp_roa, self.sim_roa)))
        ax[1].text(0.985, 0.05, wtp, transform=ax[1].transAxes,
                   verticalalignment='bottom', horizontalalignment='right')
        ax[1].legend(handles=[g1, g2], loc=1,
                     fancybox=True).get_frame().set_alpha(0)
        ax[1].set_xlim([minv, maxv])
        ax[1].set_ylim([-12, 12])
        ax[1].set_yticks([-8, 0, 8])

        ax[1].set_ylabel('$\mathregular{I_R-I_L [a.u.]}$')
        ax[1].xaxis.grid()  # vertical lines
        ax[1].set_xlabel('$\mathregular{\~\\nu \ [cm^{-1}]}$')

        plt.savefig('figures/RamanROA{0}.png'.format(self.all_data.iteration),
                    format='png', transparent=True, bbox_inches='tight')
        plt.savefig('figures/RamanROA{0}.pdf'.format(self.all_data.iteration),
                    format='pdf', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()

        ##########################################################################
        # NMR data - H shifts
        figsh = 5
        figsw = 20

        f, a1 = plt.subplots(1, 1, figsize=(figsw, figsh))
        vp1 = a1.violinplot(self.all_data.nf_sim_data_array[3], positions=np.arange(
            np.shape(self.all_data.nf_sim_data_array[3])[1]), showextrema=False, widths=0.75)
        sa = self.weights > 0
        atp = self.all_data.nf_sim_data_array[3][sa]

        ws = self.weights[sa]
        ws = ws/np.max(ws)*100

        atp_n = []
        for i in zip(atp, ws):
            for j in np.arange(int(i[1])):
                atp_n.append(list(i[0]))
        atp_n = np.array(atp_n)

        vp2 = a1.violinplot(atp_n, positions=np.arange(np.shape(
            self.all_data.nf_sim_data_array[3])[1]), showextrema=False, widths=0.5)

        for pc in vp1['bodies']:
            # c.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        for pc in vp2['bodies']:
            # pc.set_facecolor('r')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        ###
        g1, = a1.plot(np.arange(len(self.sch_idx)), self.exp_sch, '--x',
                      color=spectra_c1, ms=20, linewidth=2, mew=4, label='exp')
        g2, = a1.plot(np.arange(len(self.sch_idx)), self.sim_sch, '--x',
                      color=spectra_c2, ms=20, linewidth=2, mew=4, label='sim')
        a1.legend(handles=[g1, g2], loc=0,
                  fancybox=True).get_frame().set_alpha(0)
        tl = [(int(i)) for i in self.sch_idx]
        a1.set_xticks(np.arange(len(tl)))
        a1.set_xticklabels(tl, rotation=90)
        a1.set_ylabel('$\mathregular{^1H\ [ppm] }$')

        plt.savefig('figures/Hshifts{0}.png'.format(self.all_data.iteration),
                    format='png', transparent=True, bbox_inches='tight')
        plt.savefig('figures/Hshifts{0}.pdf'.format(self.all_data.iteration),
                    format='pdf', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()

        ##########################################################################
        # NMR data - differences
        figsh = 5
        figsw = 20

        f, a1 = plt.subplots(1, 1, figsize=(figsw, figsh))
        ###
        vp1 = a1.violinplot(self.all_data.nf_sim_data_array[3]-self.all_data.nf_exp_data_array[3], positions=np.arange(
            np.shape(self.all_data.nf_sim_data_array[3])[1]), showextrema=False, widths=0.75)
        sa = self.weights > 0
        atp = self.all_data.nf_sim_data_array[3][sa] - \
            self.all_data.nf_exp_data_array[3]

        ws = self.weights[sa]
        ws = ws/np.max(ws)*100

        atp_n = []
        for i in zip(atp, ws):
            for j in np.arange(int(i[1])):
                atp_n.append(list(i[0]))
        atp_n = np.array(atp_n)

        vp2 = a1.violinplot(atp_n, positions=np.arange(np.shape(
            self.all_data.nf_sim_data_array[3])[1]), showextrema=False, widths=0.5)

        for pc in vp1['bodies']:
            # c.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        for pc in vp2['bodies']:
            # pc.set_facecolor('r')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        ###
        g1, = a1.plot(np.arange(len(self.sch_idx)), self.exp_sch-self.exp_sch,
                      '--x', color=spectra_c1, ms=20, linewidth=2, mew=4, label='exp')
        g2, = a1.plot(np.arange(len(self.sch_idx)), self.sim_sch-self.exp_sch,
                      '--x', color=spectra_c2, ms=20, linewidth=2, mew=4, label='sim')
        a1.legend(handles=[g1, g2], loc=0,
                  fancybox=True).get_frame().set_alpha(0)
        a1.set_ylim([-1, 1])
        tl = [(int(i)) for i in self.sch_idx]
        a1.set_xticks(np.arange(len(tl)))
        a1.set_xticklabels(tl, rotation=90)
        a1.set_ylabel('$\mathregular{^1H\ \delta\ [ppm] }$')

        plt.savefig('figures/HshiftsDiff{0}.png'.format(
            self.all_data.iteration), format='png', transparent=True, bbox_inches='tight')
        plt.savefig('figures/HshiftsDiff{0}.pdf'.format(
            self.all_data.iteration), format='pdf', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()

        figsh = 5
        figsw = 20

        f, a2 = plt.subplots(1, 1, figsize=(figsw, figsh))

        vp1 = a2.violinplot(self.all_data.nf_sim_data_array[4], positions=np.arange(
            np.shape(self.all_data.nf_sim_data_array[4])[1]), showextrema=False, widths=0.75)
        sa = self.weights > 0
        atp = self.all_data.nf_sim_data_array[4][sa]

        ws = self.weights[sa]
        ws = ws/np.max(ws)*100

        atp_n = []
        for i in zip(atp, ws):
            for j in np.arange(int(i[1])):
                atp_n.append(list(i[0]))
        atp_n = np.array(atp_n)

        vp2 = a2.violinplot(atp_n, positions=np.arange(np.shape(
            self.all_data.nf_sim_data_array[4])[1]), showextrema=False, widths=0.5)

        for pc in vp1['bodies']:
            # c.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        for pc in vp2['bodies']:
            # pc.set_facecolor('r')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        ###
        g1, = a2.plot(self.exp_scc, '--x', color=spectra_c1,
                      ms=20, linewidth=2, mew=4, label='exp')
        g2, = a2.plot(self.sim_scc, '--x', color=spectra_c2,
                      ms=20, linewidth=2, mew=4, label='sim')
        a2.legend(handles=[g1, g2], loc=0,
                  fancybox=True).get_frame().set_alpha(0)
        # a2.set_ylim([-10+np.min(self.all_data.sim_data_array[4]),10+np.max(self.all_data.sim_data_array[4])])
        tl = [(int(i)) for i in self.scc_idx]
        a2.set_xticks(np.arange(len(tl)))
        a2.set_xticklabels(tl, rotation=90)
        a2.set_ylabel('$\mathregular{^{13}C\ [ppm] }$')

        plt.savefig('figures/Cshifts{0}.png'.format(self.all_data.iteration),
                    format='png', transparent=True, bbox_inches='tight')
        plt.savefig('figures/Cshifts{0}.pdf'.format(self.all_data.iteration),
                    format='pdf', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()

        ##########################################################################
        # NMR data - differences
        figsh = 5
        figsw = 20

        f, a2 = plt.subplots(1, 1, figsize=(figsw, figsh))

        vp1 = a2.violinplot(self.all_data.nf_sim_data_array[4]-self.all_data.nf_exp_data_array[4], positions=np.arange(
            np.shape(self.all_data.nf_sim_data_array[4])[1]), showextrema=False, widths=0.75)
        sa = self.weights > 0
        atp = self.all_data.nf_sim_data_array[4][sa] - \
            self.all_data.nf_exp_data_array[4]

        ws = self.weights[sa]
        ws = ws/np.max(ws)*100

        atp_n = []
        for i in zip(atp, ws):
            for j in np.arange(int(i[1])):
                atp_n.append(list(i[0]))
        atp_n = np.array(atp_n)

        vp2 = a2.violinplot(atp_n, positions=np.arange(np.shape(
            self.all_data.nf_sim_data_array[4])[1]), showextrema=False, widths=0.5)

        for pc in vp1['bodies']:
            # c.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        for pc in vp2['bodies']:
            # pc.set_facecolor('r')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        ###
        g1, = a2.plot(self.exp_scc-self.exp_scc, '--x',
                      color=spectra_c1, ms=20, linewidth=2, mew=4, label='exp')
        g2, = a2.plot(self.sim_scc-self.exp_scc, '--x',
                      color=spectra_c2, ms=20, linewidth=2, mew=4, label='sim')
        a2.legend(handles=[g1, g2], loc=0,
                  fancybox=True).get_frame().set_alpha(0)
        a2.set_ylim([-10, 10])
        tl = [(int(i)) for i in self.scc_idx]
        a2.set_xticks(np.arange(len(tl)))
        a2.set_xticklabels(tl, rotation=90)
        a2.set_ylabel('$\mathregular{^{13}C\ \delta\ [ppm] }$')

        plt.savefig('figures/CshiftsDiff{0}.png'.format(
            self.all_data.iteration), format='png', transparent=True, bbox_inches='tight')
        plt.savefig('figures/CshiftsDiff{0}.pdf'.format(
            self.all_data.iteration), format='pdf', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()

        figsh = 5
        figsw = 20

        f, a3 = plt.subplots(1, 1, figsize=(figsw, figsh))

        vp1 = a3.violinplot(self.all_data.nf_sim_data_array[5], positions=np.arange(
            np.shape(self.all_data.nf_sim_data_array[5])[1]), showextrema=False, widths=0.75)
        sa = self.weights > 0
        atp = self.all_data.nf_sim_data_array[5][sa]

        ws = self.weights[sa]
        ws = ws/np.max(ws)*100

        atp_n = []
        for i in zip(atp, ws):
            for j in np.arange(int(i[1])):
                atp_n.append(list(i[0]))
        atp_n = np.array(atp_n)

        vp2 = a3.violinplot(atp_n, positions=np.arange(np.shape(
            self.all_data.nf_sim_data_array[5])[1]), showextrema=False, widths=0.5)

        for pc in vp1['bodies']:
            # c.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        for pc in vp2['bodies']:
            # pc.set_facecolor('r')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        ###
        tl = [str(int(i[0]))+','+str(int(i[1]))
              for i in zip(self.jij_idx1, self.jij_idx2)]
        g1, = a3.plot(np.arange(len(tl)), self.exp_jij, '--x',
                      color=spectra_c1, ms=20, linewidth=2, mew=4, label='exp')
        g2, = a3.plot(np.arange(len(tl)), (self.sim_jij), '--x',
                      color=spectra_c2, ms=20, linewidth=2, mew=4, label='sim')
        a3.legend(handles=[g1, g2], loc=0,
                  fancybox=True).get_frame().set_alpha(0)
        # a3.set_ylim([-2,14])
        a3.set_xticks(np.arange(len(tl)))
        a3.set_xticklabels(tl, rotation=90)
        a3.set_ylabel('$\mathregular{J_{ij}\ [Hz] }$')

        plt.savefig('figures/Jij{0}.png'.format(self.all_data.iteration),
                    format='png', transparent=True, bbox_inches='tight')
        plt.savefig('figures/Jij{0}.pdf'.format(self.all_data.iteration),
                    format='pdf', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()

        ##########################################################################
        # NMR data - differences
        figsh = 5
        figsw = 20

        f, a3 = plt.subplots(1, 1, figsize=(figsw, figsh))

        vp1 = a3.violinplot(self.all_data.nf_sim_data_array[5]-self.all_data.nf_exp_data_array[5], positions=np.arange(
            np.shape(self.all_data.nf_sim_data_array[5])[1]), showextrema=False, widths=0.75)
        sa = self.weights > 0
        atp = self.all_data.nf_sim_data_array[5][sa] - \
            self.all_data.nf_exp_data_array[5]

        ws = self.weights[sa]
        ws = ws/np.max(ws)*100

        atp_n = []
        for i in zip(atp, ws):
            for j in np.arange(int(i[1])):
                atp_n.append(list(i[0]))
        atp_n = np.array(atp_n)

        vp2 = a3.violinplot(atp_n, positions=np.arange(np.shape(
            self.all_data.nf_sim_data_array[5])[1]), showextrema=False, widths=0.5)

        for pc in vp1['bodies']:
            # c.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        for pc in vp2['bodies']:
            # pc.set_facecolor('r')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        ###
        tl = [str(int(i[0]))+','+str(int(i[1]))
              for i in zip(self.jij_idx1, self.jij_idx2)]
        g1, = a3.plot(np.arange(len(tl)), self.exp_jij-self.exp_jij, '--x',
                      color=spectra_c1, ms=20, linewidth=2, mew=4, label='exp')
        g2, = a3.plot(np.arange(len(tl)), (self.sim_jij-self.exp_jij),
                      '--x', color=spectra_c2, ms=20, linewidth=2, mew=4, label='sim')
        a3.legend(handles=[g1, g2], loc=0,
                  fancybox=True).get_frame().set_alpha(0)
        a3.set_ylim([-5, 5])
        a3.set_xticks(np.arange(len(tl)))
        a3.set_xticklabels(tl, rotation=90)
        a3.set_ylabel('$\mathregular{\delta J_{ij}\ [Hz] }$')

        plt.savefig('figures/JijDiff{0}.png'.format(self.all_data.iteration),
                    format='png', transparent=True, bbox_inches='tight')
        plt.savefig('figures/JijDiff{0}.pdf'.format(self.all_data.iteration),
                    format='pdf', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()

    def plot_raman(self):
        try:
            if self.kwargs['scale_roa_to_exp'] == True:
                roa_scale_f = 1/(np.max(np.abs(self.sim_roa)) /
                                 np.max(np.abs(self.exp_roa)))
        except:
            roa_scale_f = 1

        try:
            ramanroa_frequency_range = self.kwargs['ramanroa_frequency_range']
        except:
            ramanroa_frequency_range = [np.min(self.freq), np.max(self.freq)]

        # set Raman/ROA frequency x values
        minv, maxv = ramanroa_frequency_range

        # expand - what experimental data do we have?
        ed_raman, ed_roa, ed_shifts_H, ed_shifts_C, ed_spinspin = self.experimental_data_av

        # Make nice plots
        ##########################################################################
        import matplotlib as mpl
        from matplotlib import rc

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [
            'Times New Roman'] + plt.rcParams['font.serif']

        figsh = 7  # figure height
        figsw = 13  # figure width
        #
        SMALL_SIZE = 24
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 24

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        mpl.rcParams['axes.linewidth'] = 2.0  # set the value globally

        # coloring
        spectra_c1 = 'dodgerblue'
        spectra_c2 = 'r'

        ##########################################################################

        # Subplots Vibrational spectroscopy

        f, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [
                             1, 1]}, figsize=(figsw, figsh))
        plt.subplots_adjust(hspace=0)

        # raman
        ind_sp_scaling = 1.0
        ax[0].plot([minv, maxv], [0, 0], 'k', linewidth=4.0)
#        for i in range(len(self.all_data.sim_data_array[0])):
#            ax[0].plot(self.freq,self.all_data.sim_data_array[1][i]/ind_sp_scaling,color='k',linewidth=0.15)
        g1, = ax[0].plot(self.freq, self.exp_ram, linewidth=4.0,
                         label='exp', color=spectra_c1)
        g2, = ax[0].plot(self.freq, self.sim_ram, linewidth=4.0,
                         label='sim', color=spectra_c2)
        ax[0].legend(handles=[g1, g2], loc=1,
                     fancybox=True).get_frame().set_alpha(0)
        wtp = '$\mathregular{S_{fg}=}$' + \
            str('{0:.3f}'.format(S_fg_calculate(self.exp_ram, self.sim_ram)))
        ax[0].text(0.16, 0.78, wtp, transform=ax[0].transAxes,
                   verticalalignment='bottom', horizontalalignment='right')
        ax[0].set_xlim([minv, maxv])
        ax[0].set_ylim([-0.5, 12])
        ax[0].set_yticks([0, 4, 8])
        ax[0].tick_params(axis='x', which='both', bottom=False,
                          top=False, labelbottom=False)
        ax[0].set_ylabel('$\mathregular{I_R+I_L [a.u.]}$')
        ax[0].xaxis.grid()  # vertical lines

        # ROA
        ind_sp_scaling = 3.0
        ax[1].plot([minv, maxv], [0, 0], 'k', linewidth=4.0)
#        for i in range(len(self.all_data.sim_data_array[0])):
#            ax[1].plot(self.freq,self.all_data.sim_data_array[2][i]/ind_sp_scaling,color='k',linewidth=0.15)
        g1, = ax[1].plot(self.freq, self.exp_roa, linewidth=4.0,
                         label='exp', color=spectra_c1)
        g2, = ax[1].plot(self.freq, self.sim_roa*roa_scale_f,
                         linewidth=4.0, label='sim', color=spectra_c2)
        wtp = '$\mathregular{S_{fg}=}$' + \
            str('{0:.3f}'.format(S_fg_calculate(self.exp_roa, self.sim_roa)))
        ax[1].text(0.16, 0.78, wtp, transform=ax[1].transAxes,
                   verticalalignment='bottom', horizontalalignment='right')
        ax[1].legend(handles=[g1, g2], loc=1,
                     fancybox=True).get_frame().set_alpha(0)
        ax[1].set_xlim([minv, maxv])
        ax[1].set_ylim([-12, 12])
        ax[1].set_yticks([-8, 0, 8])

        ax[1].set_ylabel('$\mathregular{I_R-I_L [a.u.]}$')
        ax[1].xaxis.grid()  # vertical lines
        ax[1].set_xlabel('$\mathregular{\~\\nu \ [cm^{-1}]}$')

        plt.savefig("figures/f_ramroa_it{0}.pdf".format(
            self.all_data.iteration), format='pdf', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()

    def plot_nmr_shifts_h(self):

        # Make nice plots
        ##########################################################################
        import matplotlib as mpl
        from matplotlib import rc

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [
            'Times New Roman'] + plt.rcParams['font.serif']

        figsh = 2  # figure height
        figsw = 13  # figure width
        #
        SMALL_SIZE = 24
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 24

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        mpl.rcParams['axes.linewidth'] = 2.0  # set the value globally

        # coloring
        spectra_c1 = 'dodgerblue'
        spectra_c2 = 'r'

        ##########################################################################

        # expand - what experimental data do we have?
        ed_raman, ed_roa, ed_shifts_H, ed_shifts_C, ed_spinspin = self.experimental_data_av
        f, a1 = plt.subplots(1, 1, figsize=(figsw, figsh))

        g1, = a1.plot(np.arange(len(self.sch_idx)), self.exp_sch-self.exp_sch,
                      '--x', color=spectra_c1, ms=20, linewidth=2, mew=4, label='exp')
        g2, = a1.plot(np.arange(len(self.sch_idx)), self.sim_sch-self.exp_sch,
                      '--x', color=spectra_c2, ms=20, linewidth=2, mew=4, label='sim')
#         a1.legend(handles=[g1, g2],loc=0,fancybox=True,ncol = 2).get_frame().set_alpha(0)

        ymin, ymax = [-0.5, 0.5]
        a1.set_ylim([ymin, ymax])
        a1.set_yticks(np.linspace(ymin, ymax, 3))
        a1.set_ylabel('$\mathregular{\Delta^1H\ [ppm] }$')
        #         a1.set_ylabel('$\mathregular{^1H_{sim}-^1H_{exp}\ [ppm] }$')

        xmin, xmax = [-0.25+np.min(np.arange(len(self.exp_sch))),
                      0.25+np.max(np.arange(len(self.exp_sch)))]
        a1.set_xlim([xmin, xmax])

        tl = [(int(i)) for i in self.sch_idx]
        a1.set_xticks(np.arange(len(tl)))
        a1.set_xticklabels(tl, rotation=0)

        MAE = np.average(np.abs(self.sim_sch-self.exp_sch))
        wtp = 'MAE = {0:.3f} ppm'.format(MAE)
        a1.text(0.92, 0.1, wtp, transform=a1.transAxes,
                verticalalignment='bottom', horizontalalignment='right')

        # plot grid
        for i in np.linspace(ymin, ymax, 5):
            a1.plot([xmin, xmax], [i, i], '-', color='grey', linewidth=0.4)

        plt.savefig("figures/f_sch_it{0}.pdf".format(self.all_data.iteration),
                    format='pdf', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()

    def plot_nmr_shifts_c(self):

        # Make nice plots
        ##########################################################################
        import matplotlib as mpl
        from matplotlib import rc

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [
            'Times New Roman'] + plt.rcParams['font.serif']

        figsh = 2  # figure height
        figsw = 13  # figure width
        #
        SMALL_SIZE = 24
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 24

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        mpl.rcParams['axes.linewidth'] = 2.0  # set the value globally

        # coloring
        spectra_c1 = 'dodgerblue'
        spectra_c2 = 'r'

        ##########################################################################

        # expand - what experimental data do we have?
        ed_raman, ed_roa, ed_shifts_H, ed_shifts_C, ed_spinspin = self.experimental_data_av

        f, a2 = plt.subplots(1, 1, figsize=(figsw, figsh))

        g1, = a2.plot(self.exp_scc-self.exp_scc, '--x',
                      color=spectra_c1, ms=20, linewidth=2, mew=4, label='exp')
        g2, = a2.plot(self.sim_scc-self.exp_scc, '--x',
                      color=spectra_c2, ms=20, linewidth=2, mew=4, label='sim')
#         a2.legend(handles=[g1, g2],loc=0,fancybox=True,ncol = 2).get_frame().set_alpha(0)

        ymin, ymax = [-8, 8]
        a2.set_ylim([ymin, ymax])
        a2.set_yticks(np.linspace(ymin, ymax, 3))
#         a2.set_ylabel('$\mathregular{^{13}C_{sim}-^{13}C_{exp}\ [ppm] }$')
        a2.set_ylabel('$\mathregular{\Delta^{13}C\ [ppm] }$')

        xmin, xmax = [-0.25+np.min(np.arange(len(self.exp_scc))),
                      0.25+np.max(np.arange(len(self.exp_scc)))]
        a2.set_xlim([xmin, xmax])
        tl = [(int(i)) for i in self.scc_idx]
        a2.set_xticks(np.arange(len(tl)))
        a2.set_xticklabels(tl, rotation=0)

        MAE = np.average(np.abs(self.sim_scc-self.exp_scc))
        wtp = 'MAE = {0:.2f} ppm'.format(MAE)
        a2.text(0.92, 0.1, wtp, transform=a2.transAxes,
                verticalalignment='bottom', horizontalalignment='right')

        # plot grid
        for i in np.linspace(ymin, ymax, 5):
            a2.plot([xmin, xmax], [i, i], '-', color='grey', linewidth=0.4)

        plt.savefig("figures/f_scc_it{0}.pdf".format(self.all_data.iteration),
                    format='pdf', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()

    def plot_nmr_spinspin(self):

        # Make nice plots
        ##########################################################################
        import matplotlib as mpl
        from matplotlib import rc

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [
            'Times New Roman'] + plt.rcParams['font.serif']

        figsh = 2  # figure height
        figsw = 13  # figure width
        #
        SMALL_SIZE = 24
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 24

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        mpl.rcParams['axes.linewidth'] = 2.0  # set the value globally

        # coloring
        spectra_c1 = 'dodgerblue'
        spectra_c2 = 'r'

        ##########################################################################

        # expand - what experimental data do we have?
        ed_raman, ed_roa, ed_shifts_H, ed_shifts_C, ed_spinspin = self.experimental_data_av

        f, a3 = plt.subplots(1, 1, figsize=(figsw, figsh))

        tl = [str(int(i[0]))+','+str(int(i[1]))
              for i in zip(self.jij_idx1, self.jij_idx2)]
        g1, = a3.plot(np.arange(len(tl)), self.exp_jij-self.exp_jij, '--x',
                      color=spectra_c1, ms=20, linewidth=2, mew=4, label='exp')
        g2, = a3.plot(np.arange(len(tl)), (self.sim_jij-self.exp_jij),
                      '--x', color=spectra_c2, ms=20, linewidth=2, mew=4, label='sim')
#         a3.legend(handles=[g1, g2],loc=0,fancybox=True,ncol = 2).get_frame().set_alpha(0)

        ymin, ymax = [-5, 5]
        a3.set_ylim([ymin, ymax])
        a3.set_yticks(np.linspace(ymin, ymax, 3))
#         a3.set_ylabel('$\mathregular{J_{ij,sim}-J_{ij,exp}\ [Hz] }$')
        a3.set_ylabel('$\mathregular{\Delta J_{ij}\ [Hz] }$')

        xmin, xmax = [-0.25+np.min(np.arange(len(self.exp_jij))),
                      0.25+np.max(np.arange(len(self.exp_jij)))]
        a3.set_xlim([xmin, xmax])

        a3.set_xticks(np.arange(len(tl)))
        a3.set_xticklabels(tl, rotation=90)

        MAE = np.average(np.abs(self.sim_jij-self.exp_jij))
        wtp = 'MAE = {0:.2f} Hz'.format(MAE)
        a3.text(0.92, 0.1, wtp, transform=a3.transAxes,
                verticalalignment='bottom', horizontalalignment='right')

        # plot grid
        for i in np.linspace(ymin, ymax, 5):
            a3.plot([xmin, xmax], [i, i], '-', color='grey', linewidth=0.4)

        plt.savefig("figures/f_jij_it{0}.pdf".format(self.all_data.iteration),
                    format='pdf', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()

    def plot_Jij_UJ_peptide(self, **kwargs):

        try:
            fig_name = kwargs['fig_name']
        except:
            fig_name = 'it0'

        # Make nice plots
        ##########################################################################
        import matplotlib as mpl
        from matplotlib import rc

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [
            'Times New Roman'] + plt.rcParams['font.serif']

        SMALL_SIZE = 24
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 24

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        mpl.rcParams['axes.linewidth'] = 2.0  # set the value globally

        # coloring
        spectra_c1 = 'dodgerblue'
        spectra_c2 = 'r'

        figsh = 5
        figsw = 20

        f, a3 = plt.subplots(1, 1, figsize=(figsw, figsh))

        vp1 = a3.violinplot(self.all_data.nf_sim_data_array[5], positions=np.arange(
            np.shape(self.all_data.nf_sim_data_array[5])[1]), showextrema=False, widths=0.75)
        sa = self.weights > 0
        atp = self.all_data.nf_sim_data_array[5][sa]

        ws = self.weights[sa]
        ws = ws/np.max(ws)*100

        atp_n = []
        for i in zip(atp, ws):
            for j in np.arange(int(i[1])):
                atp_n.append(list(i[0]))
        atp_n = np.array(atp_n)

        try:
            if kwargs['hide_picked_w_vp'] == True:
                pass
        except:
            vp2 = a3.violinplot(atp_n, positions=np.arange(np.shape(
                self.all_data.nf_sim_data_array[5])[1]), showextrema=False, widths=0.5)
            for pc in vp2['bodies']:
                # pc.set_facecolor('r')
                pc.set_edgecolor('black')
                pc.set_alpha(0.5)

        for pc in vp1['bodies']:
            # c.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        ###
        # x axis - atoms numbers
#        tl = [str(int(i[0]))+','+str(int(i[1])) for i in zip(self.jij_idx1,self.jij_idx2)]
        # x axis - J_i
        tl = [r'$J_{{{0}}}$'.format(i+1) for i in range(len(self.jij_idx1))]

        g1, = a3.plot(np.arange(len(tl)), self.exp_jij, '--x',
                      color=spectra_c1, ms=20, linewidth=2, mew=4, label='exp')
        g2, = a3.plot(np.arange(len(tl)), (self.sim_jij), '--x',
                      color=spectra_c2, ms=20, linewidth=2, mew=4, label='sim')
        a3.legend(handles=[g1, g2], loc=0,
                  fancybox=True).get_frame().set_alpha(0)
        # a3.set_ylim([-2,14])
        a3.set_xticks(np.arange(len(tl)))
        # a3.set_xticklabels(tl,rotation=90)
        a3.set_xticklabels(tl)
        a3.set_ylabel('$\mathregular{J_{ij}\ [Hz] }$')

        MAE = np.average(np.abs(self.sim_jij-self.exp_jij))
        wtp = 'MAE = {0:.2f} Hz'.format(MAE)
        a3.text(0.15, 0.85, wtp, transform=a3.transAxes,
                verticalalignment='bottom', horizontalalignment='right')

        plt.savefig("figures/UJ_jij_{0}.pdf".format(fig_name),
                    format='pdf', transparent=True, bbox_inches='tight')
        plt.savefig("figures/UJ_jij_{0}.png".format(fig_name),
                    format='png', transparent=True, bbox_inches='tight')

        plt.show()
        plt.close()
