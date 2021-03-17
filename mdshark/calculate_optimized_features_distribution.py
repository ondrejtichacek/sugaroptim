import numpy as np


# This class calculates the new distribution of extracted features using new optimized weights
class calculate_new_distribution:
    def __init__(self, weights, sim_data, molecule_features, **kwargs):

        # def some constants
        R = 8.314459
        temp_targeted_MD = 300

        # set what % to sample according to optimized distribution, rest will be random
        # example:  K = 0.75 i.e. sample 75 % according to the new dist + 25 % randomly
        try:
            K = kwargs['K']
        except:
            K = 0.75

        # calculate smeared point using periodic angle distance - dihedral angles/puckering phi
        def calcg(x_my, x_array, weights_array):
            sigma = 2*np.pi/25
            x = np.amin([np.abs(x_array-x_my), 2*np.pi -
                        np.abs(x_array-x_my)], axis=0)
            point_sum = np.sum(weights_array*np.exp(-(x**2)/2/sigma/sigma))
            return point_sum

        # calculate smeared point puckering - theta
        def calcg_theta(x_my, x_array, weights_array):
            sigma = 2*np.pi/40
            x = np.amin([np.abs(x_array-x_my), 2*np.pi -
                        np.abs(x_array-x_my)], axis=0)
            point_sum = np.sum(weights_array*np.exp(-(x**2)/2/sigma/sigma))
            return point_sum

        # calculate new population + fes + derivation of fes - dihedral angle
        def calc_pop_fes_grad_dih(feature, weights):
            x = np.linspace(-np.pi, np.pi, 300)
            y_opt = np.zeros(300)
            y_old = np.zeros(300)
            for idx, j in enumerate(x):
                y_opt[idx] = calcg(j, feature, weights)
                y_old[idx] = calcg(j, feature, np.ones(len(weights)))
            y_new = y_opt/abs(np.sum(y_opt)*(x[0]-x[1]))
            y_old = y_old/abs(np.sum(y_old)*(x[0]-x[1]))

            dx = x[-1] - x[-2]
            sy = np.linspace(0, np.max(y_new), 1000)

            Po = np.array([np.sum((y_new[y_new > i]-i)*dx) for i in sy])
            Pa = np.array([i*len(x)*dx for i in sy])
            idx_min = np.argmin([np.abs(Po/(Po+Pa)-K) for i in sy])

            K_opt = sy[idx_min]
            y_new[y_new < K_opt] = K_opt

            y_new = y_new/abs(np.sum(y_new)*(x[0]-x[1]))

            yfes = -R*temp_targeted_MD*np.log(y_new)/1000
            yfes = yfes-np.min(yfes)
            ygrad = np.gradient(yfes, x)

            return x, y_opt, y_old, y_new, yfes, ygrad

        # calculate new population + fes + derivation of fes - puckering - phi
        def calc_pop_fes_grad_puck_phi(feature, weights):
            x = np.linspace(0, 2*np.pi, 300)
            y_opt = np.zeros(300)
            y_old = np.zeros(300)
            for idx, j in enumerate(x):
                y_opt[idx] = calcg(j, feature, weights)
                y_old[idx] = calcg(j, feature, np.ones(len(weights)))
            y_new = y_opt/abs(np.sum(y_opt)*(x[0]-x[1]))
            y_old = y_old/abs(np.sum(y_old)*(x[0]-x[1]))

            dx = x[-1] - x[-2]
            sy = np.linspace(0, np.max(y_new), 1000)

            Po = np.array([np.sum((y_new[y_new > i]-i)*dx) for i in sy])
            Pa = np.array([i*len(x)*dx for i in sy])
            idx_min = np.argmin([np.abs(Po/(Po+Pa)-K) for i in sy])

            K_opt = sy[idx_min]
            y_new[y_new < K_opt] = K_opt

            y_new = y_new/abs(np.sum(y_new)*(x[0]-x[1]))

            yfes = -R*temp_targeted_MD*np.log(y_new)/1000
            yfes = yfes-np.min(yfes)
            ygrad = np.gradient(yfes, x)

            return x, y_opt, y_old, y_new, yfes, ygrad

        # calculate new population + fes + derivation of fes - puckering - theta
        def calc_pop_fes_grad_puck_theta(feature, weights):
            x = np.linspace(0, np.pi, 300)
            y_opt = np.zeros(300)
            y_old = np.zeros(300)
            for idx, j in enumerate(x):
                y_opt[idx] = calcg_theta(j, feature, weights)
                y_old[idx] = calcg_theta(j, feature, np.ones(len(weights)))
            y_new = y_opt/abs(np.sum(y_opt)*(x[0]-x[1]))
            y_old = y_old/abs(np.sum(y_old)*(x[0]-x[1]))

            dx = x[-1] - x[-2]
            sy = np.linspace(0, np.max(y_new), 1000)

            Po = np.array([np.sum((y_new[y_new > i]-i)*dx) for i in sy])
            Pa = np.array([i*len(x)*dx for i in sy])
            idx_min = np.argmin([np.abs(Po/(Po+Pa)-K) for i in sy])

            K_opt = sy[idx_min]
            y_new[y_new < K_opt] = K_opt

            y_new = y_new/abs(np.sum(y_new)*(x[0]-x[1]))

            yfes = -R*temp_targeted_MD*np.log(y_new)/1000
            yfes = yfes-np.min(yfes)
            ygrad = np.gradient(yfes, x)

            return x, y_opt, y_old, y_new, yfes, ygrad

        # calculate new population + fes + derivation of fes - puckering5 - phs
        def calc_pop_fes_grad_puck5_phs(feature, weights):
            x = np.linspace(-np.pi, np.pi, 300)
            y_opt = np.zeros(300)
            y_old = np.zeros(300)
            for idx, j in enumerate(x):
                y_opt[idx] = calcg(j, feature, weights)
                y_old[idx] = calcg(j, feature, np.ones(len(weights)))
            y_new = y_opt/abs(np.sum(y_opt)*(x[0]-x[1]))
            y_old = y_old/abs(np.sum(y_old)*(x[0]-x[1]))

            dx = x[-1] - x[-2]
            sy = np.linspace(0, np.max(y_new), 1000)

            Po = np.array([np.sum((y_new[y_new > i]-i)*dx) for i in sy])
            Pa = np.array([i*len(x)*dx for i in sy])
            idx_min = np.argmin([np.abs(Po/(Po+Pa)-K) for i in sy])

            K_opt = sy[idx_min]
            y_new[y_new < K_opt] = K_opt

            y_new = y_new/abs(np.sum(y_new)*(x[0]-x[1]))

            yfes = -R*temp_targeted_MD*np.log(y_new)/1000
            yfes = yfes-np.min(yfes)
            ygrad = np.gradient(yfes, x)

            return x, y_opt, y_old, y_new, yfes, ygrad

        self.features = sim_data.features
        self.nfeatures = len(sim_data.features[0])
        self.features_def_list_idx = molecule_features.features_def_list_idx

        # calculate how much the two distributions differ (old vs new), calculate new distribution
        nd_array = []

        importance_list = []
        for idx, i in enumerate(np.arange(self.nfeatures)):
            if self.features_def_list_idx[idx] == 1:
                x, y1, y2, y3, yfes, ygrad = calc_pop_fes_grad_dih(
                    self.features[:, i], weights)
            elif self.features_def_list_idx[idx] == 21:
                x, y1, y2, y3, yfes, ygrad = calc_pop_fes_grad_puck_phi(
                    self.features[:, i], weights)
            elif self.features_def_list_idx[idx] == 22:
                x, y1, y2, y3, yfes, ygrad = calc_pop_fes_grad_puck_theta(
                    self.features[:, i], weights)
            elif self.features_def_list_idx[idx] == 31:
                x, y1, y2, y3, yfes, ygrad = calc_pop_fes_grad_puck5_phs(
                    self.features[:, i], weights)

            importance_list.append([int(idx), 100*np.sum((y2-y1)**2)/len(x)])
            nd_array.append([x, y1, y2, y3, yfes, ygrad])
        importance_list_sorted = sorted(
            importance_list, key=lambda x: x[1], reverse=True)
        importance_list_sorted = np.array(importance_list_sorted)
        self.importance_list = importance_list_sorted
        self.nd_array = nd_array
