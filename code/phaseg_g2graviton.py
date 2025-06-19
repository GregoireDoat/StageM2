import numpy as np
import scipy as sp

import pygeomphase as pgp
import bispy as bsp
import quaternion

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.collections import LineCollection

from tqdm import tqdm, trange

dont_care = False
show_mine = False


# Chargement et formatage des données 

# chargement
path2data = '../StageM2/data/'
data_load = np.load(path2data + 'data_precession_OG_Pancharatnam.npz')

print('\n --- Format des données ---')

print(f'\n keys : {data_load.keys()}')
print(f'\n shapes : ')
for i in range(1,5):
    time = data_load[f'time{i}']
    h_p = data_load[f'hp{i}']
    h_c = data_load[f'hc{i}']
    print(f'{time.shape}\t {h_p.shape}\t {h_c.shape}')

# formatage
staring_point = 1_000_000

time = [data_load[f'time{i}'][staring_point:] for i in range(1,5)]

h = [np.array([data_load[f'hp{i}'], data_load[f'hc{i}']])[:, staring_point:] for i in range(1,5)]



# Transorfmé en SA, suppression des zéros et normalisation


# transformation en SA
SAh = [sp.signal.hilbert(h_i, axis=-1) for h_i in h]
norm = [np.linalg.norm(SAh_i, axis=0) for SAh_i in SAh]
SAh_nrmlz = [None]*4

for i in range(4):

    #selection des 0
    non_zeros = np.where(norm[i] == 0, False, True)#np.array([np.where(norm[i] == 0, False, True)]*2)
    print(f'{non_zeros = }')

    # suppression des ces données
    time[i] = time[i][non_zeros]
    h[i] = h[i][:,non_zeros]
    SAh[i] = SAh[i][:,non_zeros]
    norm[i] = norm[i][non_zeros]

    # normalisation
    SAh_nrmlz[i] = SAh[i] / norm[i]



# Calcul de la phase géométrique

# le calcul par bargmann
PhaseG_norm_norm  =  [-np.angle(pgp.bargmann_invariant(SAh_i))                for SAh_i in SAh_nrmlz]
PhaseG = [np.unwrap(-np.angle(pgp.bargmann_invariant_unnormalized(SAh_i)))   for SAh_i in SAh_nrmlz]
PhaseG_norm_unnorm = [-np.angle(pgp.bargmann_invariant(SAh_i))                for SAh_i in SAh]
PhaseG_unnorm_unnorm = [-np.angle(pgp.bargmann_invariant_unnormalized(SAh_i)) for SAh_i in SAh]

# plots
if not dont_care :
    for i in range(4):

        fig, ax = plt.subplots(2,2)#, figsize=(16,12))

        #plot de h
        ax[0,0].plot(time[i], h[i][0])
        ax[0,1].plot(time[i], h[i][1])


        ax[1,0].plot(h[i][0], h[i][1])

        #points = np.array([h[i][0], h[i][1]]).T.reshape(-1, 1, 2)
        #segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #print(f'{segments.shape = }')
        #color_grad = plt.Normalize(PhaseG[i].min(), PhaseG[i].max())
        #lc = LineCollection(segments, cmap='viridis', norm=color_grad)
        ## Set the values used for colormapping
        #lc.set_array(PhaseG[i][1:])
        #lc.set_linewidth(2)
        #line = ax[1,0].add_collection(lc)
        #fig.colorbar(line, ax=ax[1,0])

        # plot de la phase géométrique
        ax[1,1].plot(time[i], PhaseG_norm_norm[i], label='norm norm')
        ax[1,1].plot(time[i], np.unwrap(PhaseG[i]), label='unnorm norm')
        ax[1,1].plot(time[i], PhaseG_norm_unnorm[i], label='norm unnorm')
        ax[1,1].plot(time[i], PhaseG_unnorm_unnorm[i], label='unnorm unnorm')

        ax[1,1].legend()
    plt.show()


if show_mine :
    def my_phaseG(h):
        phase_geom = np.empty((h.shape[-1]))

        for i in trange(1,h.shape[-1]+1):
            prod_hermi = np.einsum('ij, ij -> i', np.roll(h[:i], shift=(-1,0)), np.conjugate(h[:i]))
            phase_geom[i] = np.sum(np.angle(prod_hermi))
        return phase_geom

    for i in range(4):
        PhaseG2 = my_phaseG(SAh_nrmlz[i])

        plt.plot(time[i], PhaseG)
        plt.plot(time[i], PhaseG2)
        plt.show()
