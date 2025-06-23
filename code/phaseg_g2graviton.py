import numpy as np
import scipy as sp

import pygeomphase as pgp
import bispy as bsp
import quaternion

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import palettable as pal

#import tikzplotlib as tpl

from tqdm import tqdm, trange


# Calcul des coordonnées sur la sphère de Poincaré

def get_Stokes_param(x):
    x_bar = np.conjugate(x)
    rho = np.einsum('ik, jk -> ijk', x_bar, x)
    #print(f'{rho.shape = }')
    
    Stokes = np.empty((3, x.shape[1]))
    Stokes[0] = np.real(rho[0,1])
    Stokes[1] = np.imag(rho[1,0])
    Stokes[2] = np.real(rho[0,0]) - 1.
    Stokes = Stokes/np.linalg.norm(Stokes, axis=0)
    #print(f'{np.linalg.norm(Stokes[:,0]) = }')

    return Stokes

# Calcul des phases géométriques (pt. III, sec. 1)

def get_phases(x):

    phase_tot = np.zeros(x.shape[1])
    phase_dyn = np.zeros(x.shape[1])

    # conjué de x pour produit hermitien
    x_bar = np.conjugate(x)
    #print(f'\n{x.shape = }')

    # calcul de la phase totale en tout point
    phase_tot[1:] = np.angle(np.einsum('ij, i -> j', x[:,1:], x_bar[:,0]))
    
    # calcul de la phase dynamique
    prod_hermi = np.einsum('ij, ij -> j', x[:,1:], x_bar[:,:-1])
    phases_loc = np.angle(prod_hermi)
    phase_dyn[1:] = np.cumsum(phases_loc)

    # et, enfin, de la phase géométrique (on rajoute le permier terme, manquant dans le calcul)
    phase_tot = np.unwrap(phase_tot)
    phase_dyn = np.unwrap(phase_dyn)
    phase_geo = np.unwrap(phase_tot - phase_dyn) 
    return phase_tot, phase_dyn, phase_geo

# Plot et sauvegarde des figures

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))  # outward by 10 points
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def plot2D(t, x, phases, labels=['u(t)', 'v(t)'], save_as = None):
    cmap = pal.colorbrewer.qualitative.Set2_3.mpl_colors
        
    labelsize=12
    fig = plt.figure(figsize=(10, 7))

    gs = GridSpec(3, 5, figure=fig, hspace=0.2,wspace=1, top=0.99, bottom=0.15, right=0.93, left=0.1)
    ax1 = fig.add_subplot(gs[:2, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, 2:], sharey =ax2)
    ax4 = fig.add_subplot(gs[2, 2:])

    # ax1
    ax1.plot(x[0], x[1])

    # ax2 
    ax2.plot(t, x[0])

    # ax3
    ax3.plot(t, x[1], zorder=0)

    #ax4 
    ax4.plot(t, phases[1],  label=r'$\Phi_{\mathrm{dyn}}(t)$', color=cmap[0], linewidth=3)
    ax4.plot(t, phases[0], label=r'$\Phi_{\mathrm{tot}}(t)$', color='k', linewidth=2, linestyle='dotted')

    ax4bis = ax4.twinx()
    p2, = ax4bis.plot(t, phases[2], color=cmap[2], label=r'$\Phi_{\mathrm{geo}}(t)$')

    # graphic stuff 

    #ax1.set_xlim(-1.25, 1.25)
    #ax1.set_ylim(-1.25, 1.25)

    #ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
    #ax1.set_yticks([-1, -0.5, 0, 0.5, 1])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_position(('outward', 5))
    ax1.spines['bottom'].set_position(('outward', 5))
    adjust_spines(ax2, ['left'])
    adjust_spines(ax3, ['left'])
    adjust_spines(ax4, ['left', 'bottom'])
    ax4bis.spines['top'].set_visible(False)
    ax4bis.spines['left'].set_visible(False)
    ax4bis.spines['bottom'].set_visible(False)
    ax4bis.spines['right'].set_position(('outward', 5))



    ax1.set_xlabel(labels[0], fontsize=labelsize)
    ax1.set_ylabel(labels[1], fontsize=labelsize)

    ax2.set_ylabel(labels[0], fontsize=labelsize)
    ax3.set_ylabel(labels[1], fontsize=labelsize)

    ax4.set_xlabel('temps [s]', fontsize=labelsize)
    ax4bis.set_ylabel('phase géo. [rad]')
    ax4.set_ylabel(r'phase tot. & phase dyn. [rad]')

    fig.legend(ncol=3, loc=(0.5, 0.01), fontsize=12, frameon=True)
    # colors
    ax4bis.yaxis.label.set_color(p2.get_color())
    ax4bis.tick_params(axis='y', colors=p2.get_color())
    ax4bis.spines['right'].set_color(p2.get_color())

    # fixes
    ax1.set_box_aspect(1)


    if save_as is not None:
        tpl.clean_figure()
        tpl.save(f'{save_name}.tex', axis_width=r'\figwidth', axis_height=r'\figheight')
        plt.savefig(f'{save_as}.pdf')

def plot3D(S1, S2, S3, save_as=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
    r1 = np.cos(u) * np.sin(v)
    r2 = np.sin(u) * np.sin(v)
    r3 = np.cos(v)

    r = 0.99
    ls = LightSource(azdeg=210, altdeg=30)
    ax.plot_surface(
        r * r1,
        r * r2,
        r * r3,
        color=(0.9, 0.9, 0.9),
        alpha=0.1,
        zorder=-1,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
    )

    ax.plot(S1, S2, S3)

    if save_as is not None:
        plt.savefig(f'{save_as}.pdf')



if __name__=='__main__':


# Chargement et formatage des données 

    # chargement

    path2data = '../StageM2/.data_sets/'
    data_load = np.load(path2data + 'data_precession_OG_Pancharatnam.npz')

    print(f'\n keys : \n{data_load.keys()}')
    print(f'\n shapes : ')
    for i in range(1,5):
        print(f'{data_load[f'time{i}'].shape = }\t {data_load[f'hp{i}'].shape = }\t\t {data_load[f'hc{i}'].shape = }')

    # formatage

    nb_point = -851#50_000
    nb_point += 850     # 850 pt seront enlevé car post merge
    resolution = 5
    time = [data_load[f'time{i}'][-nb_point:] for i in range(1,5)]
    time = [time_i[::resolution] for time_i in time]

    h = [np.array([data_load[f'hp{i}'], data_load[f'hc{i}']])[:, -nb_point:] for i in range(1,5)]
    h = [h_i[:,::resolution] for h_i in h]

    make_noisy = False
    if make_noisy :
        rng = np.random.default_rng()
        for h_i in h:
            h_i += rng.normal(loc=np.zeros((2,1)), scale=1e-22, size=h_i.shape)



# Transgormée en SA, suppression des zéros et normalisation

    # transformation en SA
    SAh = [sp.signal.hilbert(h_i, axis=-1) for h_i in h]
    norm = [np.linalg.norm(SAh_i, axis=0) for SAh_i in SAh]
    SAh_nrmlz = [None]*4

    # suppression des valeurs nulles + bord
    cut_border = 2_000 # points enlevés pour cacher les effets de bord dû à la transformée de Fourier
    for i in range(4):

        # suppressions de bord + données post-merge
        time[i] = time[i][cut_border:-850]
        h[i] = h[i][:,cut_border:-850]
        SAh[i] = SAh[i][:,cut_border:-850]
        norm[i] = norm[i][cut_border:-850]
        #selection des 0
        non_zeros = np.where(norm[i] == 0, False, True)
        #print(f'{non_zeros = }')

        # suppression des ces données
        time[i] = time[i][non_zeros]
        h[i] = h[i][:,non_zeros]
        SAh[i] = SAh[i][:,non_zeros]
        norm[i] = norm[i][non_zeros]

        # normalisation
        SAh_nrmlz[i] = SAh[i] / norm[i]


# Calcul des 3 phases

    Phis = [get_phases(SAh_i) for SAh_i in SAh_nrmlz]


# Plot des résultats

    for i, (time_i, h_i, SAh_i, Phis_i) in enumerate(zip(time, h, SAh_nrmlz, Phis)):

        plot2D(time_i, h_i, Phis_i, 
            labels=[r'$h_+$', r'$h_\times$'], 
            save_as=f'../StageM2/fig/part-3/GW_full_{i+1}'
            )
        #plt.close()
        plt.show()

        S1, S2, S3 = get_Stokes_param(SAh_i)
        plot3D(S1, S2, S3
            ,save_as=f'../StageM2/fig/part-3/GW_proj_full{i+1}'
            )
        #plt.close()
        plt.show()



