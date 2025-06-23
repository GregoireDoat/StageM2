import bispy as bsp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import palettable as pal
from scipy.linalg import expm

cmap = (pal.colorbrewer.qualitative.Set2_8.mpl_colors)  # Colormap dans laquelle on peut piocher
labelsize = 12

# Fonction de julien pour gérer les axes
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 5))  # outward by 10 points
        else:
            spine.set_color("none")  # don't draw spine

    # turn off ticks where there is no spine
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def plot_trivariate(x, y, z, t):
    fig = plt.figure(figsize=(14, 8))

    gs = GridSpec(
        3,
        6,
        figure=fig,
        hspace=0.2,
        wspace=1,
        top=0.97,
        bottom=0.15,
        right=0.93,
        left=0.1,
    )
    ax1 = fig.add_subplot(gs[:3, :3], projection="3d")
    ax2 = fig.add_subplot(gs[0, 3:])
    ax3 = fig.add_subplot(gs[1, 3:], sharey=ax2)
    ax4 = fig.add_subplot(gs[2, 3:], sharey=ax3)

    # ax1

    # Plot plane (x,y,0)
    ax1.plot(x, y, z, linewidth=2)

    # Set plot properties
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Set the limits of the plot
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])

    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)

    ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_zticks([-1, -0.5, 0, 0.5, 1])

    # ax2

    ax2.plot(t, x, linewidth=2)
    ax2.set_title(r"$x(t)$")

    # ax3
    ax3.plot(t, y, zorder=0, linewidth=2)
    ax3.set_title(r"$y(t)$")

    # ax4
    ax4.plot(t, z, linewidth=2)
    ax4.set_title(r"$z(t)$")

    # graphic stuff

    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_position(("outward", 5))
    ax2.spines["bottom"].set_position(("outward", 5))
    adjust_spines(ax2, ["right"])
    adjust_spines(ax3, ["right"])
    adjust_spines(ax4, ["right", "bottom"])

    # ticks
    ax2.yaxis.set_ticks_position("right")
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.yaxis.set_ticks_position("right")
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax4.yaxis.set_ticks_position("right")
    ax4.set_yticks([-1, -0.5, 0, 0.5, 1])

    # labels
    ax1.set_xlabel(r"$x(t)$", fontsize=labelsize, loc="right")
    ax1.set_ylabel(r"$y(t)$", fontsize=labelsize)
    ax1.set_zlabel(r"$z(t)$", fontsize=labelsize)
    # ax2.set_ylabel(r'$x(t)$', fontsize=labelsize)
    # ax3.set_ylabel(r'$y(t)$', fontsize=labelsize)
    # ax4.set_ylabel(r'$z(t)$', fontsize=labelsize)
    ax4.set_xlabel("Time [s]", fontsize=labelsize)
    # Add the three axes at the origin
    V = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origin = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # origin point
    ax1.quiver(*origin, V[:, 0], V[:, 1], V[:, 2], color=["r", "g", "b"])
    # display
    plt.show()


L1 = np.matrix(np.array(((0, 1, 0), (1, 0, 0), (0, 0, 0))))
L2 = np.matrix(np.array(((0, -1j, 0), (1j, 0, 0), (0, 0, 0))))
L7 = np.matrix(np.array(((0, 0, 0), (0, 0, -1j), (0, 1j, 0))))
# I = np.matrix(np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1))))


# durée et précision du signal
N = 1024 
t = np.linspace(0, 1, N)


# param du signal
phi1  =  20*np.pi * t**2 
theta1 = np.pi/8  * np.ones_like(t)
chi1  =  np.pi/10 * np.ones_like(t)
alpha1 = np.pi/1  * t #np.ones_like(t)
beta1  = np.pi/4  * t #np.ones_like(t)

def mat_exp(s, mat):
    return np.array([expm(1j * s_ * mat) for s_ in s])

def cumpute_multiphase(alpha, beta, phi, theta, chi):
    # calcul des expo
    exp_a = mat_exp(alpha,  L2)
    exp_b = mat_exp(beta,   L7)
    exp_t = mat_exp(theta,  L2)
    exp_c = mat_exp(-chi,   L1)

    # calcul du produit de ces expo
    P = np.einsum('ijk,  k -> ij', exp_c, np.array([1,0,0]))
    P = np.einsum('ijk, ik -> ij', exp_t, P)
    P = np.einsum('ijk, ik -> ij', exp_b, P)
    P = np.einsum('ijk, ik -> ij', exp_a, P)
    return P

P = cumpute_multiphase(alpha1, beta1, phi1, theta1, chi1)
s = np.einsum('i, ij -> ij', np.exp(1j * phi1), P).T

plot_trivariate(s[0].real, s[1].real, s[2].real, t)