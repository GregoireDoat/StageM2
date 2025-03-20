import pygeomphase as pgp
import bispy as bsp
import numpy as np
import matplotlib.pyplot as plt
import quaternion
import scipy as sp

def terme_entro(i):
    Dtheta = theta[i]-theta[0]
    Dchi = chi[i]-chi[0]

    bb = np.tan(Dtheta) * (np.tan(chi[i]) + np.tan(chi[0])) / (1 + np.tan(chi[i])*np.tan(chi[0]))
    return np.arctan(bb)


    # Construction du signal

# durée et précision du signal
L = 10
N = 1024//2
t = np.linspace(0, L, L*N)


# param du signal
a = np.ones_like(t)
phi  =  0.5*np.pi * t**2 
theta = np.pi/3  * t #np.ones_like(t)
chi  =  np.pi/10 * np.ones_like(t)


# le signal
env = bsp.utils.windows.hanning(L*N)

x = bsp.signals.bivariateAMFM(env, theta, chi, phi)
#print(f'{x = }\n')
x1, x2 = bsp.utils.sympSplit(x)
xvec = np.array([x1, x2])


    # Plot

from matplotlib.colors import LightSource
# en 2D d'abord
bsp.utils.visual.plot2D(t, x)
#plt.show() 

# puis dans l'espace des paramètres
S0, S1, S2, S3 = bsp.utils.geo2Stokes(a, theta, chi)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

# ajout de la sphère
u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
r1 = np.cos(u) * np.sin(v)
r2 = np.sin(u) * np.sin(v)
r3 = np.cos(v)

r = 0.99
ls = LightSource(azdeg=210, altdeg=30)
ax.plot_surface(r * r1, r * r2, r * r3,
                color=(0.9, 0.9, 0.9),
                alpha=0.1,
                zorder=-1,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=False)

# ajout du signal
ax.plot(S1/S0, S2/S0, S3/S0)
plt.show()

    # Calcul de la phase g

# la "vraie" phase g
phig = -np.angle(pgp.bargmann_invariant(xvec))

print(f'{phig = }')
print(f'{xvec.shape =} \n {xvec = }')

calc_pg = np.empty_like(t)
for i,t_ in enumerate(t) :
    calc_pg[i] = sp.integrate.quad(lambda t: (np.pi/8 * np.sin(np.pi/10)), t[0], t_)[0]
    plus = terme_entro(i)

    if i%100 == 0:
        print(f'{i} / calc_pg[i] = {calc_pg[i]} + {plus}')

    calc_pg[i] += plus

plt.plot(phig, label='real')
plt.plot(calc_pg, label='calc')
plt.legend()
plt.show()