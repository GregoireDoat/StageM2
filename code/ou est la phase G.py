import quaternion
import numpy as np
import matplotlib.pyplot as plt
import bispy as bsp
import pygeomphase as pgp
import scipy as sp

# stuff for 3D plotting
from matplotlib.colors import LightSource

def terme_entro(i):
    Dtheta = theta[i]-theta[0]
    Dchi = chi[i]-chi[0]

    bb = np.tan(Dtheta) * (np.tan(chi[i]) + np.tan(chi[0])) / (1 + np.tan(chi[i])*np.tan(chi[0]))
    return np.arctan(bb)


# params
f0 = 12  # in Hz
theta0 = 0
chi0 = np.pi / 5 * 0.2
x0 = bsp.utils.euler2quat(1, theta0, chi0, 0)

params = {
    "w0": 2 * np.pi * f0,
    "deltaw": 2 * np.pi * f0 * 0.0,
    "gamma": -2 * np.pi * f0 * 0.07,
    "phiw": 0.5,
    "x0": x0,
}


N = 1024
t = np.linspace(0, 1, endpoint=True, num=N)

x = pgp.rabi_bivariate(params, t, unit_rabi=True)
#x += 0.0 * (bsp.signals.bivariatewhiteNoise(N, 1) + quaternion.y * bsp.signals.bivariatewhiteNoise(N, 1))

# calcul Stokes
a, theta, chi, phi = bsp.utils.quat2euler(x)
#S0, S1, S2, S3 = bsp.utils.geo2Stokes(a, theta, chi)

#bsp.utils.visual.plot2D(t, x)

# param du signal
a = np.ones_like(t)
phi  =  0.5*np.pi * t*2
theta = np.pi/4  * t #np.ones_like(t)
chi  =  np.pi/10 * t #np.ones_like(t)


# le signal
env = bsp.utils.windows.hanning(N)
x2 = bsp.signals.bivariateAMFM(env, theta, chi, phi)


S0, S1, S2, S3 = bsp.utils.geo2Stokes(a, theta, chi)


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
plt.show()



# compute and plot geometric phase
x1, x2 = bsp.utils.sympSplit(x2)
xvec = np.array([x1, x2])

barg = pgp.bargmann_invariant(xvec)
barg2 = pgp.bargmann_invariant_unnormalized(xvec)

calc_pg = np.empty_like(t)
for i,t_ in enumerate(t) :
    calc_pg[i] = sp.integrate.quad(lambda s: (np.pi/8 * np.sin(np.pi/10 * s)), t[0], t_)[0]
    plus = terme_entro(i)

    if i%100 == 0:
        print(f'{i} / calc_pg[i] = {calc_pg[i]} + {plus}')

    calc_pg[i] += plus


plt.plot(-np.angle(barg))
plt.plot(-np.angle(barg2), "--")  # should be equal
plt.plot(calc_pg)
plt.show()