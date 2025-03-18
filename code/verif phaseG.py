import pygeomphase as pgp
import bispy as bsp
import numpy as np
import matplotlib.pyplot as plt
import quaternion
import scipy as sp

def terme_entro(i):
    Dtheta = theta1[i]-theta1[0]
    Dchi = chi1[i]-chi1[0]

    bb = np.tan(Dtheta) * (np.tan(chi1[i]) + np.tan(chi1[0])) / (1 + np.tan(chi1[i])*np.tan(chi1[0]))
    return np.arctan(bb)

    # Construction du signal

# durée et précision du signal
N = 1024 
t = np.linspace(0, 1, N)


# param du signal
phi1  =  20*np.pi * t**2 
theta1 = np.pi/8  * t #np.ones_like(t)
chi1  =  np.pi/10 * t #np.ones_like(t)
alpha1 = np.pi/1  * t #np.ones_like(t)


# le signal
env = bsp.utils.windows.hanning(N)

x = bsp.signals.bivariateAMFM(env, theta1, chi1, phi1)
x1, x2 = bsp.utils.sympSplit(x)
xvec = np.array([x1, x2])


    # Calcul de la phase g

# la "vraie" phase g
phig = -np.angle(pgp.bargmann_invariant(xvec))
#bsp.utils.visual.plot2D(t, x)
#plt.show()

print(f'{phig = }')
print(f'{xvec.shape =} \n {xvec = }')

calc_pg = np.empty_like(t)
for i,t_ in enumerate(t) :
    calc_pg[i] = sp.integrate.quad(lambda t: (np.pi/8 * np.sin(np.pi/5 * t)), t[0], t_)[0]
    plus = terme_entro(i)
    print(f'{i} / calc_pg[i] = {calc_pg[i]} + {plus}')

    calc_pg[i] += plus

plt.plot(phig, label='real')
plt.plot(calc_pg, label='calc')
plt.legend()
plt.show()