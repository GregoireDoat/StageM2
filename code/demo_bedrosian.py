import numpy as np
import matplotlib.pyplot as plt

omega = 2*np.pi* [3, 0.1]

t = np.arange(-5, 5, 0.01)

x = np.cos(omega[0]*t) * np.cos(omega[1]*t)

Fx = np.fft.fft(x)

plt.plot(x)
plt.plot(Fx)

plt.show()

