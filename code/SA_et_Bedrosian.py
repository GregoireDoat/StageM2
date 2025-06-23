import numpy as np
import scipy as sc

import matplotlib.pyplot as plt
import tikzplotlib as tpl
from tqdm import tqdm,trange

# Signal et ses dérivée

# temps
t = np.linspace(-10,10,5_000)

# signal réel bivarié
nu = [1.5, 1.3]
x = np.cos(2*np.pi*nu[0]*t) * np.cos(2*np.pi*nu[1]*t)

# transformée en SA
SAx = sc.signal.hilbert(x)

# paramètres instantanées
a_inst = np.abs(SAx)
phi_inst = np.angle(SAx)
freq_inst = np.diff(phi_inst) / (2.0*np.pi)

# suppression des bords (qui pose problème dans Fourier)
x = x[np.abs(t)<=4]
a_inst = a_inst[np.abs(t)<=4]
phi_inst = phi_inst[np.abs(t)<=4]
freq_inst = freq_inst[np.abs(t[:-1])<=4]

t = t[np.abs(t)<=4]


plt.plot(t, x)
plt.plot(t, np.cos(2*np.pi*nu[0]*t))
plt.plot(t, np.cos(2*np.pi*nu[1]*t))
plt.show()

plt.plot(t, x)
plt.plot(t, a_inst)
#plt.plot(t, phi_inst)
plt.plot(t, freq_inst)
plt.show()


def save_tikz_2Dplot(x, ys, save_as='tmp.tex', axis_opts='standard'):
    '''
    x = input du plot
    ys = list de dico avec valeurs de la fonction, coulour et eventuellement legende 
    '''

    tikzpicture = r'''\begin{tikzpicture}
    \begin{axis}''' + f'[{axis_opts}]\n\n\t'

    for y in ys :
        val = y['val']
        color = y['color']
        opt = y['option']
        data = ''

        for x_i, y_i in zip(x, val) :
            data += f'\n{x_i}\t{y_i}'
        
        tikzpicture += r'\addplot' + f'[color={color}, {opt}]' + ' table {' + data + '};\n\n\t'

    tikzpicture += r'''\end{axis}
\end{tikzpicture}'''

    with open(f'{save_as}.tex', 'w') as f:
        f.write(tikzpicture)
    print(tikzpicture)

ys = [{'val': x, 'color': 'black', 'option': 'semithick'},
      {'val': np.cos(2*np.pi*nu[0]*t), 'color': 'blue', 'option': ''},
      {'val': np.cos(2*np.pi*nu[1]*t), 'color': 'violet', 'option': ''}]

#save_tikz_2Dplot(t, ys, save_as='tikz/part-1/prod_cos-simple', axis_opts='standard')


ys = [{'val': x, 'color': 'black', 'option': 'semithick'},
      {'val': a_inst, 'color': 'blue', 'option': ''},
      {'val': freq_inst, 'color': 'violet', 'option': ''}]

#save_tikz_2Dplot(t, ys, save_as='tikz/part-1/prod_cos-paraminst', axis_opts='standard')
