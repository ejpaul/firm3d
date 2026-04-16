import matplotlib.pyplot as plt
import numpy as np


peta_final = np.loadtxt(f'peta_final.txt')
peta_init = np.loadtxt(f'peta_init.txt')
Es_final = np.loadtxt(f'Es_final.txt')
E_init = np.loadtxt(f'E_init.txt')


plt.clf()
plt.hist(peta_init, alpha=0.5,bins=50, label=r"initial $p_\eta$")
plt.hist(peta_final, alpha=0.5,bins=50, label=r"final $p_\eta$")
plt.legend()
plt.xlabel(r"$p_\eta$")
#plt.ylabel("Frequency")
plt.savefig(f'peta_hist.png')

plt.clf()
plt.hist(E_init,alpha=0.5,bins=50, label=r"initial $E$")
plt.hist(Es_final, alpha=0.5,bins=50, label=r"final $E$")
plt.legend()
plt.xlabel(r"$E$")
#plt.ylabel("Frequency")
plt.savefig(f'E_hist.png')