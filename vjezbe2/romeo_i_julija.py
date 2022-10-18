"""
Romeo i Julija
R' = sR * R + oR * J
J' = sJ * J + oJ * R
analiza_sustava' = (koji + kurac + radimo) * 2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def D(U, t, sR, oR, sJ, oJ):
    R, J = U
    return np.array([sR * R + oR * J, sJ * J + oJ * R])


# parametri
sR = -2
oR = 1
sJ = -2
oJ = 1

# rjesenje
R0, J0 = 0.75, 0
T = np.linspace(0, 10, 100)
RJ = odeint(D, [R0, J0], T, args=(sR, oR, sJ, oJ))
R = RJ[:, 0]
J = RJ[:, 1]

# fazni portret i rjesenje
plt.figure(figsize=(6, 6))
MR, MJ = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))

Rt = sR * MR + oR * MJ
Jt = sJ * MJ + oJ * MR

plt.quiver(MR, MJ, Rt, Jt, alpha=0.5, angles='xy', scale_units='xy')
plt.plot(R, J)
plt.axhline(color='black')
plt.axvline(color='black')
plt.xlabel('R')
plt.ylabel('J')
plt.title(f'fazni portret, sR={sR}, oR={oR}, sJ={sJ}, oJ={oJ}')
plt.show()

