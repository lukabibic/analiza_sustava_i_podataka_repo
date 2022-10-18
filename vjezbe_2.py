"""
masa na nagnutoj zici
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ulazni podaci
m = 1
g = 9.81
k = 1000
L0 = 1  # duljina opustene opruge
c = 1  # faktor prigusenja

# parametri sustava
a = 0.95 * L0
theta = np.pi / 8


def D2x(x, v, a, theta):
    return (1 / m) * (-k * x * (1 - (L0 / np.sqrt(x ** 2 + a ** 2))) - c * v + g * m * np.sin(theta))


# fiksne tocke
# interpolacija polinomom (XDDDDD)
Xint = np.linspace(-0.5, 0.5)
Xroots = np.roots(np.polyfit(Xint, D2x(Xint, 0, a, theta), 7))

Xfixpoints = np.empty(0)

for x in Xroots:
    if np.isreal(x):
        Xfixpoints = np.append(Xfixpoints, np.real(x))

Xfixpoints.sort()
print(Xfixpoints)
# fazni dijagram
Xplot = np.linspace(-0.5, 0.5)
plt.plot(Xplot, D2x(Xplot, 0, a, theta))
plt.plot(Xfixpoints, np.zeros_like(Xfixpoints), 'ro')
plt.axhline(color='black')
plt.axvline(color='black')
plt.xlabel('x')
plt.ylabel('x\'\'')
plt.title(f'fazni dijagram za a={a/L0}*L0, theta={theta:.2}')
plt.show()


# rjesenje modela
def D(U, t, a, theta):
    x, v = U
    return np.array([v, D2x(x, v, a, theta)])


x0 = 0.3
T = np.linspace(0, 10, 1000)
R = odeint(D, [x0, 0], T, args=(a, theta))
X = R[:, 0]
V = R[:, 1]
plt.figure()
plt.plot(T, X)
plt.axhline(color='black')
plt.axvline(color='black')
plt.xlabel('t')
plt.ylabel('x')
plt.title(f'rjesenje za x0={x0}, a={a/L0}*L0, theta={theta:.2}')
plt.show()