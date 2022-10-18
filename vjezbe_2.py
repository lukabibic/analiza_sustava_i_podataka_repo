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


x0 = 0.7
# x0 = Xfixpoints[1]
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

# fazni portret
plt.figure()
MX, MV = np.meshgrid(np.linspace(-0.5, 0.5, 20), np.linspace(-3, 3, 20))

Xt = MV
Vt = D2x(MX, MV, a, theta)
plt.quiver(MX, MV, Xt, Vt, alpha=0.5, angles='xy', scale_units='xy')
plt.plot(X, V)
plt.plot(Xfixpoints, np.zeros_like(Xfixpoints), 'ro')
plt.axhline(color='black')
plt.axvline(color='black')
plt.xlabel('x')
plt.ylabel('v')

plt.show()

# dijagram atraktora s obzirom na pocetnu tocku
X0 = np.linspace(-1, 1, 100)
A = np.empty_like(X0)

for i in range(np.size(X0)):
    R = odeint(D, [X0[i], 0], T, args=(a, theta))
    A[i] = R[-1, 0]

plt.figure()
plt.plot(X0, A)
plt.axhline(color='black')
plt.axvline(color='black')
plt.xlabel('x0')
plt.ylabel('x*')
plt.title(f'Atraktori')
plt.show()