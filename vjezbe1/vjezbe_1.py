import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# def f(x, t):
#     return x ** 2 - 1
#
# # -------------- PRVI ZAD
#
# # fazni prostor
# x_plot = np.linspace(-2, 2, 100)
# plt.plot(x_plot, f(x_plot, None))
# plt.axvline(color='k')
# plt.axhline(color='k')
# plt.xlabel('x')
# plt.ylabel('x\'')
# plt.title('fazni dijagram')
# plt.show()
#
# # rjesenje
# T = np.linspace(0, 10, 200)
# x0 = 0.99  # pocetna tocka
# X = odeint(f, x0, T)
# X = X[:, 0]
# print(X)
#
# plt.figure()
# plt.plot(T, X)
# plt.axhline(y=1, color='red')
# plt.axhline(y=-1, color='red')
# plt.xlabel('x')
# plt.ylabel('x\'')
# plt.title(f'rjesenje, x0 = {x0}')
# plt.show()

# ----------------------------


# -------------- ZAD 2
"""
x' = r + x ** 2
"""


def f(x, t, r):
    return r + x**2


# parametar sustava
R = -1
T = np.linspace(-0, 10, 100)
x0 = - np.sqrt(-R) - 0.5
X = odeint(f, x0, T, args=(R, ))
X = X[:, 0]

plt.figure()
plt.plot(T, X)
plt.axhline(-1, color='k')
plt.axhline(1, color='k')
plt.xlabel('t')
plt.ylabel('x')
plt.title(f'rjesenje za r = {R}, x0 = {x0}')
plt.show()

# dijagram atraktora
Rs = np.linspace(-10, 0, 100)
XA = np.empty_like(Rs)

for i, R in enumerate(Rs):
    x0 = -10
    X = odeint(f, x0, T, args=(R, ))
    X = X[:, 0]
    XA[i] = X[-1]

plt.figure()
plt.plot(Rs, XA)
plt.axhline(color='red')
plt.axvline(color='red')
plt.xlabel('r')
plt.ylabel('x*')
plt.title(f'rjesenje za r = {R}, x0 = {x0}')
plt.show()
# ------------------------


# ------------------ 3. ZAD

# -------------------------
