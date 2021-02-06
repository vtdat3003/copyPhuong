from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

# def grad(x):
#     return 2*x+5*np.cos(x)
#
# def cost(x):
#     return x**2 + 5*np.sin(x)
#
# def GD(eta, x_0):
#     x =[x_0]
#     for i in range(100):
#         x_new = x[-1] -eta*grad(x[-1])
#         if abs(grad(x_new)) < 1e-3:
#             break
#         x.append(x_new)
#     return x, i
#
# x1, i1 = GD(.1, -5)
# x2, i2 = GD(.1, 5)
# print('Solution x1 = %f, cost = %f, obtained after %d interations'%(x1[-1], cost(x1[-1]), i1))
# print('Solution x2 = %f, cost = %f, obtained after %d interations'%(x2[-1], cost(x2[-1]), i2))


X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2 * np.random.randn(1000, 1)  # noise added

# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)


def grad(w):
    N = Xbar.shape[0]
    return 1 / N * Xbar.T.dot(Xbar.dot(w) - y)


def myGD(w_init, grad, eta):
    w = [w_init]
    for i in range(100):
        w_new = w[-1] - eta * grad(w[-1])
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
        w.append(w_new)
    return w, i


w_init = np.array([[2], [1]])
w1, i1 = myGD(w_init, grad, 1)
print('Solution found by GD: w = ', w1[-1].T, '\n after %d iterations' % (i1 + 1))
