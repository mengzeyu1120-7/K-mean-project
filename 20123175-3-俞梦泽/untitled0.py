# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:13:45 2024

@author: 27963
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x, eps, r = sp.symbols('x eps r')
rbf_symbol = sp.exp(-eps ** 2 * (x ** 2))
rbf_fn = sp.lambdify([x, eps], rbf_symbol)
lrbf_fn = sp.lambdify([x, eps], rbf_symbol.diff(x, 2))

a = 20


def f(x):
    return -a * a * np.sin(a * x)


def u(x):
    return np.sin(a * x)


def rbf(epsilon, DM_eval):
    return rbf_fn(r=DM_eval, eps=epsilon)


def lrbf(epsilon, DM_eval):
    return lrbf_fn(r=DM_eval, eps=epsilon)


def DistanceMatrix1d(evalpts, ctrs):
    return np.abs(evalpts.reshape(-1, 1) - ctrs.reshape(1, -1))


def createPoints1d(m, method='u'):
    if method == 'u':
        return np.linspace(0, 1, m, endpoint=True)




def update(N):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.set_tight_layout(True)
    
    collpts = createPoints1d(N, 'u')
    collpts = 2 * collpts - 1
    indx = np.logical_or(collpts == 1, collpts == -1.)
    bdypts = collpts[indx]
    intpts = collpts[np.logical_not(indx)]
    ctrs = np.concatenate([intpts, bdypts], axis=0)
    evalpts = createPoints1d(M, 'u')
    evalpts = 2 * evalpts - 1
    DM_eval = DistanceMatrix1d(evalpts, ctrs)
    EM = rbf_fn(eps=epsilon, x=DM_eval)
    DM_int = DistanceMatrix1d(intpts, ctrs)
    DM_bdy = DistanceMatrix1d(bdypts, ctrs)
    LCM = lrbf_fn(eps=epsilon, x=DM_int)
    BCM = rbf_fn(eps=epsilon, x=DM_bdy)
    CM = np.concatenate([LCM, BCM], axis=0)
    rhs_int = f(intpts)
    rhs_bry = u(bdypts)
    rhs = np.concatenate([rhs_int, rhs_bry], axis=0)
    Pf = EM@(np.linalg.solve(CM, rhs))
    ax.plot(evalpts, Pf)
    ax.plot(evalpts, u(evalpts), '--')
    ax.set_title(f'N={N}')
    ax.set_ylim([-2, 2])
    EMA = abs(Pf - u(evalpts))
    
    return max(EMA)
     


M = 200
epsilon = 3
for i in range(5,26,2):
    print(update(i))
    plt.show()