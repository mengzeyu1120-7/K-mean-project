# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:31:14 2024

@author: 27963
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def DistanceMatrix1d(evalpts, ctrs):
    return np.abs(evalpts.reshape(-1, 1) - ctrs.reshape(1, -1))

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
fig.set_tight_layout(True)