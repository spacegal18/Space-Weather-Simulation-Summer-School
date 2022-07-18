# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:36:52 2022

@author: Neha
"""

import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0,1)
plt.plot(x, np.exp(x))
plt.xlabel(r'$0 \leq x<1$')
plt.ylabel(r'$e^x$')
plt.title('Exponential function')
plt.show()