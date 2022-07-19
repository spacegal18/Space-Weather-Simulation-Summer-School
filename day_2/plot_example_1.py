#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4 * np.pi)
y1 = np.sin(x)
y2 = np.cos(x)

fig = plt.figure(figsize = (10,10))
ax1 = fig.add_subplot(211)
ax1.plot(x, y1)
ax1.set_ylabel('sin')

ax2 = fig.add_subplot(212)
ax2.plot(x, y2)
ax2.set_ylabel('cos')

outfile = 'plot_example1.png'
print('Writing file : ' + outfile)
plt.savefig(outfile)
plt.close()
