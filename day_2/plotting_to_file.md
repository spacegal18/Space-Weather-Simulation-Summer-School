
---
title: Outputting plots to files
description: Day 2 - Afternoon
author: Aaron Ridley (ridley@umich.edu)
keywords: space-weather, space, python, matplotlib
---

# Python: Understanding plotting to a file

## General Idea

When we run from a command line and make plots, matplotlib can't open a window and make the plot, so we need to save the plot as a file instead.  We can then view the file.  This is pretty easy to do!

Saving the file can be done with the savefig function.  It can be used like this:
```python
plt.savefig('myfile.png')
```
This will save the plot into the file called "myfile.png".  You can change the file type to something like jpg or pdf and it will also work.

## Example 1:


```python
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
```

---------------
## Assignment 1:

Let's combine a few things that you have worked on already:

- Take the example code for reading in the Dst file
- Modify it so that you can specify the file name, index to plot, and output filename using argparse
- Use these variables to read the file and write out a plot file.

