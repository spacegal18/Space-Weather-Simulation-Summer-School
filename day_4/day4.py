# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:31:38 2022

@author: Neha
"""
__author__ = 'Neha Srivastava'
__email__ = 'ns1202@usnh.edu'

#%%

"""Using zip function """
names = ['Ahmed', 'Becky', 'Cantor']
ages = [21, 30, 45]
favorite_colors = ['Pink', 'Grey', 'Blue']

print(list(zip(names, ages, favorite_colors)))

for name, age, color in zip(names, ages, favorite_colors):
    print(name, age, color)
    
    
"""Output
[('Ahmed', 21, 'Pink'), ('Becky', 30, 'Grey'), ('Cantor', 45, 'Blue')]
Ahmed 21 Pink
Becky 30 Grey
Cantor 45 Blue
"""


#%%
from datetime import datetime

num_of_days = 10
years = [2009]*num_of_days
months = [12]*num_of_days
days = list(range(1,11))

times = [datetime(year, month, day)
         for year, month, day
         in zip(years, months, days)]
for time in times:
    print(time.isoformat())
    
"""Output
2009-12-01T00:00:00
2009-12-02T00:00:00
2009-12-03T00:00:00
2009-12-04T00:00:00
2009-12-05T00:00:00
2009-12-06T00:00:00
2009-12-07T00:00:00
2009-12-08T00:00:00
2009-12-09T00:00:00
2009-12-10T00:00:00
"""  

#%%
"""USing meshgrid instead of contourf"""

import numpy as np
import matplotlib.pyplot as plt

num_of_x = 10
num_of_y = 20
x = np.linspace(0, 1, num_of_x)
y = np.linspace(0, 1, num_of_y)
z = np.random.randn(num_of_y, num_of_x)     #first term is y instead of using x
plt.pcolormesh(x, y, z)
plt.colorbar()

#%%


"""Working on WAMP-IPE data"""

import netCDF4 as nc

dataset = nc.Dataset('D:/SWSS/WAMP/wfs.t12z.ipe05.20220721_020000.nc')
print(dataset)


dataset['tec'][:]  # How you get the numpy array of the data
dataset['tec'].units  # How you get the units of data
print(dataset['tec'][:])
print(dataset['tec'])

#%%
"""
Created on Thu Jul 21 12:31:38 2022

@author: Neha
"""
__author__ = 'Neha Srivastava'
__email__ = 'ns1202@usnh.edu'


"""
We are trying to plot Total electron content from WAMP data for 21st july """



import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    """Giving input filename from user in the terminal"""
    parser = argparse.ArgumentParser(description = \
                                   'WAMP TEC Plots')
    parser.add_argument('-filename', \
                             type = str)
    args = parser.parse_args()
    return args





def plot_tec(dataset, figsize=(12,6)):
    """Defining a plotting function to avoid writing/copying plot code everytime
    pcolormesh creates a pseudocolor plot with a non-regular rectangular grid of lon,lat dimension.
    lon - longitude
    lat - latitude
    tec - total electron content"""
    fig, ax = plt.subplots(1,figsize=figsize)
    cp=ax.pcolormesh(dataset['lon'][:],dataset['lat'][:],dataset['tec'][:])
    ax.set_title('Total Electron Content('+dataset['tec'].units+')', fontsize=20)
    ax.set_xlabel('Longitude('+dataset['lon'].units+')', fontsize=18)
    ax.set_ylabel('Latitude('+dataset['lat'].units+')', fontsize=18)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    fig.colorbar(cp,ax=ax)
    return fig, ax

def plot_sav(outfile):
    """SAving the plots"""
    plt.savefig(outfile)
    plt.close()
    return
    




args = parse_args()
filename=args.filename
infile = 'D:/SWSS/WAMP/'+filename
outfile = filename+'.png'

#loading the 21st july data using netcFD

dataset = nc.Dataset(infile) 


#calling plot function
plot_tec(dataset)
plot_sav()








    
    
    
