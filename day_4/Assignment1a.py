# -*- coding: utf-8 -*-
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
    parser.add_argument('-filenames', \
                             type = str, nargs='+', help='files to save a file of the same name')
    """nargs='+' creates a list of given names"""
    args = parser.parse_args()
    return args



def plot_tec(dataset, quantity='tec', figsize=(12,6)):
    """Defining a plotting function to avoid writing/copying plot code everytime
    pcolormesh creates a pseudocolor plot with a non-regular rectangular grid of lon,lat dimension.
    lon - longitude
    lat - latitude
    tec - total electron content"""
    fig, ax = plt.subplots(1,figsize=figsize)
    cp=ax.pcolormesh(dataset['lon'][:],dataset['lat'][:],dataset[quantity][:])
    ax.set_title('Total Electron Content('+dataset[quantity].units+')-Time:', fontsize=20)
    ax.set_xlabel('Longitude('+dataset['lon'].units+')', fontsize=18)
    ax.set_ylabel('Latitude('+dataset['lat'].units+')', fontsize=18)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    fig.colorbar(cp,ax=ax, label = dataset[quantity].units)
    return fig, ax

def plot_sav(filename, vmin=0, vmax=70):
    """Defining a function to save the plot
    filename is given by the user and saves the file with same input name"""
    dataset = nc.Dataset(filename)
    fig, ax = plot_tec(dataset, vmin=vmin, vmax=vmax)
    fig.savefig(filename+'.png')
    plt.close()
    return filename


"""Calling parse function"""
args =parse_args()
filenames = args.filenames

for filename in filenames :
        plot_sav(filename)

