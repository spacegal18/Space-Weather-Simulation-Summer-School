#!/usr/bin/env python

import argparse
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Example 2 with a few more input arguments
# ----------------------------------------------------------------------

def parse_args():

    """ Function to parse input arguments

    Parameters
    ----------
    None!

    Returns
    -------
    args namespace

    Notes
    -----
    Parses the input arguments. Created this for the summer school
    To see how all of this works and all of the full functionality,
    take a look here:
    https://docs.python.org/3/library/argparse.html

    """
    
    parser = argparse.ArgumentParser(description = 'Read in omni file')

    parser.add_argument('file', \
                        help = 'Filename to read')

    parser.add_argument('output', \
                        help = 'Filename to output (e.g. test.png)')

    parser.add_argument('-var', \
                        help = 'Variable to plot')

    args = parser.parse_args()

    return args


# ----------------------------------------------------------------------
# Read file and return a dictionary!
# ----------------------------------------------------------------------

def read_ascii_file_all(file):

    """ Function to parse input arguments

    Parameters
    ----------
    file - this is the file to read

    Returns
    -------
    a dictionary containing the data in the file

    Notes
    -----
    This reads an ascii file and assigns var names
    This assumes that the format is:
    year month day hour minute and then variables....

    """

    with open(file) as f:

        # skip a bunch of lines:
        nLines = 3
        for iLine in range(nLines):
            dummy = f.readline()

        # read in the variables line and convert to variable names:
        header = f.readline()
        vars = header.split()

        # Create our dictionary:
        data_dict = {'time': []}
        # Add variables to dictionary:
        for var in vars:
            data_dict[var] = []
    
        for line in f:
            tmp = line.split()
            # create datetime in each line, assumes file is:
            # year month day hour minute
            time0 = dt.datetime(int(tmp[0]),int(tmp[1]),int(tmp[2]),
                                int(tmp[3]), int(tmp[4]))
            data_dict['time'].append(time0)

            for iVar, var in enumerate(vars):
                data_dict[var].append(float(tmp[iVar]))
    
    return data_dict


# ----------------------------------------------------------------------
# My Main code:
# ----------------------------------------------------------------------

args = parse_args()

print(args)

filename = args.file
var = args.var

data_dict = read_ascii_file_all(filename)

print(data_dict)

fig, ax = plt.subplots()

ax.plot(data_dict['time'], data_dict[var], marker='.', c='gray',
        label='All Events', alpha=0.5)

ax.set_xlabel('Time')
ax.set_ylabel(var)
ax.grid(True)
ax.legend()

outfile = args.output
print('Writing file : ' + outfile)
plt.savefig(outfile)
plt.close()
