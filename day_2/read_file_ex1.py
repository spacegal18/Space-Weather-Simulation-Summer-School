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

    parser.add_argument('-index', \
                        help = 'Index to read', type = int)

    args = parser.parse_args()

    return args


# ----------------------------------------------------------------------
# Read file and return a dictionary!
# ----------------------------------------------------------------------

def read_ascii_file(file, index):

    """ Function to parse input arguments

    Parameters
    ----------
    file - this is the file to read
    index - which column to read in the file

    Returns
    -------
    a dictionary containing the data in the file

    Notes
    -----
    This reads an ascii file of omni data

    """

    with open(file) as f:
        time = []          
        data = []
        for line in f:
            tmp = line.split()
            
            # create datetime in each line
            time0 = dt.datetime(int(tmp[0]),1,1,int(tmp[2]),int(tmp[3]),0)\
                          + dt.timedelta(days=int(tmp[1])-1)     
            time.append(time0)
            data.append(float(tmp[index]))
    
    time = np.array(time)
    data = np.array(data)

    data_dict = {'time': time,
                 'data': data}
    
    return data_dict


# ----------------------------------------------------------------------
# My Main code:
# ----------------------------------------------------------------------

args = parse_args()

print(args)

filename = args.file
index = args.index

data_dict = read_ascii_file(filename, index)

print(data_dict)

fig,ax = plt.subplots()

ax.plot(data_dict['time'],data_dict['data'],marker='.',c='gray',
        label='All Events',alpha=0.5)

ax.set_xlabel('Year of 2013')
ax.set_ylabel('SYMH (nT)')
ax.grid(True)
ax.legend()

outfile = args.output
print('Writing file : ' + outfile)
plt.savefig(outfile)
plt.close()
