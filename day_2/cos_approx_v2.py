#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function with argparse
"""
__author__ = 'Aaron Ridley'
__email__ = 'ridley@umich.edu'

from math import factorial
from math import pi
import argparse

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
    
    parser = argparse.ArgumentParser(description = 'Approximate cos')

    parser.add_argument('-npts', \
                        help = 'number of points (accuracy)', type = int,
                        default = 10)

    args = parser.parse_args()

    return args


def cos_approx(x, accuracy=10):
    """
    Approximate cos function using summation
    """

    ind_terms = [(-1.0) ** n / factorial(2*n) * x**(2 * n)
                 for n in range(accuracy+1)]
    tot = sum(ind_terms)
    
    return tot


# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    args = parse_args()
    n = args.npts
    print("cos(0) = ", cos_approx(0, n))
    print("cos(pi) = ", cos_approx(pi, n))
    print("cos(2*pi) = ", cos_approx(2*pi, n))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
