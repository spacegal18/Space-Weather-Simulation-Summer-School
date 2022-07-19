# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:03:30 2022

@author: Neha
"""

#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Neha Srivastava'
__email__ = 'ns1202@usnh.edu'

from math import factorial
from math import pi
import numpy as np
import argparse

#




def parse_args():
    parser = argparse.ArgumentParser(description = \
                                   'Cosine Code')
    parser.add_argument('-npts', \
                    help = 'approximation points(default = 10)', \
                    type = int, default = 10)
    parser.add_argument('-x', \
                        help = 'angle of approximation (default = 0)', \
                        type = float, default = 0)
    args = parser.parse_args()
    return args

# ------------------------------------------------------
# My Main code:
# ------------------------------------------------------



def cos_approx(x, npts):
    """Finding Cosine taylor expansion"""
    cos_list = [integral(x,npts) for npts in range(npts)]         
    cos_approx = sum(cos_list)
    return cos_approx

def integral(x,npts):
    """calaculating integral seperately for each n"""
    x_sq = x**2         
    fact_2n = factorial(2*npts)
    value = (((-1)**npts) * (x_sq**npts)) / fact_2n
    return value



#calculating cos value from input given through terminal
if __name__ == '__main__':  # main code block

    args = parse_args()
    print(args)

    npts = args.npts
    print("the number of points used in the approximation = ",npts)


    x = args.x
    print("the angle of approximation(x) = ",x)

    print("the approximation value of cos(",x,")=",cos_approx(x, npts))
    
    print("the actual value of cos(",x,")=",np.cos(x))
    
    
    
    diff = np.abs(np.cos(x) -cos_approx(x, npts))   # difference with actual value and approximate value
    
    eta = 1e-4  
    print(diff)
    
    if diff <= eta:
        print("this is good approximation")
    else :
        print ("this is bad approximation")
    
    



# Will only run if this is run from command line as opposed to imported
#    print("cos(0) = ", cos_approx(0))
#    print("cos(pi) = ", cos_approx(pi))
#    print("cos(2*pi) = ", cos_approx(2*pi))
#    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))





