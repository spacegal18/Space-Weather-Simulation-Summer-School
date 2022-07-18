#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Neha Srivastava'
__email__ = 'ns1202@usnh.edu'

from math import factorial
from math import pi

#

def cos_approx(x, accuracy=50):
    """Finding Cosine taylor expansion"""
    cos_list = [integral(x,n) for n in range(accuracy)]         
    cos_approx = sum(cos_list)
    return cos_approx

def integral(x,n):
    """calaculating integral seperately for each n"""
    x_sq = x**2         
    fact_2n = factorial(2*n)
    value = (((-1)**n) * (x_sq**n)) / fact_2n
    return value


# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
