# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:19:57 2022

@author: Neha
"""

"""A 3D plot script for spherical coordinates.
"""
__author__ = 'Neha Srivastava'
__email__ = 'ns1202@usnh.edu'


import numpy as np
import matplotlib.pyplot as plt


r = np.linspace(0, 1)           #radius
theta = np.linspace(0, 2*np.pi)         #theta and phi can be azimuthal or zenith angle
phi = np.linspace(0, 2*np.pi)


"""We are trying to convert spherical coordinates to cartesian coordinates then making a directory of converted coordinates"""

def coordinate_conversion(r, phi, theta):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    coordinate_conversion = {'x':x,
                             'y':y,
                             'z':z}
    return coordinate_conversion


fig = plt.figure()  # better control

axes = fig.gca(projection='3d')  # make 3d axes
coords = coordinate_conversion(r, phi, theta)   
axes.plot(coords['x'],coords['y'],coords['z'])      #plotting cartesian coordinates



#if __name__ == '__main__':  # main code block
#    print("cartesian_coordinate(x,y,z) = ", coordinate_conversion(r,phi,theta))
    
    #print("cos(pi) = ", cos_approx(pi))




    