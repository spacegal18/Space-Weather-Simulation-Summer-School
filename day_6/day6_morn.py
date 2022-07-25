# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:33:27 2022

"""
__author__ = "Neha Srivastava"
__email_id__ = 'ns1202@usnh.edu'


#%%
"""Documentation - Using finite difference method to get the derivative
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


def function_fx(x):
    """Genereric Function definition
    Parameter
    ----------
    """
    
    function_fx = np.cos(x) + np.sin(x)
    return function_fx

def function_deri(x):
    """Derivative of the above function"""
    function_deri = x*np.cos(x)
    return function_deri

x_in = -6   #initial point
x_fin = 6   #final point
npts = 2000 #number of points
x = np.linspace(x_in,x_fin,npts)        #independent variable

y = function_fx(x)
y_dot = function_deri(x)


fig1 = plt.figure()
plt.plot(x,y)
plt.plot(x,y_dot)
plt.grid()
plt.legend([r'$y=\cos(x)$',r'$\dot{y}$'], fontsize = 16) #[] to seperate different lines 
plt.xlabel('X', fontsize = 18)
plt.ylabel('Y', fontsize = 18)
plt.title('Generic Function and Derivative')

#sys.exit() #this stops reading the script below this

#%%

import numpy as np
import matplotlib.pyplot as plt
import sys


def function_fx(x):
    """Genereric Function definition"""
    function_fx = np.cos(x) + x*np.sin(x)
    return function_fx


def function_deri(x):
    """Derivative of the above function
    Parameter: x is input value
    
    
    """
    function_deri = x*np.cos(x)
    return function_deri


def f_approx_forward(x0,h):
    """Forward difference"""
    f_current = function_fx(x0)
    f_forward = function_fx(x0+h)
    f_approx_forward = (f_forward - f_current)/h
    return f_approx_forward

def f_approx_backward(x0,h):
    """Backward difference"""
    f_current = function_fx(x0)
    f_backward = function_fx(x0-h)
    f_approx_backward = (f_current - f_backward)/h
    return f_approx_backward

def f_approx_central(x0,h):
    """central difference"""
    f_forward = function_fx(x0+h)
    f_backward = function_fx(x0-h)
    f_approx_central = (f_forward - f_backward)/(2*h)
    return f_approx_central

"""Step size ="""

x_initial = -6
x_final = 6
x0= x_initial
h = 0.25
f_dot_forward = np.array([]) #defining empty arrays
f_dot_backward = np.array([])
f_dot_central = np.array([])
x = np.array([x0])


y_truth = np.array([])


#forward difference
while x0 <= x_final:    
    f_slope_for = f_approx_forward(x0, h)
    f_slope_back = f_approx_backward(x0, h)
    f_slope_central = f_approx_central(x0, h)
    y_dot = function_deri(x0)
    x0 = x0 + h
    x = np.append(x, x0)
    f_dot_forward = np.append(f_dot_forward,f_slope_for)
    f_dot_backward = np.append(f_dot_backward,f_slope_back)
    f_dot_central = np.append(f_dot_central, f_slope_central)
    y_truth = np.append(y_truth,y_dot)
    
    
    
fig1 = plt.figure()    
plt.plot(x[:-1],y_truth,'-r')
plt.plot(x[:-1],f_dot_backward,'-m')
plt.plot(x[:-1],f_dot_forward,'-b')
plt.plot(x[:-1],f_dot_central,'-g')
plt.xlabel('X', fontsize = 18)
plt.ylabel('Y', fontsize = 18)
plt.legend([r'$\dot y truth$',r'$\dot y backward$', r'$\dot y forward$',r'$\dot y central$'])
plt.grid()
    





#%%

"""Integration -ODEs
Euler method or RK 1st order"""

from scipy.integrate import odeint

def RHS(y, t):
    """ODE Right hand side"""
    return -2*y

def int_RK1(y0,t0,h):
    y_int = y0 + h* RHS(y0,t0)
    return y_int

def int_RK2(y_current2,current_time,h):
    y = y_current2 + (0.5*h*RHS(y_current2,current_time))
    t = current_time + (0.5*h)
    y_int2 = y_current2 + (h*RHS(y,t))
    return y_int2
    


"""Initialization"""

y0 = 3  #
t0 = 0
tf = 2
h = 0.2
y_soln = np.array([y0])
time = np.array([t0])

"Evaluate exact solution"
time1 = np.linspace(t0,tf)
y_true = odeint(RHS,y0,time1)

current_time = t0
current_value = y0
sol_rk4 = np.array([y0])
time = np.array([t0])

while current_time <= tf-h:
    """y0 : current value
    t0 : current time
    h :  stepsize"""
    # Solve ODE
    k1 = RHS(current_value, current_time)
    k2 = RHS(current_value + k1*h*0.5, current_time + h*0.5)
    k3 = RHS(current_value + k2*h*0.5, current_time + h*0.5)
    k4 = RHS(current_value + k3*h, current_time + h)
    next_value = current_value + (k1 + 2*k2 + 2*k3 +k4)*h/6
    
    
    next_time = current_time + h
    time = np.append(time, next_time)
    sol_rk4 = np.append(sol_rk4, next_value)
    
    current_time = next_time
    current_value = next_value
    
    
    



fig1 = plt.figure()
plt.plot(time1,y_true,'-k',linewidth=2)
plt.plot(time,sol_rk4,'r-o',linewidth=2)
plt.grid()
plt.xlabel(r'time(t)')
plt.ylabel(r'$y(t)$')
plt.legend(['True value', 'Runge-Kutta 4'])

#%%
"""Non Linear PEndulum """

from scipy.integrate import odeint


def pendulum_free(x, time):
    g = 9.81
    l=3
    xdot = np.zeros(2)
    xdot[0] = x[1]
    xdot[1] = (-g*np.sin(x[0]))/l
    return xdot



t0 = 0
tf = 15   
 
x0 = np.array([np.pi/3 , 0])
n_points = 100
time_new = np.linspace(t0,tf,n_points)
y =odeint(pendulum_free,x0, time_new)


    


fig1 = plt.figure()
plt.title('Simple Pendulum')
plt.subplot(2,1,1)
plt.plot(time_new,y[:,0], 'r-', linewidth =2)
plt.grid()
plt.xlabel('time[s]')
plt.ylabel(r'$\theta$')

plt.subplot(2,1,2)
plt.plot(time_new,y[:,1], 'g-', linewidth =2)
plt.grid()
plt.xlabel('time[s]')
plt.ylabel(r'$\dot \theta$')


#%%

"""Lorentz63
Assignment 1a"""
from mpl_toolkits.mplot3d import Axes3D

def lorenz63(x,t,sigma,rho,beta):
    xdot = np.zeros(3)
    xdot[0] = sigma * (x[1] - x[0]);
    xdot[1] = (x[0])*(rho - x[2]) - x[1] ;
    xdot[2] = x[0]*x[1] - beta*x[2] ;
    return xdot

"""Parameters"""
sigma = 10
rho = 28
beta = 8/3

"""Initialization"""
t0 = 0
tf = 20
npts = 1000
x0 = np.array([5,5,5])
time = np.linspace(t0,tf,npts)
solution = odeint(lorenz63, x0, time, args=(sigma,rho,beta))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(solution[:,0],solution[:,1],solution[:,2])

#%%
"""Lorentz63
Assignment 1b - lorentz attractor """
from mpl_toolkits.mplot3d import Axes3D

def lorenz63(x,t,sigma,rho,beta):
    xdot = np.zeros(3)
    xdot[0] = sigma * (x[1] - x[0]);
    xdot[1] = (x[0])*(rho - x[2]) - x[1] ;
    xdot[2] = x[0]*x[1] - beta*x[2] ;
    return xdot

"""Parameters"""
sigma = 10
rho = 28
beta = 8/3

"""Initialization"""
t0 = 0
tf = 20
npts = 1000


"""Using random generator to get 20 initial values of x,y,z and plot for each"""
rand_pts = 20
x0x = np.random.randint(-20,20,rand_pts)
x0y = np.random.randint(-30,30,rand_pts)
x0z = np.random.randint(0,50,rand_pts)

x0 = np.array([x0x, x0y, x0z])
time = np.linspace(t0,tf,npts)
#solution = np.array(rand_pts)
fig = plt.figure()
ax = plt.axes(projection='3d')

for i in range(rand_pts):
    solution = odeint(lorenz63, x0[:,i], time, args=(sigma,rho,beta))    
    ax.plot3D(solution[:,0],solution[:,1],solution[:,2])



    




