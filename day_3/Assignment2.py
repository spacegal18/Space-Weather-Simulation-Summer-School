#!/usr/bin/env python3
# -*- coding: utf-8 -*-




# Import required packages
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = \
                                   'JB2008 and TIEGCM density plots')
    parser.add_argument('-altitude', \
                             type = int, default = 300)
    parser.add_argument('-fileout', \
                    help = 'the file name for plot', \
                    type = str)
    args = parser.parse_args()
    return args

dir_density_Jb2008 = r'C:\Users\Neha\SWSS\Data\JB2008\2002_JB2008_density.mat'


loaded_data = loadmat(dir_density_Jb2008)

# Uses key to extract our data of interest

JB2008_dens = loaded_data['densityData']

# The shape command now works :  shape of density array : Density_state x timestep(hours)
#print(JB2008_dens.shape)

#%%


#lST corresponds to longitude  #the discretization is done in the metadata itslef
localSolarTimes_JB2008 = np.linspace(0,24,24)       #somethimes it is included in the metadata not in this case
#print(localSolarTimes_JB2008)
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
nofAlt_JB2008 = altitudes_JB2008.shape[0]  #36 pts
nofLst_JB2008 = localSolarTimes_JB2008.shape[0]  #24 pts
nofLat_JB2008 = latitudes_JB2008.shape[0]       #20 pts

# We can also impose additional constratints such as forcing the values to be integers.
time_array_JB2008 = np.linspace(0,8760,20, dtype = int)  #time throughout the year

# For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008,8760), order='F') # Fortra-like index order
#tranforming 2d array to 4d array (localSolarTime, LAt, ALt, Time)
#order F rifgt order of data






#%%


localSolarTimes_JB2008 = np.linspace(0,24,24)       
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
nofAlt_JB2008 = altitudes_JB2008.shape[0]  #36 pts
nofLst_JB2008 = localSolarTimes_JB2008.shape[0]  #24 pts
nofLat_JB2008 = latitudes_JB2008.shape[0]       #20 pts



#%%

# Import required packages
import h5py

loaded_data = h5py.File('C:/Users/Neha/SWSS/Data/TIEGCM/2002_TIEGCM_density.mat')



tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T # convert from g/cm3 to kg/m3
#.T is for transpose 
altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten() #convert any input into column, can also used squeeze
latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten()
localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
nofAlt_tiegcm = altitudes_tiegcm.shape[0]     #shape of array
nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
nofLat_tiegcm = latitudes_tiegcm.shape[0]

time_array_tiegcm = np.linspace(0,8759,20, dtype = int)



tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')



time_index = 31*24

#interpolating tiegcm density data
tiegcm_interpolated_func = RegularGridInterpolator((localSolarTimes_tiegcm, latitudes_tiegcm, altitudes_tiegcm),\
                                    tiegcm_dens_reshaped[:,:,:,time_index], bounds_error=False, fill_value=None)






#%%
args = parse_args()

alt=args.altitude

fileout = args.fileout




hi = np.where(altitudes_JB2008==alt)
dens_find_tiegcm = np.zeros((24,20))
for lst_i in range(24):
    for lat_i in range(20):
        dens_find_tiegcm[lst_i,lat_i] = tiegcm_interpolated_func((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],alt))

        

fig, ax = plt.subplots(2, figsize=(15, 10), sharex=True)

cs = ax[0].contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_index].squeeze().T)   
ax[0].set_title('JB2008 density at {} km, t = {} hrs'.format(alt,time_index), fontsize=18)
ax[0].set_ylabel("Latitudes", fontsize=18)
ax[0].tick_params(axis = 'both', which = 'major', labelsize = 16)

cbar = fig.colorbar(cs,ax=ax[0])
cbar.ax.set_ylabel('Density')

cs = ax[1].contourf(localSolarTimes_JB2008, latitudes_JB2008, dens_find_tiegcm.T)   
ax[1].set_title('TIEGCM density at {} km, t = {} hrs'.format(alt,time_index), fontsize=18)
ax[1].set_ylabel("Latitudes", fontsize=18)
ax[1].tick_params(axis = 'both', which = 'major', labelsize = 16)


cbar = fig.colorbar(cs,ax=ax[1])
cbar.ax.set_ylabel('Density')

ax[1].set_xlabel("Local Solar Time", fontsize=18)

outfile = 'TIEGCM vs JD2008 '+fileout+'.png'
print('TIEGCM vs JD2008 ' + outfile)
plt.savefig(outfile)
plt.close()




