# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:51:33 2022

@author: Neha
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:03:05 2022

@author: Neha
"""


""" using argparse trying to save plot by giving the input value in the terminal """

import numpy as np
import datetime as dt
import argparse

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description = \
                                   'Plotting DST file and saving using parse')
    parser.add_argument('-fileout', \
                    help = 'the file name for plot', \
                    type = str)
    args = parser.parse_args()
    return args

def read_ascii_file(filename,index):
    
    with open(filename) as f:
        """year = []
        day = []
        hour = []
        minute = []
        time =[]
        data = []"""
        data_dic = {"time":[],
                    "year":[],
                    "day":[],
                    "hour":[],
                    "minute":[],
                    "data":[]}
        for line in f:
            tmp = line.split()
            #create datetime in each line
            time0 = dt.datetime(int(tmp[0]),1,1,int(tmp[2]),int(tmp[3]),0)\
                +dt.timedelta(days=int(tmp[1])-1)
            data_dic["year"].append(int(tmp[0]))        #convert a string to integer
            data_dic["day"].append(int(tmp[1]))
            data_dic["hour"].append(int(tmp[2]))
            data_dic["minute"].append(int(tmp[3]))
            
            
            data_dic["time"].append(time0)
            data_dic["data"].append(float(tmp[index]))   
            
    return data_dic



#Main body

file = "omni_min_def_aqukvr4tR_.lst"
index =-1
data_dic= read_ascii_file(file, index)
#print(data_dic["time"])
#print(data_dic["year"])
#print(data_dic["day"])
#print(data_dic["hour"])
#print(data_dic["data"])

time = np.array(data_dic["time"])
symh = np.array(data_dic["data"])   #magnetic disturbance
fig,ax = plt.subplots()


#finding maximum and minimum peaks and printing in the terminal
max_symh = np.max(symh)
min_symh = np.min(symh)
min_time = np.where(symh == min_symh)
max_time = np.where(symh == max_symh)
min_value = time[min_time]
max_value = time[max_time][0]
print("minimum symh value =", min_symh)
print("time of minimum symh =",time[min_time][0].isoformat())




#plotiing all events
ax.plot(time,symh,marker='.',c='gray',label='All Events',alpha=0.5)

lp = symh<-100
#print(lp)
""""differentiating the <-100 value and plotting with different linestyle"""
ax.plot(time[lp],symh[lp],marker='+',linestyle='',c='tab:orange',label='<-100 nT',alpha=0.6)

#ax.axvspan(min_value,max_value, c='tab:green')

""""vertical lines for min and max peaks"""
ax.axvline(min_value, color = 'b')
ax.axvline(max_value, color = 'b')


ax.set_xlabel('year of 2013')
ax.set_ylabel('SYMH(nT)')
ax.grid(True)
ax.legend()

args = parse_args()

fileout = args.fileout

#saving the file with given input name
outfile = fileout+'.png'
print('Writing file : ' + outfile)
plt.savefig(outfile)
plt.close()



