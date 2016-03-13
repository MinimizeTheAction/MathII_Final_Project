"""
A script that generates data to be used in the math final project and saves it to *.csv
"""

import numpy as np
t10000 = np.linspace(0,10,10000) # time, 100 samples
t1000 = np.linspace(0,10,1000) # time, 1000 samples
t500 = np.linspace(0,10,500) # time, 500 samples
t200 = np.linspace(0,10,200) # time, 500 samples
t100 = np.linspace(0,10,100) # time, 100 samples


# Parameters of obvious signal
A = 1. # amplitude
w = 3. # frequency
p = 1. # phase

# calculate the real obvious signal
realS10000 = A*np.cos(w*t10000+p)
realS1000 = A*np.cos(w*t1000+p)
realS500 = A*np.cos(w*t500+p)
realS200 = A*np.cos(w*t200+p)
realS100 = A*np.cos(w*t100+p)

# add on gaussian noise, of 0.3*A sigma
s10000 = realS10000 + np.random.normal(0,0.3*A,len(realS10000))
np.savetxt('10000Gauss0-3.csv',np.column_stack((t10000,s10000)),delimiter=',')

s1000 = realS1000 + np.random.normal(0,0.3*A,len(realS1000))
np.savetxt('1000Gauss0-3.csv',np.column_stack((t1000,s1000)),delimiter=',')

s500 = realS500 + np.random.normal(0,0.3*A,len(realS500))
np.savetxt('500Gauss0-3.csv',np.column_stack((t500,s500)),delimiter=',')

s200 = realS200 + np.random.normal(0,0.3*A,len(realS200))
np.savetxt('200Gauss0-3.csv',np.column_stack((t200,s200)),delimiter=',')

s100 = realS100 + np.random.normal(0,0.3*A,len(realS100))
np.savetxt('100Gauss0-3.csv',np.column_stack((t100,s100)),delimiter=',')
