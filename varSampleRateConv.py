import numpy as np
import cos_mcmc as cmc
import matplotlib.pyplot as plt

files = ['100Gauss0-3.csv','200Gauss0-3.csv','500Gauss0-3.csv',\
			'1000Gauss0-3.csv','10000Gauss0-3.csv']
			
varArray = np.zeros([len(files),3])
i = 0 # iter counts

for f in files:
	n = 1000
	a = 0
	w = 0
	p = 0
	numRuns = 100
	for j in range(0,numRuns): # do multiple runs to find average
		chain = cmc.main(n,f) # run a mcmc
		# extract chains
		aChain = chain[:,0]
		wChain = chain[:,1]
		pChain = chain[:,2]
		# find mean values
		a += np.mean(aChain)
		w += np.mean(wChain)
		p += np.mean(pChain)
	a = a/numRuns
	w = w/numRuns
	p = p/numRuns
	
	# find absolute error
	aErr = abs(a-1.)
	wErr = abs(w-3.)/3.
	pErr = abs(p-1.)
	
	varArray[i,:] = np.array([aErr,wErr,pErr])
	i += 1
	
samplingRates = np.array([10,20,50,100,1000])

plt.loglog(samplingRates,varArray[:,0],'-k',label='Amplitude')
plt.loglog(samplingRates,varArray[:,1],'--k',label='Frequency')
plt.loglog(samplingRates,varArray[:,2],'-.k',label='Phase')
plt.legend()
plt.title('Sampling Rate\'s Effect on Convergence')
plt.xlabel('Sampling rate (Hz)')
plt.ylabel('Relative Error')
plt.show()
