import numpy as np
import cos_mcmc as cmc
import matplotlib.pyplot as plt

f = '100Gauss0-3.csv'
varArray = np.zeros([6,3])

for i in range(1,7):
	n = 10**i
	a = 0
	w = 0
	p = 0
	numRuns = 10
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
	
	print i
	varArray[i-1,:] = np.array([aErr,wErr,pErr])
	

plt.semilogy(np.linspace(1,6,6),varArray[:,0],'-k',label='Amplitude')
plt.semilogy(np.linspace(1,6,6),varArray[:,1],'-.k',label='Frequency')
plt.semilogy(np.linspace(1,6,6),varArray[:,2],'--k',label='Phase')

plt.title('Convergence with Number of Links in Chain')
plt.xlabel('Log Links')
plt.ylabel('Relative Error')
plt.legend()
plt.show()
