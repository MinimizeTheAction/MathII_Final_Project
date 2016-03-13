"""
Implements mcmc with Metropolis-Hastings algorithm to extract the amplitude, 
angular frequency, and phase of a noisy signal.
"""
import numpy as np
import scipy.stats as s
import matplotlib.pyplot as plt

global eps
eps = 0.1

def model(para,t):
	""" Model function: A*cos(w*t + p)"""
	A = para[0]
	w = para[1]
	p = para[2]
	return A*np.cos(w*t+p)

def checkPriors(para):
	""" 
	Check that parameters are within priors
	
	A -- [0.01,10]
	w -- [0.01,50]
	p -- [0.0, 2 pi]
	
	If not place them randomly inside the prior. This was chosen to improve mixing 
	between the parameters so that as much of parameter space as possible could be explored.
	"""
	
	# extract parameters
	A = para[0]
	w = para[1]
	p = para[2]
	
	# check them
	if (A<0.01 or A>10.0): A = s.uniform.rvs(0.01,10.)
	
	if (w<0.01 or w>10.0): w = s.uniform.rvs(0.01,50.)
		
	if ( p<0. or p>2*np.pi): p = s.uniform.rvs(00,2*np.pi)
	
	return np.array([A,w,p])
	
def runChain(paraCur,data,n,t):
	"""
	Run the mcmc simulation
	"""
	
	chainArray = np.zeros([n,len(paraCur)]) # initialize array to hold chain links
	accept = np.zeros(n) # to analyze acceptance rates
	
	for i in range(0,n):
		# propose a jump
		covProp = eps*np.eye(len(paraCur)) # covariance matrix
		paraProp = s.multivariate_normal.rvs(paraCur,covProp)
		
		
		paraProp = checkPriors(paraProp) # check that proposed jump 
										 # is valid
		
		# calculate the Hastings Ratio, assuming Gaussian likelihood
		sigCur = model(paraCur,t)
		sigProp = model(paraProp,t)

		covData = 1 # covariance matrix
		difProp = np.sum((sigProp-data)**2)
		difCur = np.sum((sigCur-data)**2)
		
		# handle an infinite ratio
		if (   np.isinf(np.exp((difCur-difProp)/(2*covData**2)))   ):
			hRatio = 10. # something high so that step is accepted
		else: 
			hRatio = np.exp((difCur-difProp)/(2*covData**2))
		
		choice = min(1,hRatio)
		
		u = s.uniform.rvs(0,1)
		if (u <= choice):
			paraCur = paraProp
			accept[i] = 1.
			
		
		chainArray[i,:] = paraCur
	
	acceptRate = sum(accept)/len(accept)
	
	return chainArray, acceptRate
	
def burnIn(paraCur,data,n,t):
	"""
	Burn in period
	essentially just run a chain for a fraction of the time
	
	note: Later I do want to implement something that chooses the period based
		 on the autocorrelation length.
	"""
	
	burnTime = int(round(0.1*n)) # duration of burn in
	
	for i in range(0,burnTime):
		
		# propose a jump
		covProp = eps*np.eye(len(paraCur)) # covariance matrix
		paraProp = s.multivariate_normal.rvs(paraCur,covProp)
		
		
		paraProp = checkPriors(paraProp) # check that proposed jump 
										 # is valid
	
		
		# calculate the Hastings Ratio, assuming Gaussian likelihood
		sigCur = model(paraCur,t)
		sigProp = model(paraProp,t)

		covData = 1 # covariance matrix
		difProp = np.sum((sigProp - data)**2)
		difCur = np.sum((sigCur-data)**2)
		
		# handle an infinite ratio
		if (   np.isinf(np.exp((difCur-difProp)/(2*covData**2)))   ):
			hRatio = 10. # something high so that step is accepted
		else: 
			hRatio = np.exp((difCur-difProp)/(2*covData**2))
		
		choice = min(1,hRatio)
		
		u = s.uniform.rvs(0,1)
		if (u <= choice):
			paraCur = paraProp
			
	return paraCur
	
	
	
def main(n,filename): 
	fileData = np.genfromtxt(filename,delimiter=',') # read in data
	# separate it out
	time = fileData[:,0]
	data = fileData[:,1]
	
	# intialize arbitrary parameters
	A = s.uniform.rvs(0.01,10)
	w = s.uniform.rvs(0.01,10)
	p = s.uniform.rvs(0.,np.pi)
	
	paraCur = np.array([A,w,p]) # store in an array
	
	paraCur = burnIn(paraCur,data,n,time) # burn in the chain
	
	[chainArray,acceptRate] = runChain(paraCur,data,n,time) 
	
	aChain = chainArray[:,0]
	wChain = chainArray[:,1]
	pChain = chainArray[:,2]
	itr = np.arange(0,n)
	
	"""
	plt.plot(itr,aChain,label='Amplitude')
	plt.plot(itr,wChain,label='Frequncy')
	plt.plot(itr,pChain,label='Phase')
	plt.legend()
	plt.show()
	"""
	
	return chainArray	
	
	
if __name__ == "__main__":
	main(1000,'1000Gauss0-3.csv')
