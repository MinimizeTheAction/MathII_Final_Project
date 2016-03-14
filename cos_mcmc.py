"""
Implements mcmc with Metropolis-Hastings algorithm to extract the amplitude, 
angular frequency, and phase of a noisy signal.
"""
import numpy as np
import scipy.stats as s
import matplotlib.pyplot as plt
import mcmcTools as mct


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
	w -- [0.01,10]
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
	
	if (w<0.01 or w>10.0): w = s.uniform.rvs(0.01,10.)
		
	if ( p<0. or p>2*np.pi): p = s.uniform.rvs(0.0,2*np.pi)
	
	return np.array([A,w,p])	
	
	
def main(n,filename): 
	np.seterr(over='ignore') # to ignore overflow errors
	fileData = np.genfromtxt(filename,delimiter=',') # read in data
	# separate it out
	time = fileData[:,0]
	data = fileData[:,1]
	
	# intialize arbitrary parameters
	A = s.uniform.rvs(0.01,10)
	w = s.uniform.rvs(0.01,10)
	p = s.uniform.rvs(0.,np.pi)
	
	paraCur = np.array([A,w,p]) # store in an array
	
	# burn in the chain
	paraCur = mct.basicMCMC(paraCur,model,checkPriors,data,n, \
									0.01,0.05*n,time)
	# run mcmc
	[chainArray, acceptRate] = mct.basicMCMC(paraCur,model,checkPriors,data,n, \
									0.01,0,time)
	
	# uncomment block for use
	print acceptRate*100
	
	aChain = chainArray[:,0]
	wChain = chainArray[:,1]
	pChain = chainArray[:,2]
	itr = np.arange(0,n)
	
	plt.subplot(3,1,1)
	plt.plot(itr,aChain,label='Amplitude')
	plt.plot(itr,np.ones(len(itr)),'-.',label='Actual')
	plt.legend()
	plt.subplot(3,1,2)
	plt.plot(itr,wChain,label='Frequncy')
	plt.plot(itr,3*np.ones(len(itr)),'-.',label='Actual')
	plt.legend()
	plt.subplot(3,1,3)
	plt.plot(itr,pChain,label='Phase')
	plt.plot(itr,np.ones(len(itr)),'-.',label='Actual')
	plt.legend()
	plt.xlabel('Chain Link')
	plt.show()
	
	
	return chainArray	
	
	
if __name__ == "__main__":
	main(10000,'100Gauss0-3.csv')
