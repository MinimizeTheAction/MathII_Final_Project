"""
This is a set of tools I commonly use while performing MCMC analysis.
"""

import numpy as np
import scipy.stats as s
import scipy.signal as sig


def autoCorrLen(data):
	"""
	Calculates the autocorrelation length. Starting from 0 lag (each lag is 
	scaled by 1/(N-lag), N being length of data array), this finds where the 
	autocorrelation scaled by zero-lag autocorrelation drops to 0.01 first.
	
	parameters:
			data - 1-D array of numbers
	return:
			autoLen - autocorrelation length
			corr - autocorrelation
			
	note: later I want to instead implement a linear fit in log space to find
		  the exponential scale factor then use that to solve for the lag which	
		  the amplitude drops to 0.01 instead
		  i.e. autocorrLen = - scale ln(0.01) where scale is the -1/slope of the
		  fit
	
	"""
	arg = data - data.mean()
	
	# calculate the autocorrelation and lag array
	corr = sig.correlate(arg,arg)
	center = np.floor(len(corr)/2)+1 # find center index
	corr = corr[center:len(corr)]/corr[center] # restrict to positive lag
	where = np.where(corr<0.01) # find where autocorr equals
	
	autoLen = float(where[0][0]) # normalize by length of array
	
	return [autoLen, corr]
	
def basicMCMC(paraCur,model,priors,data,n,jumpSig,burn,t=None):
	"""
	Run a basic MCMC - Metropolis Hastings. Assumes a guassian likelihood between the
	model and data. Jump are proposed from a Gaussian centered at the current location
	in parameter space. Can also perform burn in period.
	
	parameters:
		paraCur - current array of parameters 
		model - function that specifies model for data
		data - data array to calculate likelihood
		n - length of chain
		jumpSig - standard deviation of proposal distribution
		burn - if 0, not a burn in, if non-zero burnTime
		t (optional) - any other parameters needed for data model that aren't being fit
		
	return:
		paraChain - array of the chains generated
	"""
	if (burn == 0): # not a burn in
		chainArray = np.zeros([n,len(paraCur)]) # initialize array to hold chain links
		accept = np.zeros(n) # to analyze acceptance rates
		runtime = n
	else:
		runtime = int(burn)
	
	for i in range(0,runtime):
		# propose a jump
		paraProp = s.multivariate_normal.rvs(paraCur,jumpSig)
		
		# check that proposed jump is valid
		paraProp = priors(paraProp)  		 
		
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
		
		# select the minimum
		choice = min(1,hRatio)
		
		# decide whether to accept the jump or not
		u = s.uniform.rvs(0,1)
		if (u <= choice):
			paraCur = paraProp
			if (burn == 0): # not burn in
				accept[i] = 1.
		# if not burn in 		
		if (burn == 0): # not burn in
			chainArray[i,:] = paraCur
	
	if (burn == 0): # not burn in
		acceptRate = sum(accept)/len(accept)
		return chainArray,acceptRate
	else: # burn in
		return paraCur
	
	
	
	

	
