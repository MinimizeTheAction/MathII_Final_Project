"""
This is a set of tools I commonly use while performing MCMC analysis.
"""

import numpy as np
import scipy.stats as stat
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
	
	autoLen = float(where[0][0])/float(len(corr)) # normalize by length of array
	
	return autoLen
	

	
