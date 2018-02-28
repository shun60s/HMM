
import numpy as np
from cluster1 import *
from hmmlearn import hmm

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)
#  hmmlearn (0.2.0)


class GHMM_sub:
	"""  subset of Hidden Markov Model with Gaussian mixture emissions,  Left-to-right (one way)  """
	"""  n_components is number of hidden states          """
	
	def __init__(self,test_data,test_length, n_components, n_mix):
		
		self.n_components = n_components
		self.n_mix= n_mix
		# c: covars
		# m: means
		# w: GMM mixing weights
		# t: transmat
		self.model = hmm.GMMHMM(n_components=self.n_components, n_mix=self.n_mix, n_iter=100, covariance_type="diag", init_params="c", params="cmtw")
		
		# set mean
		means= get_means_main(test_data,test_length, NSTATES=self.n_components, NMIX=self.n_mix)
		self.model.means_ = means
		
		# set equal weight 
		weights= np.ones((self.n_components, self.n_mix)) / (1.0 * self.n_mix)
		self.model.weights_= weights
		
		startprob = np.zeros(self.n_components)
		# start always n_component 0:  startprob is FIXED
		startprob[0] = 1.0  
		self.model.startprob_ = startprob
		
		transmat =  np.zeros((self.n_components, self.n_components))
		# Left-to-right: 
		for i in range(self.n_components):
			if i == self.n_components - 1:
				transmat[i, i] = 1.0
			else:
				transmat[i, i] = transmat[i, i + 1] = 0.5
		#print transmat
		
		self.model.transmat_ = transmat

	def fit(self,Xsum,lsum):
		self.model.fit(Xsum,lsum)
		print (self.model.monitor_ )
		
	def predict(self,Xsum,lsum):
		state_sequence = self.model.predict(Xsum,lsum)
		# check start state
		sp0=0
		for i in range( len(lsum)):
			print ('i, state_sequence[i]', i, state_sequence[sp0])
			sp0+=lsum[i]
		return state_sequence
		
	def predict_proba(self,Xsum,lsum):
		posteriors = self.model.predict_proba(Xsum,lsum)
		return posteriors
		
	def score2(self,Xsum,lsum):
		# each score, not whole value 
		logprob=np.zeros(len(lsum)) 
		sp0=0
		for i in range( len(lsum)):
			#print ( Xsum[sp0:sp0+lsum[i]] )
			logprob[i] = self.model.score( Xsum[sp0:sp0+lsum[i]] )
			sp0+=lsum[i]
		return logprob

	@property
	def means(self):
		return self.model.means_
	
	"""  can not find covars_ attribute
	@property
	def covars(self):
		return self.model.covars_
	"""
	
	@property
	def transmat(self):
		return self.model.transmat_


# this file use TAB

