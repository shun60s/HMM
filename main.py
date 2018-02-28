
import os
import sys
import numpy as np
from HMM import *
from get_dataset1 import *


# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)
#  scikit-learn (0.19.1)



def make_GHMM(test_data, test_length,  n_components, nmix):
	h_all=['h0','h1','h2','h3','h4','h5','h6','h7','h8','h9']
	#h_all=['h0']
	
	for i in range(len(h_all)):
		print ('h_all of ', h_all[i])
		h_all[i]=GHMM_sub(test_data[i],test_length[i], n_components, nmix) 
		h_all[i].fit(test_data[i],test_length[i] ) 
		#print ( h_all[i].means )
		#print ( h_all[i].transmat )
	return h_all


def get_scores(h_all,test_data, test_length, TITLE0='SCORE'):
	s_all=['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9']
	
	print (TITLE0)
	count=0
	tcount=0
	for k in range(len(s_all)):
		print ('correct label', k)
		for i in range(len(s_all)):
			s_all[i]= h_all[i].score2(test_data[k],test_length[k])
		
		for i in range (len(test_length[k])):
			a= [s_all[j][i] for j in range(len(s_all))] 
			if  np.argmax(a) == k:
				tcount+=1
			else:
				print ('  incorrcet label', np.argmax(a), 'Value',  a[k] , '<' , a[np.argmax(a)])
			count+=1

	print ('number of samples', count)
	print ('accuracy rate [%]', (tcount * 100.0) /count)

	return count, tcount


def json_out(NDIM0, NMIX0, NSTATES0, train_data, count, tcount, count2, tcount2):
	import json
	from collections import OrderedDict

	OUT_DIR = 'result'
	OUTPUT_JSON='result.json'
	
	if not os.path.exists(OUT_DIR):
		os.mkdir(OUT_DIR)
	
	trst=OrderedDict()
	rst=OrderedDict()
	
	trst["number of  sequences"]=NDIM0
	trst["number of states in the GMM"]=NMIX0
	trst["number of states in the model"]=NSTATES0
	
	rst=OrderedDict()
	rst["transformed.shape"]= get_DatSet_shape()
	trst["transformed data"]=rst
	
	rst=OrderedDict()
	rst["number of samples"]=  count 
	rst["accuracy rate [%]"]=  (tcount * 100.0) /count 
	trst["train_data"]=rst
	
	rst=OrderedDict()
	rst["number of samples"]=  count2 
	rst["accuracy rate [%]"]= (tcount2 * 100.0) /count2
	trst["test_data"]=rst
	
	f=open( os.path.join(OUT_DIR, OUTPUT_JSON), 'a')  # append mode
	json.dump(trst,f, indent=2)
	print ('result was wrote in ', os.path.join(OUT_DIR, OUTPUT_JSON))
	f.close()

if __name__ == '__main__':

	NDIM0=10    # number of  sequences
	NMIX0=2     # number of states in the GMM
	NSTATES0=3  # number of states in the model
	
	
	train_data, train_length = get_DatSet(ndimension=NDIM0)
	h_all= make_GHMM(train_data, train_length, n_components=NSTATES0, nmix=NMIX0)
	count, tcount=get_scores(h_all,train_data, train_length, TITLE0='---train data score---')
	
	print ('')
	test_data, test_length = get_DatSet(FILE_NAME='test_transformed', ndimension=NDIM0)
	count2, tcount2=get_scores(h_all,test_data, test_length, TITLE0='---test data score---')
	
	json_out(NDIM0, NMIX0, NSTATES0, train_data, count, tcount, count2, tcount2)
	

# This file uses TAB.






