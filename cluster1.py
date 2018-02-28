
import os
import sys
import numpy as np
from sklearn.cluster import KMeans

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)
#  scikit-learn (0.19.1)


def get_kmean(test_data, n_clusters0=2):
	# Clustering via K means method
	kmeans0= KMeans(n_clusters=n_clusters0,verbose=0).fit( test_data)
	centroids = kmeans0.cluster_centers_
	#print kmeans0.cluster_centers_
	labels=kmeans0.predict(test_data)
	#print labels
	return centroids

def get_means_main(test_data, test_length, NSTATES=3, NMIX=2):
	# divide test_data into NSTATES portion
	# get means as centroids of the cluster of each portion
	
	for j in range (NSTATES):
		count=0
		s0=0
		for k,l0 in enumerate(test_length):
			if l0 < NSTATES:
				s0+=l0
				continue
			l=l0/NSTATES
			s0=s0 + l * j
			s1=s0+l
			if j == (NSTATES-1):
				s1=s0+l0
			if count == 0:
				sub_test=test_data[s0:s1]
			else:
				sub_test=np.append(sub_test, test_data[s0:s1],axis=0)
			s0+=l0
			count+=1
		# get centroids
		if j ==0:
			means=[get_kmean(sub_test, n_clusters0=NMIX)]
		else:
			means=np.append( means, [get_kmean(sub_test, n_clusters0=NMIX)],axis=0)
	return means

if __name__ == '__main__':

	from get_dataset1 import *
	
	# check
	NDIM0=10
	NMIX0=2
	NSTATES0=3
	
	train_data, train_length = get_DatSet(ndimension=NDIM0)
	means=get_means_main(train_data[0], train_length[0], NSTATES=NSTATES0, NMIX=NMIX0 )
	
	print (means)

# This file uses TAB.






