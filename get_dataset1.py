
import os
import numpy as np

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)


def get_DatSet(IN_DIR = 'DataSet', FILE_NAME='transformed', ndimension=5, nstart=0):
	transformed = np.load(os.path.join(IN_DIR, FILE_NAME + '.npy'))
	print (FILE_NAME + '.shape', transformed.shape)
	train_label = np.load(os.path.join(IN_DIR, FILE_NAME + '_label.npy'))
	NUM0= max(train_label) +1  # number of digit
	#print ('max_of_train_label + 1', NUM0 )
	train_length = np.load(os.path.join(IN_DIR, FILE_NAME + '_length.npy'))
	#print ('number of whole samples ', len(train_length))
	test_data = [[] for row in range(NUM0)]
	test_length = [[] for row in range(NUM0)]
	
	clist0=np.zeros(NUM0,dtype=np.int32)  # each number count
	tcount=0  # total count
	for (i,l) in  enumerate(train_length):
		index=train_label[i]
		loop=train_length[i]
		
		if clist0[index] == 0:
			test_length[index]=loop
		else:
			test_length[index]=np.append( test_length[index],loop)
		
		for j in range(loop):
			if clist0[index] == 0:
				test_data[index]=[transformed[tcount,nstart:ndimension]]
			else:
				test_data[index]=np.append(test_data[index], [transformed[tcount,nstart:ndimension]], axis=0)
			tcount+=1
			clist0[index]+=1
			
	#print ('sum of test_length ',np.sum(test_length))
	return test_data, test_length


def get_DatSet_shape(IN_DIR = 'DataSet', FILE_NAME='transformed'):
	transformed = np.load(os.path.join(IN_DIR, FILE_NAME + '.npy'))
	return  transformed.shape

# This file uses TAB.
