
import numpy as np
from get_dataset1 import *
import matplotlib.pyplot as plt


# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)

# Show distribution of train data and test data

if __name__ == '__main__':

	
	train_data, train_length = get_DatSet()
	test_data, test_length = get_DatSet(FILE_NAME='test_transformed')
	
	
	
	print('distribution: Blue is train. Red is test')
	for k in range (10):
		
		plt.subplot(5,2,k+1)
		plt.title( 'number ' + str(k))
		sp0=0
		sp1=0
		for i in range (len(train_length[k])):
			x=np.array([])
			y=np.array([])
			for j in range (train_length[k][i]) :
				x = np.append( x, train_data[k][sp0,0]) 
 				y = np.append( y, train_data[k][sp0,1])
 				sp0+=1
			plt.scatter(x, y, c='b',marker='x', alpha=0.1)
			
		for i in range (len(test_length[k])):
			x=np.array([])
			y=np.array([])
			for j in range (test_length[k][i]) :
				x = np.append( x, test_data[k][sp1,0]) 
 				y = np.append( y, test_data[k][sp1,1])
 				sp1+=1
			plt.scatter(x, y, c='r',marker='o', alpha=0.1)			
			
	plt.show()


# This file uses TAB.






