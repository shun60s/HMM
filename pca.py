
import os
import sys
import random
import numpy as np
import skimage.io
import skimage.draw
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)
#  scikit-image (0.13.1)
#  scikit-learn (0.19.1)
#  matplotlib (2.1.2)

# Test persons , excluded from train data
TestNames=['Samantha', 'Tom']

# frame length  ex: 2 continuous frames as one data set
N_CONT0=2

IN_DIR = 'spectrogram'
OUT_DIR = 'DataSet'

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)


train_data = np.array([]).astype(np.float32)
train_label = []
train_length = []

test_data = [[]]  # np.array([]).astype(np.float32)
test_label = []
test_length = []


# name => (number)
number_dics = {}

with open('labels.txt') as fp:
    for line in fp:
        line = line.rstrip()
        cols = line.split()
        assert len(cols) == 2, ' Not expect input format'
        number = int(cols[0])
        name = cols[1]
        number_dics[name] = (number)


count=0
test_count=0
# random scan 
files=os.listdir(IN_DIR)
random.shuffle(files)
for f in files:
    # get head letter
    assert f[0:1].isdigit(), ' Not expect input file name'
    idx = int( f[0:1] )
    
    for name in TestNames:
        index=f.find( name)
        if index != -1:
        	break   # goto numer_dics
    
    for name in number_dics:
        if number_dics[ name ] == idx:
            print (f)
            source = os.path.join(IN_DIR, f)
            image = skimage.img_as_float(skimage.io.imread(source)).astype(np.float32)
            image=image.T
            print ('image.shape', image.shape)
            label=np.int32(idx)
            
            if index != -1:    # test persons
                if test_count == 0:
            	     test_data=image
            	     #print image
                else:
                     test_data=np.append(test_data, image ,axis=0)
                test_label.append(label)
                test_length.append(image.shape[0])
                #print train_data
                #print train_label
                test_count+=1
            
            else:              # train persons, except test persons
                if count == 0:
            	     train_data=image
            	     #print image
                else:
                     train_data=np.append(train_data, image ,axis=0)
                train_label.append(label)
                train_length.append(image.shape[0])
                #print train_data
                #print train_label
                count+=1
            
            break   # next file



print ('count ', count)
print ('train_data.shape', train_data.shape)
print ('test_count ', test_count)
print ('test_data.shape', test_data.shape)



# 1st get memory and then set the value
# because np.append memory allocation process is too slow
new_total=train_data.shape[0] - (len(train_length) * (N_CONT0 - 1))
new_dim=train_data.shape[1] * N_CONT0

train_cdata = np.zeros([new_total, new_dim]).astype(np.float32)
train_clabel = []
train_clength = []


count0=0
tcount=0
for l in range( len(train_length)):
    #print ('l', l)
    for i in range( train_length[l] - (N_CONT0 -1) ):
        train_cdata[count0]=np.array(np.hstack( train_data[tcount+i+j] for j in range (N_CONT0)))
        count0+=1
    train_clength.append(train_length[l] - (N_CONT0-1))
    train_clabel.append(train_label[l])
    tcount+=train_length[l]


print ('train_clabel.len', len(train_clabel))
print ('train_clength.len', len(train_clength))



new_total=test_data.shape[0] - (len(test_length) * (N_CONT0 - 1))
new_dim=test_data.shape[1] * N_CONT0

test_cdata = np.zeros([new_total, new_dim]).astype(np.float32)
test_clabel = []
test_clength = []


count0=0
tcount=0
for l in range( len(test_length)):
    #print ('l', l)
    for i in range( test_length[l] - (N_CONT0 -1) ):
        test_cdata[count0]=np.array(np.hstack( test_data[tcount+i+j] for j in range (N_CONT0)))
        count0+=1
    test_clength.append(test_length[l] - (N_CONT0-1))
    test_clabel.append(test_label[l])
    tcount+=test_length[l]

print ('test_clabel.len', len(test_clabel))
print ('test_clength.len', len(test_clength))



#PCA : Principal component analysis
print ('Principal component analysis')
pca = PCA()
transformed = pca.fit_transform(train_cdata)  # fit and transform
print('transformed.shape', transformed.shape)

# Explained (kiyoritu)
ev_ratio = pca.explained_variance_ratio_
ev_ratio = np.hstack([0,ev_ratio.cumsum()])
plt.title('explained variance')

plt.plot(range(1,len(ev_ratio)+1), ev_ratio * 100.)
plt.xlabel('n_component')
plt.ylabel('percentage[%]')
plt.show()

# plot 3D figure of 1-3rd factors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# each transformed data and label
np.save(os.path.join(OUT_DIR,'transformed.npy'), transformed)
np.save(os.path.join(OUT_DIR,'transformed_label.npy'), train_clabel)
np.save(os.path.join(OUT_DIR,'transformed_length.npy'), train_clength)

# test data
test_transformed = pca.transform(test_cdata)
print('test_transformed.shape', test_transformed.shape)
np.save(os.path.join(OUT_DIR,'test_transformed.npy'), test_transformed)
np.save(os.path.join(OUT_DIR,'test_transformed_label.npy'), test_clabel)
np.save(os.path.join(OUT_DIR,'test_transformed_length.npy'), test_clength)

# plot scatter
ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2])
ax.scatter(test_transformed[:, 0], test_transformed[:, 1], test_transformed[:, 2],c='r')
ax.set_title('Principal component analysis: Blue is train, Red is test')
ax.set_xlabel('1st principal component')
ax.set_ylabel('2nd principal component')
ax.set_zlabel('3rd principal component')
plt.show()