#this is a beginning ideas for the machine learning for image slices

#one idea is to use the package Keras. this works in a modular manner- and is
#was used for the EM ones

#import required packages
import numpy as np
from PIL import Image
import matplotlib.pyplot  as plt
np.random.seed(123) # for reproduciblility
import os

from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import plot_model, to_categorical
#fnumsamples = len(filearray[:0])numsamples = len(filearray[:0])rom keras.applications import vgg16, inception_v3, resnet50, mobilenet



#first we will need to read the data into an array and divide it into train and
#test data. 

#create an array of all the images 
#loop though  .../blur2_5/ to find protein names
#loop through sub directories to find file names
#create a dictionary with filenames and image files

#read document for assigning g or b and link to correct dictionary place
#combine to make an array/dataset with x_data=(image,x,y) and y_data=(catagory, 2)
#reshape data to make it explicitly declare the depth of x_data: x_data=(n,
#depth,width,height)
#change x_data to floats and normalise between 0 and 1
#divide into test and train (might have to do this earlier)

def inputTrainingImages(dirin,input_shape,fractionTrain):
  #this code is to read the image files into an array and divide it into test
  #and train data.

  #reading in data - the images should all be the same size in the end
  #we will need to redo this when the files are sorted 

  #reading the images into a filelist
  filelist = []
  #list of file paths
  for file in os.listdir(dirin):
    filelist.append(os.path.join(dirin,file))


  #reading the images into an array
  filearray=np.array([np.array(plt.imread(filename)).flatten() for filename in filelist])
  #in the line above, we may need to use .flatten() to move the file into 1D
  print('the images have been read in')
  numsamples = len(filearray[0:])
  
  label = np.ones(numsamples)

  #shuffle data
  filearray,label = shuffle(filearray,label,random_state = 2)
  print('x after shuffle: ', filearray.shape)
  print('y after shuffle: ', label.shape)

  #spliting into train and test
  ntrain = int(numsamples*fractionTrain)

  x_train = filearray[:ntrain].reshape(ntrain,input_shape[0],input_shape[1],input_shape[2])
  y_train = to_categorical(label[:ntrain],2)#this is used to create a binary class matrix

  ntest = numsamples - ntrain
  x_test = filearray[ntrain:].reshape(ntest,input_shape[0],input_shape[1],input_shape[2])
  y_test = to_categorical(label[ntrain:],2)

  batch_size = numsamples

  return x_train, x_test, y_train, y_test

#inputTrainingImages(dirin='/dls/mx-scratch/ycc62267/imgfdr/blur2_5/3S6E/3S6E_i/X',input_shape=[160,288,3],fractionTrain= 0.8)

x_train, x_test, y_train, y_test =inputTrainingImages(dirin='/dls/mx-scratch/ycc62267/imgfdr/blur2_5/3S6E/3S6E_i/X',input_shape= [160,288,3],fractionTrain=0.8)


#define the model architecture
#should probably design a model or at least think about the most helpful. 
#start by trying Sequential model format - read more about this 
#start with the input layer, - look at what the kernel_initialiser does (and
#padding)
#build up layers here
#end the final tayer with an output size of 2 to be the g or b label for the
#image


#creating a model
class mapModel(Sequential):
  #have chosen a sequential model
  
  def __init__(self):
    Sequential.__init__(self)

  def createCustom1(self,input_shape1):
    #mess with these once code is working... currently from keras tutorial -
    #generic setup
    self.add(Convolution2D(32,(3,3),activation ='relu',input_shape=input_shape1))
    self.add(Convolution2D(32,(3,3),activation = 'relu'))
    self.add(MaxPooling2D(pool_size =(2,2)))
    self.add(Dropout(0.25))

    self.add(Flatten())
    self.add(Dense(128,activation='relu'))
    self.add(Dropout(0.5))
    self.add(Dense(2,activation = 'softmax'))



model = mapModel()
model.createCustom1(input_shape1=[160,288,3])








#compile the model
#loss function probably using binary_crossentropy
#metrics may be ['accuracy'] but should check
#optimiser could be opt=keras.optimizers.SGD(lr=0.0001), but should check

#initiate RMSprop optimizer - mess with this once code is working
#opt = keras.optimizers.Adam

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
print('model compiled')



#fit the model on the training data- this is the bit that trains the model -
#varies the weights.
#model.fit(x_train,y_train, etc..) look at the meaning of augmenting the data

model.fit(x_train,y_train,batch_size = 7, epochs=5, verbose =1)
#verbose gives a progress bar



#evaluate the model on the test data
#this sees how good the model is by applying it to the test data

score = model.evaluate(x_test, y_test, verbose =1)
print(score)

#
