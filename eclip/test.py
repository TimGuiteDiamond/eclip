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
from sklearn.utils import class_weight
import keras
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.utils import plot_model, to_categorical
from keras.applications import vgg16
################################################################################################


#############################################################################################
import sqlite3
def normalarray(n_array):
  maxi=np.amax(n_array)
  mini=np.amin(n_array)
  norm_array=(n_array-mini)/(maxi-mini)
  return norm_array

def inputTrainingImages(database,input_shape,Protein_id):
  conn=sqlite3.connect(database)
  cur=conn.cursor()

  
  #Protein_id is a list of the pdb_id_ids from the database
  #cur.execute('''SELECT pdb_id_id FROM Phasing''')
  #Protein_id=cur.fetchall()

  #Create a list of directory names (x_dir) and a list of the scores (y_list)
  x_dir=[]
  y_list=[]
  #get rid of m for the real thing- cuts down the number of samples used
  m=0
  for item in Protein_id:
    m+=1
    if m>4:
      break
    cur.execute('''SELECT ep_success_o FROM Phasing WHERE pdb_id_id = "%s"'''%item)
    y_o=list(cur.fetchall()[0])[0]
    if not y_o==None:
      y_list.append(y_o)
      cur.execute('''SELECT ep_img_o FROM Phasing WHERE pdb_id_id = "%s"'''%item)
      x_o=list(cur.fetchall()[0])[0]
      x_dir.append(x_o)
    cur.execute('''SELECT ep_success_i FROM Phasing WHERE pdb_id_id =
    "%s"'''%item)
    y_i=list(cur.fetchall()[0])[0]
    if not y_i==None:
      y_list.append(y_i)
      cur.execute('''SELECT ep_img_i FROM Phasing WHERE pdb_id_id="%s"'''%item)
      x_i=list(cur.fetchall()[0])[0]
      x_dir.append(x_i)
  
  #x_dir is a list of directories, in these directories there are X,Y,Z
  #directories, in each of these there are files.
  
  #Create an array of the images using x_dir (and keeping the same order!!! -
  #CHECK THIS!!)
  filelist=[]
  label=[]
  for i in range(0,len(y_list)):
    dirin = x_dir[i]
    for dir in os.listdir(dirin):
      img_dir=os.path.join(dirin,dir)
      for file in os.listdir(img_dir):
        location=os.path.join(img_dir,file)
        filelist.append(location)
       # print(location)
      #  print('adding %s to filelist'%file)
        label.append(y_list[i])
  
  #Normalising:
  filearray=np.array([normalarray(np.array(plt.imread(filename))).flatten() for filename in
  filelist])
  print('images have been read in')
  numsamples = len(filearray)
  print(numsamples)
  print(filearray.shape)
  #print(label) 
  #label is a list of scores for each image
  #filearray is an array of each image
  
  #need to make label and array:
  label=np.asarray(label)
  #print(label)
  #label is an array of scores for each image
  #filearray is an array of each image
  
  #need to shuffle the data:
  filearray,label=shuffle(filearray,label,random_state=2)
  print('x after shuffle: ', filearray.shape)
  print('y after shuffle: ', label.shape)

#setting all y=0 to all zeros for x - testing
  for i in range(0,len(label)):
    if label[i] ==0:
      #print(filearray[i].shape)
      length=len(filearray[i])
      filearray[i]=filearray[i]/1

  ntrain=len(filearray)
  x_train = filearray.reshape(ntrain,input_shape[0],input_shape[1],input_shape[2])
  y_train = to_categorical(label,2)#this is used to create a binary class matrix
#  y_train=label[:ntrain]
 
  return x_train, y_train

#inputTrainingImages(dirin='/dls/mx-scratch/ycc62267/imgfdr/blur2_5/3S6E/3S6E_i/X',input_shape=[160,288,3],fractionTrain= 0.8)

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
    self.add(Convolution2D(32,(3,3),kernel_initializer.he_normal(),activation ='relu',input_shape=input_shape1))
    self.add(Convolution2D(32,(3,3),activation = 'relu'))
    self.add(MaxPooling2D(pool_size =(2,2)))
    self.add(Dropout(0.25))

    self.add(Flatten())
    self.add(Dense(128,activation='relu'))
    self.add(Dropout(0.5))
    self.add(Dense(2,activation = 'softmax'))

  def createCustom2(self,input_shape2):
    #another model, with more layers and a different final layer
    #padding??
    self.add(Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(),activation='relu',padding='same',input_shape=input_shape2))
    self.add(MaxPooling2D(pool_size=(2,2)))
    self.add(BatchNormalization())
    self.add(Convolution2D(64,(3,3),activation='relu',padding='same'))
    self.add(Convolution2D(64,(3,3),activation='relu',padding='same'))
    self.add(MaxPooling2D(pool_size=(2,2)))
    self.add(BatchNormalization())
    self.add(Convolution2D(128,(3,3),activation='relu',padding='same'))
    self.add(MaxPooling2D(pool_size=(2,2)))
    self.add(Flatten())
   # self.add(Dropout(0.5))
    self.add(BatchNormalization())
    self.add(Dense(128,activation='relu'))
    self.add(Dense(2,activation='softmax'))

  def createCustom3(self,input_shape3):
    #another model
    self.add(Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(),padding='same',activation='relu',input_shape=input_shape3))
    self.add(Convolution2D(32,(3,3),activation = 'relu'))
    self.add(MaxPooling2D(pool_size=(2,2)))
 #   self.add(Dropout(0.25))

 #   self.add(Convolution2D(64,(3,3),padding='same', activation='relu'))
 #   self.add(Convolution2D(64,(3,3),activation='relu'))
 #   self.add(MaxPooling2D(pool_size=(2,2)))
 #   self.add(Dropout(0.25))

    self.add(Flatten())
    self.add(Dense(80,kernel_initializer=initializers.he_normal(),activation='relu'))
 #   self.add(Dropout(0.5))
    self.add(Dense(2,activation='softmax'))


####################################################################################

x_train,y_train=inputTrainingImages(database='/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite',input_shape=[201,201,3],Protein_id=[584,504,114])

x_test, y_test=inputTrainingImages(database='/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite',input_shape=[201,201,3],Protein_id=
[334])

#x_train, x_test, y_train, y_test=inputTrainingImages(database='/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite',input_shape=[201,201,3],fractionTrain=0.8)

ones=sum(y_train)
length=len(y_train)
zeros=length-ones
ratio=ones/zeros

y_ints=[y.argmax() for y in y_train]
class_weights=dict(enumerate(class_weight.compute_class_weight('balanced',np.unique(y_ints),y_ints)))
print(class_weights)
print(len(y_train))
print(sum(y_train))


model = mapModel()
model.createCustom2(input_shape2=[201,201,3])

#compile the model
#optimiser could be opt=keras.optimizers.SGD(lr=0.0001), but should check
#initiate RMSprop optimizer - mess with this once code is working
opt = keras.optimizers.Adam(lr=0.0001)

model.compile(loss = 'categorical_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])
print('model compiled')



#fit the model on the training data- this is the bit that trains the model -
#varies the weights.
#look at the need for augmenting the data- we have a lot so it may not be
#necessary
print(y_test.shape)
history=model.fit(x_train,y_train,class_weight=class_weights,batch_size = 15,
epochs=30, verbose=1,validation_split=(0.33))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.show()
#verbose gives a progress bar

prediction=model.predict(x_test,batch_size=32,verbose=1)
print(prediction)

#evaluate the model on the test data
#this sees how good the model is by applying it to the test data

loss,acc = model.evaluate(x_test, y_test, verbose =1)
print('loss is: %s'%loss)
print('accuracy is: %s'%acc)

