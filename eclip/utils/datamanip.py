################
'''Functions to find and manipulate data for learning'''
##############
import numpy as np
import sqlite3
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.utils import shuffle, class_weight
from sklearn.metrics import confusion_matrix
import random

from keras.utils import to_categorical

####################################
def normalarray(n_array):
  maxi=np.amax(n_array)
  mini=np.amin(n_array)
  norm_array=(n_array-mini)/(maxi-mini)
  return norm_array

def inputTrainingImages(database,input_shape,fractionTrain):

  conn=sqlite3.connect(database)
  cur=conn.cursor()
  
  #Protein_id is a list of the pdb_id_ids from the database
  cur.execute(''' SELECT pdb_id_id FROM Phasing''')
  Protein_id=cur.fetchall()

  #Create a list of directory names (x_dir) and a list of the scores (y_list)
  x_dir=[]
  y_list=[]

  for item in Protein_id:
    cur.execute('''SELECT ep_success_o FROM Phasing WHERE pdb_id_id = "%s"'''%item)
    y_o=list(cur.fetchall()[0])[0]
    if not y_o==None:
      y_list.append(y_o)
      cur.execute('''SELECT ep_img_o FROM Phasing WHERE pdb_id_id =
      "%s"'''%item)
      x_o=list(cur.fetchall()[0])[0]
      x_dir.append(x_o)
    cur.execute('''SELECT ep_success_i FROM Phasing WHERE pdb_id_id =
    "%s"'''%item)
    y_i=list(cur.fetchall()[0])[0]
    if not y_i == None:
      y_list.append(y_i)
      cur.execute('SELECT ep_img_i FROM Phasing WHERE pdb_id_id = "%s"'''%item)
      x_i=list(cur.fetchall()[0])[0]
      x_dir.append(x_i)

  #x_dir is a list of directories, in these directories there are X,Y,Z
  #directories, in each of these there are files.

  #Create an array of the images using x_dir (and keeping this same order!!)
  filelist=[]
  label=[]

  for i in range(0,len(y_list)):
    dirin=x_dir[i]
    for dir in os.listdir(dirin):
      img_dir=os.path.join(dirin,dir)

      #chosing random files from directory
      for m in range(1,5):
        filename=random.choice(os.listdir(img_dir))
        location=os.path.join(img_dir,filename)
        filelist.append(location)
        label.append(y_list[i])

  filearray=np.array([normalarray(np.array(plt.imread(filename))).flatten() for filename in filelist])
  print('images have been read in')
  numsamples = len(filearray)

  #label is a list of scores for each image
  #filearray is an array of each image

  #need to make label an array
  label=np.asarray(label)

  #label is an array of scores for each image
  #filearay is an array of each image

  #need to shuffle the data:
  filearray,label=shuffle(filearray,label,random_state=2)
  print('x after shuffle: ',filearray.shape)
  print('y after shuffle: ',label.shape)
  
  ntrain=int(numsamples*fractionTrain)

  '''it may be easier to keep them as lists for now and change in the next bit of
  the code???'''


  x_train=filearray[:ntrain].reshape(ntrain,input_shape[0],input_shape[1],input_shape[2])
  y_train=to_categorical(label[:ntrain],2) #this is used to create a binary class matrix

  ntest=numsamples-ntrain
  x_test = filearray[ntrain:].reshape(ntest,input_shape[0],input_shape[1],input_shape[2])
  y_test=to_categorical(label[ntrain:],2)

  return x_train,y_train,x_test,y_test











