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

  '''
  **Arguments for normalarray:**

  * **n_array:** An array

  **Outputs of normalarray:***

  * **norm_array:** A normalised version of the input array with values between 0 and 1.

  '''

  maxi=np.amax(n_array)
  mini=np.amin(n_array)
  norm_array=(n_array-mini)/(maxi-mini)
  return norm_array

def inputTrainingImages(database,input_shape,fractionTrain,raw = False):
  '''

  **Arguments for inputTrainingImages:**

  * **database:** The file location of an sqlite database with the correct format
  * **input_shape:** The dimensions of the image files to be retrieved
  * **fractionTrain:** The fraction of the data retrieved to be used to test the model
  * **raw:** parameter specifying whether to take the raw or processed imagedata default = False

  **Outputs of inputTrainingImages:**

  * **x_train:** An array of the image data from all the images read in for training
  * **y_train:** A two dimensional array of the true score of the images for training
  * **x_test:** An array of the image data from all the images read in for testing
  * **y_test:** A two dimensional array of the true score of the images for training

  '''

  trialsplit=True

  conn=sqlite3.connect(database)
  cur=conn.cursor()
  
  #Protein_id is a list of the pdb_id_ids from the database
  cur.execute(''' SELECT pdb_id_id FROM Phasing''')
  Protein_id=cur.fetchall()

  #Create a list of directory names (x_dir) and a list of the scores (y_list)
  x_dir=[]
  y_list=[]

  t=open('/dls/science/users/ycc62267/eclip/eclip/trialsplit.txt','w')
  if trialsplit == True:
    for i in range(0,51):
      name_id=random.choice(Protein_id)
      Protein_id.remove(name_id)
      cur.execute('''SELECT pdb_id FROM PDB_id WHERE id = "%s"'''%(name_id))
      name_str=cur.fetchall()
      t.write(str(name_id)+' '+str(name_str)+'\n')
  t.close()

  for item in Protein_id:
    cur.execute('''SELECT ep_success_o FROM Phasing WHERE pdb_id_id = "%s"'''%item)
    y_o=list(cur.fetchall()[0])[0]
    if not y_o==None:
      y_list.append(y_o)
      if raw:
        cur.execute('''SELECT ep_raw_o FROM Phasing WHERE pdb_id_id =
        "%s"'''%item)
        x_o=list(cur.fetchall()[0])[0]
      else:
        cur.execute('''SELECT ep_img_o FROM Phasing WHERE pdb_id_id =
        "%s"'''%item)
        x_o=list(cur.fetchall()[0])[0]
      x_dir.append(x_o)
    cur.execute('''SELECT ep_success_i FROM Phasing WHERE pdb_id_id =
    "%s"'''%item)
    y_i=list(cur.fetchall()[0])[0]
    if not y_i == None:
      y_list.append(y_i)
      if raw:
        cur.execute('''SELECT ep_raw_i FROM Phasing WHERE pdb_id_id = "%s"'''%item)
        x_i = list(cur.fetchall()[0])[0]
      else:
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
      for m in range(1,10):
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

  #splitting into train and test
  x_train=filearray[:ntrain].reshape(ntrain,input_shape[0],input_shape[1],input_shape[2])
  y_train=to_categorical(label[:ntrain],2) #this is used to create a binary class matrix

  ntest=numsamples-ntrain
  x_test = filearray[ntrain:].reshape(ntest,input_shape[0],input_shape[1],input_shape[2])
  y_test=to_categorical(label[ntrain:],2)

  return x_train,y_train,x_test,y_test


def importData(datafileloc,proteinlist, input_shape):
  '''Function to import new data to predict

  **Arguments for importData:**

  * **datafileloc:** the directory where the protein directories containing  X,Y,Z directories containing the image files are kept
  * **proteinlist:** A list of the proteins to select from this directory
  * **input_shape:** The shape of a individual image to be selected 

  **Outputs for importData:**

  * **x_pred:** An array containing the data for the selected images
  * **name:** A list of the names of the proteins associated with each image

  '''
  name = []
  filelist = []
  protein_list = []
  for protein in proteinlist:
    path = os.path.join(datafileloc,protein)
    for dir in os.listdir(path):
      protein_name=dir
      path = os.path.join(datafileloc,protein,protein_name)
      protein_list.append(protein_name)
      for dir in os.listdir(path):
        path2= os.path.join(path,dir)
        for m in range(0,51):
          file = random.choice(os.listdir(path2))
          location= os.path.join(path2,file)
          filelist.append(location)
         
          name.append(protein_name)

  #making an array
  filearray=np.array([normalarray(np.array(plt.imread(filename))).flatten() for filename in filelist])
  numsamples= len(filearray)

  #reshaping
  x_predic = filearray[:numsamples].reshape(numsamples,input_shape[0],input_shape[1],input_shape[2])
  return x_predic, name, protein_list


def avefirstscore(proteins,name,prediction,outfile,threshold):

  ''' A function to convert the prediction of the model into a score, confidence
  measure, and a count of the ones and zeros for one map. The average first
  method averages the confidence first and then rounds to 1 or 0 for the whole
  map.

  **Arguments for avefirstscore:**

  * **proteins:** a list of the proteins being predicted
  * **predictionsL** the predictions produced by the model
  * **outfileL** the text file to save the predictions to
  * **threshold:** the certainty decimal tht the flag needs to have to classify as 1

  **Outputs of avefirstscore:**

  * **score:** the integer value (one or zero) given to the map - 1 for phased, 0 for unphased
  * **pred:** the confidence measure for how likely it is to be phased: 1 = 100% confident that this is phased, 0 = 0% confidence that this is phased
  * **ones:** the number of images that the model labeled as phased
  * **zeros:** the number of images that the model labeled as unphased

  '''

  text=open(outfile,'a')
  pred = []
  score =[]
  for protein in proteins:
    x=0
    n=0
    ones = 0
    zeros = 0
    text.write('\n'+protein+':\n')
    for i in range(0,len(name)):
      print(name[i])
      print(protein)
      if name[i] == protein:
        x+= prediction[i][1]
        text.write('prediction: '+str(prediction[i][1])+'\n')
        if prediction[i][1]>=threshold:
          ones+=1
        else:
          zeros+=1
        n+=1
       
      else:
        continue
    p=x/n
    pred.append(p)
    text.write('averaged likelihood of being phased: %s\n'%p)
    if p>=threshold:
      score.append(1)
      text.write('score of map: 1\n')
    else:
      score.append(0)
      text.write('score of map:0\n')
    text.write('ones %s\n'%ones)
    text.write('zeros %s\n'%zeros)
  text.close()
  return score, pred, ones, zeros

def roundfirstscore(proteins,name, prediction,outfile,threshold):

  ''' A function to convert the prediction of the model into a score, confidence
  measure, and a count of the ones and zeros for one map. The round first method
  rounds the prediction for each image to 1 or 0 before averaging all images in
  the map

  **Arguments for avefirstscore:**

  * **proteins:** a list of the proteins being predicted
  * **predictions:** the predictions produced by the model
  * **outfile:** the text file to save the predictions to

  **Outputs of avefirstscore:**

  * **score:** the integer value (one or zero) given to the map - 1 for phased, 0 for unphased
  * **pred:** the confidence measure for how likely it is to be phased: 1 = 100% confident that this is phased, 0 = 0% confidence that this is phased
  * **ones:** the number of images that the model labeled as phased
  * **zeros:** the number of images that the model labeled as unphased

  '''
  pred = []
  score =[]
  text=open(outfile,'a')
  for proteins in proteins:
    x=0
    n=0
    text.write('\n'+protein+':\n')
    for i in range(0,len(name)):
      if name[i]==protein:
        text.write('prediction: '+str(prediction[i][1])+'\n')
        if prediction[i][1]>=threshold:
          x=+1
          ones=+1
        else:
          zeros+=1
        n+=1
    p=x/n
    pred.append(p)
    text.write('averaged likelihood of being phased: %s\n'%p)
    if p>=threshold:
      score.append(1)
      text.write('score of map: 1\n')
    else:
      score.append(0)
      text.write('score of map: 0\n')
    text.write('ones %s\n'%ones)
    text.write('zeros %s\n'%zeros)
    text.close()
    return score, pred, ones, zeros


def trialsplitlist(listloc):
  ''' 
  A function to read the file detailing which proteins were not used to train  and read them into a list

  **Arguments for trialsplitlist:**

  * **listloc** The file location for the .txt file 

  **Output for trialsplitlist:**

  * **list_protein** The list of protein names that were excluded from training
  '''
  text=open(listloc,'r')
  list_protein=[]
  for line in text:
    protein = line[-9:-5]
    list_protein.append(protein)
  text.close()
  return list_protein



def columnclear(database, tablename, column):
  '''
  Function to change all values in a column of  a given column of an SQLite
  database to Null

  **Arguments for columnclear:**

  * **database:** file location for a given database
  * **tablename:** name of table
  * **column:** name of column

  '''
  conn = sqlite3.connect(database)
  cur = conn.cursor()
  for i in columns:
   cur.execute('''UPDATE %s SET %s = Null'''%(tablename,i))




