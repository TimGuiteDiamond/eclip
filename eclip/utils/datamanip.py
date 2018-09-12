################
'''
Functions to find and manipulate data for learning

|

'''
##############
import numpy as np
import sqlite3
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.utils import shuffle, class_weight
from sklearn.metrics import confusion_matrix
import random
import logging

from keras.utils import to_categorical
import argparse

####################################
def normal_array(n_array):

  '''
  **Arguments for normal_array:**

  * **n_array:** An array

  **Outputs of normal_array:***

  * **norm_array:** A normalised version of the input array with values between 0 and 1.

  |

  '''

  maxi=np.amax(n_array)
  mini=np.amin(n_array)
  norm_array=(n_array-mini)/(maxi-mini)
  return norm_array

def input_training_images(database,
                          input_shape,
                          fractionTrain,
                          raw =False,
                          number=10,
                          trialsplit = True):
  '''

  **Arguments for input_training_images:**

  * **database:** The file location of an sqlite database with the correct format
  * **input_shape:** The dimensions of the image files to be retrieved
  * **fractionTrain:** The fraction of the data retrieved to be used to test the model
  * **raw:** parameter specifying whether to take the raw or processed imagedata default = False
  * **number:** The number of images to call in per axis per protein
  * **trialsplit:** Boolean, true if some proteins should be kept separate

  **Outputs of input_training_images:**

  * **x_train:** An array of the image data from all the images read in for training
  * **y_train:** A two dimensional array of the true score of the images for training
  * **x_test:** An array of the image data from all the images read in for testing
  * **y_test:** A two dimensional array of the true score of the images for training

  |

  '''

  
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
    m=0
    while m<100:
      name_id=random.choice(Protein_id)
      Protein_id.remove(name_id)
      cur.execute('''SELECT pdb_id FROM PDB_id WHERE id = "%s"'''%(name_id))
      name_str=cur.fetchall()
      cur.execute('''SELECT ep_success_o FROM Phasing WHERE pdb_id_id =
      "%s"'''%(name_id))
      score_o = list(cur.fetchall()[0])[0]
      cur.execute('''SELECT ep_success_i FROM Phasing WHERE pdb_id_id =
      "%s"'''%(name_id))
      score_i = list(cur.fetchall()[0])[0]

      if raw:
        cur.execute('''SELECT ep_raw_o FROM Phasing WHERE pdb_id_id =
        "%s"'''%(name_id))
        rawimg=list(cur.fetchall()[0])[0]
        cur.execute('''SELECT ep_raw_i FROM Phasing WHERE pdb_id_id =
        "%s"'''%(name_id))
        rawimg_i=list(cur.fetchall()[0])[0]
        if not rawimg == None:
          if not score_o == None:
            m+=1
        if not rawimg_i == None:
          if not score_i == None:
            m+=1
      else:
        cur.execute('''SELECT ep_img_o FROM Phasing WHERE pdb_id_id =
        "%s"'''%(name_id))
        img=list(cur.fetchall()[0])[0]
        cur.execute('''SELECT ep_img_i FROM Phasing WHERE pdb_id_id =
        "%s"'''%(name_id))
        img_i=list(cur.fetchall()[0])[0]
        if not img == None:
          if not score_o == None:
            m+=1
        if not img_i == None:
          if not score_i == None:
            m+=1
      t.write(str(name_id)+' '+str(name_str)+'\n')
  t.close()
  
  for item in Protein_id:
 
   
   
    cur.execute('''SELECT ep_success_o FROM Phasing WHERE pdb_id_id = "%s"'''%item)
    y_o=list(cur.fetchall()[0])[0]
    if not y_o==None:
      if raw:
        cur.execute('''SELECT ep_raw_o FROM Phasing WHERE pdb_id_id =
        "%s"'''%item)
        x_o=list(cur.fetchall()[0])[0]
      else:
        cur.execute('''SELECT ep_img_o FROM Phasing WHERE pdb_id_id =
        "%s"'''%item)
        x_o=list(cur.fetchall()[0])[0]
      if not x_o==None:
        y_list.append(y_o)
        x_dir.append(x_o)

    cur.execute('''SELECT ep_success_i FROM Phasing WHERE pdb_id_id =
    "%s"'''%item)
    y_i=list(cur.fetchall()[0])[0]
    if not y_i == None:
      if raw:
        cur.execute('''SELECT ep_raw_i FROM Phasing WHERE pdb_id_id = "%s"'''%item)
        x_i = list(cur.fetchall()[0])[0]
      else:
        cur.execute('SELECT ep_img_i FROM Phasing WHERE pdb_id_id = "%s"'''%item)
        x_i=list(cur.fetchall()[0])[0]
      if not x_i==None:
        y_list.append(y_i)
        x_dir.append(x_i)
        print(x_i)

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
      for m in range(0,number):
        file_name=random.choice(os.listdir(img_dir))
        location=os.path.join(img_dir,file_name)
        filelist.append(location)
        label.append(y_list[i])

  
  filearray=np.array([normal_array(np.array(plt.imread(filename))).flatten() for filename in filelist])
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


def import_data(database, proteinlist, input_shape,raw, number = 10):
  '''Function to import new data to predict

  **Arguments for import_data:**

  * **datafileloc:** The location of the sqlite database
  * **proteinlist:** A list of the proteins to select from this directory
  * **input_shape:** The shape of a individual image to be selected 
  * **raw:** boolean, true if using unprocessed data
  * **number:** The number of images to use per axis per protein

  **Outputs for import_data:**

  * **x_predic:** An array containing the data for the selected images
  * **name:** A list of the names of the proteins associated with each image
  * **protein_list:** Proteins opened

  |

  '''
  name = []
  filelist = []
  protein_list = []
  problem = 0
  conn=sqlite3.connect(database)
  cur=conn.cursor()

  for protein in proteinlist:
    cur.execute(''' SELECT id FROM PDB_id WHERE pdb_id = "%s" ''' %(protein))
    pdb_id_id = list(cur.fetchall()[0])[0] 

    cur.execute('''SELECT ep_success_o FROM Phasing WHERE pdb_id_id = "%s"'''%(pdb_id_id))
    y_o = list(cur.fetchall()[0])[0]
    if not y_o==None:
      if raw:
        cur.execute('''SELECT ep_raw_o FROM Phasing WHERE pdb_id_id =
        "%s"'''%(pdb_id_id))
        f=list(cur.fetchall()[0])[0]
      else:
        cur.execute('''SELECT ep_img_o FROM Phasing WHERE pdb_id_id = "%s"'''
        %(pdb_id_id))
        f=list(cur.fetchall()[0])[0]
      if not f==None:
        protein_list.append(protein)
        for dir in os.listdir(f):
          direct = os.path.join(f,dir)
          for m in range(0,number):
            file = random.choice(os.listdir(direct))
            location = os.path.join(direct,file)
            filelist.append(location)
            name.append(protein)
      else: 
        problem+=1
    else:
      problem+=1

         
    cur.execute('''SELECT ep_success_i From Phasing WHERE pdb_id_id="%s"'''%(pdb_id_id))
    y_i = list(cur.fetchall()[0])[0]
    if not y_i == None:
      if raw: 
        cur.execute('''SELECT ep_raw_i FROM Phasing WHERE pdb_id_id = "%s"'''
        %(pdb_id_id))
        f=list(cur.fetchall()[0])[0]
      else:
        cur.execute('''SELECT ep_img_i FROM Phasing WHERE pdb_id_id = "%s"'''
        %(pdb_id_id))
        f=list(cur.fetchall()[0])[0]
      if not f==None:
        protein_list.append(protein+'_i')
        for dir in os.listdir(f):
          direct = os.path.join(f,dir)
          for m in range(0,number):
            file = random.choice(os.listdir(direct))
            location = os.path.join(direct,file)
            filelist.append(location)
            name.append(protein+'_i')
      else:
        problem+=1
    else: 
      problem+=1



  #making an array
  filearray=np.array([normal_array(np.array(plt.imread(filename))).flatten() for filename in filelist])
  numsamples= len(filearray)

  #reshaping
  x_predic = filearray[:numsamples].reshape(numsamples,input_shape[0],input_shape[1],input_shape[2])

  print('number of maps = %s'%(len(protein_list))) 
  print('problems = %s'%problem)
  return x_predic, name, protein_list


def ave_first_score(proteins,name,prediction,outfile,threshold):

  ''' A function to convert the prediction of the model into a score, confidence
  measure, and a count of the ones and zeros for one map. The average first
  method averages the confidence first and then rounds to 1 or 0 for the whole
  map.

  **Arguments for ave_first_score:**

  * **proteins:** a list of the proteins being predicted
  * **name:** A list of protein names
  * **predictions:** the predictions produced by the model
  * **outfile:** the text file to save the predictions to
  * **threshold:** the certainty decimal tht the flag needs to have to classify as 1

  **Outputs of ave_first_score:**

  * **score:** the integer value (one or zero) given to the map - 1 for phased, 0 for unphased
  * **pred:** the confidence measure for how likely it is to be phased: 1 = 100% confident that this is phased, 0 = 0% confidence that this is phased
  * **ones:** the number of images that the model labeled as phased
  * **zeros:** the number of images that the model labeled as unphased

  |

  '''

  
  pred = []
  score =[]
  for protein in proteins:
    x=0
    n=0
    ones = 0
    zeros = 0
    for i in range(0,len(name)):
      if name[i] == protein:
        x+= prediction[i][1]
        if prediction[i][1]>=threshold:
          ones+=1
        else:
          zeros+=1
        n+=1   
      else:
        continue
    p=x/n
    pred.append(p)   
    if p>=threshold:
      score.append(1)
      logging.info(
          '%s: averaged likelihood of being phased: %s score of map: 1'%(protein,p))
    else:
      score.append(0)
      logging.info(
          '%s: averaged likelihood of being phased: %s score of map: 0'%(protein,p))
    logging.info('ones %s: zeros %s\n'%(ones,zeros))
  print('number of maps with scores = %s'%(len(score)))
  return score, pred, ones, zeros

def round_first_score(proteins,name, prediction,outfile,threshold):

  ''' 
  A function to convert the prediction of the model into a score, confidence
  measure, and a count of the ones and zeros for one map. The round first method
  rounds the prediction for each image to 1 or 0 before averaging all images in
  the map

  **Arguments for round_first_score:**

  * **proteins:** a list of the proteins being predicted
  * **name:** list of protein names
  * **predictions:** the predictions produced by the model
  * **outfile:** the text file to save the predictions to
  * **threshold:** value to round up from

  **Outputs of round_first_score:**

  * **score:** the integer value (one or zero) given to the map - 1 for phased, 0 for unphased
  * **pred:** the confidence measure for how likely it is to be phased: 1 = 100% confident that this is phased, 0 = 0% confidence that this is phased
  * **ones:** the number of images that the model labeled as phased
  * **zeros:** the number of images that the model labeled as unphased

  |
  
  '''
  pred = []
  score =[]
  
  for proteins in proteins:
    x=0
    n=0
    ones = 0
    zeros = 0 
    for i in range(0,len(name)):
      if name[i]==protein:
        if prediction[i][1]>=threshold:
          x=+1
          ones=+1
        else:
          zeros+=1
        n+=1
    p=x/n
    pred.append(p)        
    if p>=threshold:
      score.append(1)
      logging.info(
        '%s: averaged likelihood of being phased: %s score of map: 1'%(protein,p))
    else:
      score.append(0)
      logging.info(
        '%s: averaged likelihood of being phased: %s score of map: 0'%(protein,p))
    logging.info('ones %s: zeros %s\n'%(ones,zeros))

    return score, pred, ones, zeros


def trial_split_list(listloc):
  ''' 
  A function to read the file detailing which proteins were not used to train  and read them into a list

  **Arguments for trial_split_list:**

  * **listloc** The file location for the .txt file 

  **Output for trial_split_list:**

  * **list_protein** The list of protein names that were excluded from training
  
  |
  
  '''
  text=open(listloc,'r')
  list_protein=[]
  for line in text:
    protein = line[-9:-5]
    list_protein.append(protein)
  text.close()
  return list_protein

def from_dir_list(fileloc):
  ''' 
  A function to read the names from the directory into a list

  **Arguments for from_dir_list:**

  * **fileloc:** the location of the directory to make a list from

  **Outputs for from_dir_list:**

  * **proteinlist:** the list of proteins
  
  |
  
  '''
  proteinlist = []
  for file in os.listdir(fileloc):
    name = file
    proteinlist.append(name)
  return proteinlist





def column_clear(database, tablename, column):
  '''
  Function to change all values in a column of  a given column of an SQLite
  database to Null

  **Arguments for column_clear:**

  * **database:** file location for a given database
  * **tablename:** name of table
  * **column:** name of column in list

  |
  
  '''
  conn = sqlite3.connect(database)
  cur = conn.cursor()
  for i in column:
   cur.execute('''UPDATE %s SET %s = Null'''%(tablename,i))
   print('done')
  conn.commit()
  conn.close()

def str2bool(v):
  '''
  Function to get argparse to recognise boolean arguments

  |

  '''
  if v.lower() in ('yes','true','True','t','y','1'):
    return True
  elif v.lower() in ('no','false','False','f','n','0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected')


