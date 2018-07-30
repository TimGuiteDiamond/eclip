#this is a beginning ideas for the machine learning for image slices

#one idea is to use the package Keras. this works in a modular manner- and is
#was used for the EM ones

#import required packages
import numpy as np
from PIL import Image
import matplotlib.pyplot  as plt
plt.switch_backend('agg')
np.random.seed(123) # for reproduciblility
import os

from sklearn.utils import shuffle
from sklearn.utils import class_weight
import keras
from keras import initializers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.utils import to_categorical, plot_model
from keras.applications import vgg16

from keras.utils import multi_gpu_model

from sklearn.metrics import confusion_matrix
################################################################################################


#############################################################################################
import sqlite3
def normalarray(n_array):
  maxi=np.amax(n_array)
  mini=np.amin(n_array)
  norm_array=(n_array-mini)/(maxi-mini)
  return norm_array

def inputTrainingImages(database,input_shape,fractionTrain):
  conn=sqlite3.connect(database)
  cur=conn.cursor()

  
  #Protein_id is a list of the pdb_id_ids from the database
  cur.execute('''SELECT pdb_id_id FROM Phasing''')
  Protein_id=cur.fetchall()

  #Create a list of directory names (x_dir) and a list of the scores (y_list)
  x_dir=[]
  y_list=[]

  
  for item in Protein_id:
 


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

  filelist=[]
  label=[]
  
  for i in range(0,len(y_list)):
    dirin = x_dir[i]
    for dir in os.listdir(dirin):
      img_dir=os.path.join(dirin,dir)
      m=0

      #chosing random files from directory
      #for m<5:
      #  m=+1
      #  filename=random.choice(os.listdir(img_dir))
      #  location=os.path.join(img_dir,filename)
      #  filelist.append(location)
      #  label.append(y_list[i])
      for file in os.listdir(img_dir):
        m+=1
        if m>4:
          break
        location=os.path.join(img_dir,file)
        filelist.append(location)
        label.append(y_list[i])
  
  #Normalising:
  filearray=np.array([normalarray(np.array(plt.imread(filename))).flatten() for filename in
  filelist])
  print('images have been read in')
  numsamples = len(filearray)
  #print(numsamples)
  #print(filearray.shape)

  #label is a list of scores for each image
  #filearray is an array of each image
  
  #need to make label and array:
  label=np.asarray(label)

  #label is an array of scores for each image
  #filearray is an array of each image
  
  #need to shuffle the data:
  filearray,label=shuffle(filearray,label,random_state=2)
  print('x after shuffle: ', filearray.shape)
  print('y after shuffle: ', label.shape)

#setting all y=0 to all zeros for x - testing
#  for i in range(0,len(label)):
#    if label[i] ==0:
#
#      length=len(filearray[i])
#      filearray[i]=filearray[i]/1

  ntrain=int(numsamples*fractionTrain)

  x_train = filearray[:ntrain].reshape(ntrain,input_shape[0],input_shape[1],input_shape[2])
  y_train = to_categorical(label[:ntrain],2)#this is used to create a binary class matrix

  ntest=numsamples-ntrain
  x_test = filearray[ntrain:].reshape(ntest,input_shape[0],input_shape[1],input_shape[2])
  y_test= to_categorical(label[ntrain:],2)
 
  return x_train, y_train, x_test, y_test



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

  def createCustom1(self,input_shape1,logfile):
    #mess with these once code is working... currently from keras tutorial -
    text=open(logfile,'a')
    text.write('\n Model setup: \n')
    #generic setup
    self.add(Convolution2D(32,(3,3),kernel_initializer.he_normal(),activation ='relu',input_shape=input_shape1))
    text.write("Convolution2D(32,(3,3),kernel_initializer.he_normal(),activation ='relu',input_shape=input_shape1) \n")
    self.add(Convolution2D(32,(3,3),activation = 'relu'))
    text.write("Convolution2D(32,(3,3),activation = 'relu')\n")
    self.add(MaxPooling2D(pool_size =(2,2)))
    text.write("MaxPooling2D(pool_size =(2,2))\n")
    self.add(Dropout(0.25))
    text.write("Dropout(0.25)\n")
    self.add(Flatten())
    text.write("Flatten()\n")
    self.add(Dense(128,activation='relu'))
    text.write("Dense(128,activation='relu')\n")
    self.add(Dropout(0.5))
    text.write("Dropout(0.5)\n")
    self.add(Dense(2,activation = 'softmax'))
    text.write("Dense(2,activation = 'softmax')\n")
    text.close()
  def createCustom2(self,input_shape2,logfile):
    #another model, with more layers and a different final layer
    text=open(logfile,'a')
    text.write('\nModel setup:\n')
    self.add(Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(),activation='relu',padding='same',input_shape=input_shape2,name='Conv_1'))
    text.write("Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(),activation='relu',padding='same',input_shape=input_shape2,name='Conv_1')\n")
    self.add(MaxPooling2D(pool_size=(2,2),name='Pool_1'))
    text.write("MaxPooling2D(pool_size=(2,2)name='Pool_1')\n")
    self.add(BatchNormalization())
    text.write("BatchNormalization()\n")
    self.add(Convolution2D(32,(3,3),activation='relu',padding='same',name='Conv_2'))
    text.write("Convolution2D(32,(3,3),activation='relu',padding='same',name='Conv_2')\n")
 #   self.add(Convolution2D(64,(3,3),activation='relu',padding='same',name='Conv_3'))
 #   text.write("Convolution2D(64,(3,3),activation='relu',padding='same',name='Conv_3')\n")
    self.add(MaxPooling2D(pool_size=(2,2),name='Pool_2'))
    text.write("MaxPooling2D(pool_size(2,2),name='Pool_2')\n")
    self.add(BatchNormalization())
    text.write("BatchNormalizaion()\n")
    self.add(Dropout(0.5))
    text.write("Dropout(0.5)\n")
 #   self.add(Convolution2D(128,(3,3),activation='relu',padding='same',name='Conv_3'))
 #   text.write("Convolution2D(128,(3,3),activation='relu',padding='same',name='Conv_3')\n")
    self.add(MaxPooling2D(pool_size=(2,2),name='Pool_3'))
    text.write("MaxPooling2D(pool_size=(2,2),name='Pool_3')\n")
    self.add(Flatten())
    text.write("Flatten()\n")
    self.add(Dropout(0.75))
    text.write("Dropout(0.75)\n")
    self.add(BatchNormalization())
    text.write("BatchNormalizaion()\n")
    self.add(Dense(128,activation='relu',name='Dense_1'))
    text.write("Dense(128,activation='relu',name='Dense_1')\n")
    self.add(Dense(2,activation='softmax',name='predictions'))
    text.write("Dense(2,activation='softmax',name='predictions')\n")
    text.close()

  def createCustom3(self,input_shape3):
    #another model
    text=open(logfile,'a')
    self.add(Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(),padding='same',activation='relu',input_shape=input_shape3,name='Conv_1'))
    text.write("Convolution2D(32,(3,3),kernel_initializer=initializers.he_normal(),padding='same',activation='relu',input_shape=input_shape3, name='Conv_1')\n")
    self.add(Convolution2D(32,(3,3),activation = 'relu',name='Conv_2'))
    text.write("Convolution2D(32,(3,3),activation='relu',name='Conv_2')\n")
    self.add(MaxPooling2D(pool_size=(2,2),name='Pool_1'))
    text.write("MaxPooling2D(pool_size=(2,2),name='Pool_1')\n")
    self.add(Dropout(0.25))
    text.write("Dropout(0.25)\n")

    self.add(Convolution2D(64,(3,3),padding='same', activation='relu'))
    text.write("Convolution2D(64,(3,3),padding='same',activation='relu')\n")
    self.add(Convolution2D(64,(3,3),activation='relu'))
    text.write("Convolution2S(64,(3,3),activation='relu')\n")
    self.add(MaxPooling2D(pool_size=(2,2)))
    text.write("MaxPooling2D(pool_size=(2,2))\n")
    self.add(Dropout(0.25))
    text.write("Dropout(0.25)\n")
    self.add(Flatten())
    text.write("Flatten()\n")
    self.add(Dense(80,activation='relu'))
    text.write("Dense(80,activation='relu')\n")
    self.add(Dropout(0.5))
    text.write("Dropout(0.5)\n")
    self.add(Dense(2,activation='softmax'))
    text.write("Dense(2,activation='softmax')\n")
    text.close()

##this needs some work on- dont really understand this bit!
#class WeightsCheck(keras.callbacks.Callback):
#    def on_epoch_begin(self,batch,logs={}):
#      for layer in model.layers[16:17]:
#        print(layer.get_weights())
#    def on_epoch_end(self,batch,logs={}):
#      for layer in model.layers[16:17]):
#        print(layer.get_weights())


####################################################################################
batchsize=32
epochs=150
loss_equation='categorical_crossentropy'
learning_rate = 0.0001
parallel=True
trial_num=1
date='310718'
decay_rt = 1e-8

output_directory = '/dls/science/users/ycc62267/eclip/eclip/paratry'
while os.path.exists('/dls/science/users/ycc62267/eclip/eclip/paratry/Plot'+date+'_'+str(trial_num)+'.png'):
  trial_num+=1

logfile = os.path.join(output_directory, 'log'+date+'_'+str(trial_num)+'.txt')

text=open(logfile,'w')
text.write('running test.py for parameters: \n batchsize: %s \n epochs: %s \n loss: %s\n learning rate: %s\n parallelization: %s \ndecay rate: %s \n' %(batchsize,epochs,loss_equation,learning_rate,parallel,decay_rt))
print('running test.py')

text.close()
#x_train,y_train=inputTrainingImages(database='/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite',input_shape=[201,201,3],Protein_id=[584,504,114])

#x_test, y_test=inputTrainingImages(database='/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite',input_shape=[201,201,3],Protein_id=[334])

x_train , y_train, x_test,y_test=inputTrainingImages(database='/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite',input_shape=[201,201,3],fractionTrain=0.8)

text=open(logfile,'a')
#print('train')
text.write('train: ')
ones=sum(y_train)
text.write(str(ones))
text.write('\n')

text.write('test: ')
ones=sum(y_test)
text.write(str(ones)+'\n')


y_ints=[y.argmax() for y in y_train]
class_weights=dict(enumerate(class_weight.compute_class_weight('balanced',np.unique(y_ints),y_ints)))
print('classweights %s'%class_weights)
text.write('Class weights: %s' %class_weights)
input_shape=[201,201,3]
text.write('input_shape: %s'%input_shape)
model = mapModel()
model.createCustom2(input_shape2=input_shape,logfile=logfile)

#compile the model
#optimiser could be opt=keras.optimizers.SGD(lr=0.0001), but should check
#initiate RMSprop optimizer - mess with this once code is working
opt = keras.optimizers.Adam(lr=learning_rate,decay=decay_rt)


if parallel:
  model = multi_gpu_model(model,gpus=4)
else:
  model=model

model.compile(loss = loss_equation,
              optimizer = opt,
              metrics = ['accuracy'])
print('model compiled')


##loading previous weights option
#load_weights=False
#weights_file = 'file location for previous weights'
#if load_weights:
#  model.load_weights(weights_file)
#  print('weights loaded')

#weights_check = WeightsCheck()

#fit the model on the training data- this is the bit that trains the model -
#varies the weights.
#look at the need for augmenting the data- we have a lot so it may not be
#necessary



history=model.fit(x_train,y_train,class_weight=class_weights,batch_size = batchsize,epochs=epochs, verbose=1,validation_split=(0.33),callbacks=[])

#while os.path.exists('/dls/science/users/ycc62267/eclip/eclip/paratry/Plot'+date+'_'+str(trial_num)+'.png'):
#  trial_num+=1

fig=plt.figure()
plt.subplot(2,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.subplot(2,2,2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model train')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='lower right')
fig.savefig('/dls/science/users/ycc62267/eclip/eclip/paratry/Plot'+date+'_'+str(trial_num)+'.png')
plt.close(fig)
#plt.show()
#verbose gives a progress bar


prediction=model.predict(x_test,batch_size=batchsize,verbose=1)
#print(prediction)

#evaluate the model on the test data
#this sees how good the model is by applying it to the test data

loss,acc = model.evaluate(x_test, y_test, verbose =1)
print('loss is: %s'%loss)
print('accuracy is: %s'%acc)

text.write('loss is: %s\n'%loss)
text.write('accuracy is: %s\n'%acc)
text.write('predictions:\n')
prediction(print_fn=lambda x: text.write(x+'\n'))


#exporting model
json_string=model.to_json()
with open('/dls/science/users/ycc62267/eclip/eclip/model.json','w') as outfile:
  outfile.write(json_string)
#outfile.write(json_string)???
outfile.close()

yaml_string=model.to_yaml()
with open('/dls/science/users/ycc62267/eclip/eclip/model.yaml','w') as outfile2:
  outfile2.write(yaml_string)
outfile2.close()

##writing predictions and true positive etc
fileout= '/dls/science/users/ycc62267/eclip/eclip/predictions.txt'
outfile=open(fileout,'w')
n_tn = 0
n_tp=0
n_fn=0
n_fp=0
#
#print('writing predictions')
y_pred = []
for i in range(len(prediction)):
  outfile.write(str(prediction[i])+";"+str(y_test[i])+'\n')
  if prediction[i][0] > 0.5 and y_test[i][0]>0.5:
    n_tn+=1
    y_pred.append(0)
  if prediction[i][0]>0.5 and y_test[i][1]>0.5:
    n_fn+=1
    y_pred.append(0)
  if prediction[i][1]>0.5 and y_test[i][1]>0.5:
    n_tp+=1
    y_pred.append(1)
  if prediction[i][1]>0.5 and y_test[i][0]>0.5:
    n_fp+=1
    y_pred.append(1)
#
outfile.close()
print('Number of true negatives = ',n_tn)
print('Number of false negatives = ',n_fn)
print('Number of true positives = ',n_tp)
print('Number of false positives = ',n_fp)
text.write('Number of true negatives = %s\n'%n_tn)
text.write('Number of false negatives = %s\n'%n_fn)
text.write('Number of true positives = %s\n'%n_tp)
text.write('Number of false positives = %s\n'%n_fp)


##saveing output weights to a file:
#import hdf5
weights_out = '/dls/science/users/ycc62267/eclip/eclip/model.h5'
model.save_weights(weights_out)

print(model.summary())
model.summary(print_fn=lambda x: text.write(x+'\n'))
text.close()


############################################################################################
#Confusion matrix
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
  if normalize:
    cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
  plt.imshow(cm,interpolation='nearest',cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks=np.arange(len(classes))
  plt.xticks(tick_marks,classes,rotation=45)
  plt.yticks(tick_marks,classes)

  fmt='.2f' if normalise else 'd'
  thresh = cm.max()/2.
  for i,j in intertools.product(range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j,i,format(cm[i,j],fmt),
              horizontalalignment="center",
              color = "white" if cm[i,j]>thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
print(y_pred)
print(y_test)
#y_pred=prediction
y_pred=np.asarray(y_pred)
print(y_pred)
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_names,normalize=True,title='Normalized Confusion Matrix')

fig.savefig('/dls/science/users/ycc62267/eclip/eclip/paratry/Cnf'+date+'_'+str(trial_num)+'.png')
plt.close()
#plt.show()


###############################################################################################


#to create images of what it thinks 1 and 0 look like

##Conv filter visualization-visualize via Activation Maximizatoin
#from vis.utils import utils
#
##from keras import activations
#search for layer index by name
#layer_idx = utils.find_layer_idx(model,'predictions')
#
##swap softmax with linear
#model.layers[layer_idx].activation=activations.linear
#model.save('temp.h5')
#model=load_model('temp.h5',custom_objects={'mapModel':Sequential})
#model=utils.apply_modifications(model)
#os.remove('temp.h5')
#
#from vis.visualization import visualize_activation, visualize_saliency
##plt.rcParams['figure.figsize']=(18,6)
#categories = np.asarray([0,1])
#
#for idx in categories:
#  img=visualize_activation(model,layer_idx,filter_indices=idx)
#  img=utils.draw_text(img,utils.get_imagenet_label(idx))
#  vis_images.append(img)


#indices = np.where(y_test[:,class_idx]==1.)[0]
#
#idx=indices[0]
#class_idx = 0
#grads = visualize_saliency(model,layer_idx,filter_indices=class_idx,
#seed_input=x_test[idx])
#f,ax=plt.subplot(1,2)
#ax[0].imshow(x_test[idx][...,0])
#ax[1].imshow(grads,cmap='jet')
#f.savefig('/dls/science/users/ycc62267/eclip/eclip/paratry/Sal'+date+'_'+str(trial_num)+'.png')
#plt.close()

#


#plt.rcParams['figure.figsize']=(50,50)
#stitched=utils.stitch_imgas(vis_images,cols=2)
#plt.axis('off')
#plt.imsave('/dls/science/users/ycc62267/eclip/eclip/paratry/Img'+date+'_'+str(trial_num)+'.png',stitched)

###########################################################################################################
#different way to visualise
#get the symboolic outputs of each "key" layer 
#layer_dict=dict([(layer.name,layer) for layer in model.layers])
#print(layer_dict)
##now define a loss function
#from keras import backend as K
##
#layer_name='Conv_1_input'
#filter_index=0
##
#layer_output=layer_dict[layer_name].output
#loss=K.mean(layer_output[:,:,:,filter_index])
#
#grads=K.gradients(loss,input_img)[0]
#grads/=(K.sqrt(K.mean(K.square(grads)))+1e-5)
# 
#
#input_img = model.input
#iterate=K.function([input_img],[loss,grad(loss,input_img)])
#
#input_img_data = np.random.random((1,3,201,201))*20+128
#
#for i in range(20):
#  loss_value, grads_value = iterate([input_img_data])
#  input_img_data += grads_value*step
#
#
#from scipy.misc import imsave
#
#def deprocess_image(x):
#  x-=x.mean()
#  x/=(x.sdt()+1e-5)
#  x*=0.1
#
#  x+=0.5
#  x=np.clip(x,0,1)
#
#  x*=255
#  x=x.transpose((1,2,0))
#  x=np.clip(x,0,255).astype('uint8')
#  return x
#
#img = input_image_data[0]
#img = deprocess_image(img)
#imsave('/dls/science/users/ycc62267/eclip/eclip/paratry/filter_%s_%s.png'%(date,str(trial_num)),img)
#
#this bit was from
#keras.io/how-convolutional-nerual-networks-see-the-world.html in theory this
#should work though i have not tried it yet

