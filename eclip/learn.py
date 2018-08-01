#this is a beginning ideas for the machine learning for image slices

#one idea is to use the package Keras. this works in a modular manner- and is
#was used for the EM ones

#import required packages
import numpy as np
#from PIL import Image
import matplotlib.pyplot  as plt
plt.switch_backend('agg')
np.random.seed(123) # for reproduciblility
import os

#from sklearn.utils import shuffle
from sklearn.utils import class_weight
#from sklearn.metrics import confusion_matrix

#import random


import keras
#from keras import initializers,regularizers
#from keras.models import Sequential, load_model
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import multi_gpu_model #, np_utils
#from keras.utils import plot_model, to_categorical
#from keras.applications import vgg16


from utils.datamanip import inputTrainingImages
from utils.visu import ConfusionMatrix, fitplot
from utils.modelmanip import mapModel


################################################################################################
'''Parameters:'''

batchsize = 32
epochs=100
loss_equation='categorical_crossentropy'
learning_rate=0.0001
parallel=True
trial_num=1
date='010818'
decay_rt=1e-8
input_shape = [201,201,3]

output_directory = '/dls/science/users/ycc62267/eclip/eclip/paratry'
while os.path.exists(os.path.join(output_directory,'Plot'+date+'_'+str(trial_num)+'.png')):
  trial_num+=1

logfile = os.path.join(output_directory, 'log'+date+'_'+str(trial_num)+'.txt')

text=open(logfile,'w')
text.write('''Running learn.py for parameters: \n batchsize: %s \n epochs : %s \n
loss: %s \n learning rate: %s \n parallelization: %s \n decay rate: %s \n'''
%(batchsize,epochs,loss_equation,learning_rate,parallel,decay_rt))
print('running learn.py')
text.close()

##############################################################################################

x_train, y_train,x_test,y_test=inputTrainingImages(database='/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite',input_shape=input_shape,fractionTrain=0.8)

text=open(logfile,'a')
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
text.write('input_shape: %s'%input_shape)
text.close()

model = mapModel()
model.createCustom2(input_shape2=input_shape,logfile=logfile)

#compile the model
#optimiser could be opt=keras.optimizers.SGD(lr=0.0001), but should check
#initiate RMSprop optimizer - mess with this once code is working
opt = keras.optimizers.Adam(lr=learning_rate,decay=decay_rt)

if parallel:
  model=multi_gpu_model(model,gpus=4)
else:
  model=model

model.compile(loss = loss_equation,
              optimizer = opt,
              metrics = ['accuracy'])
print('model compiled')



#fit the model on the training data- this is the bit that trains the model -
#varies the weights.
#look at the need for augmenting the data- we have a lot so it may not be
#necessary

history=model.fit(x_train,y_train,class_weight=class_weights,batch_size =batchsize, epochs=epochs, verbose=1,validation_split=(0.33),callbacks=[])

outputplot=os.path.join(output_directory,'Plot'+date+'_'+str(trial_num)+'.png')

fitplot(history.history['loss'],history.history['val_loss'],history.history['acc'],history.history['val_acc'],outputplot)


prediction=model.predict(x_test,batch_size=batchsize,verbose=1)

loss,acc = model.evaluate(x_test,y_test,verbose=1)
print('loss is: %s'%loss)
print('accuracy is: %s'%acc)

text=open(logfile,'a')
text.write('loss is: %s\n'%loss)
text.write('accuracy is: %s\n'%acc)
text.write('Predictions: \n')

with open(logfile,'a') as f:
  for line in prediction:
    np.savetxt(f,line,fmt='%.2f',delimiter=',',newline=' ')
    f.write('\n')
text.close()


#writing predictions and true positive etc
outpred=os.path.join(output_directory,'predict'+date+'_'+str(trial_num)+'.txt')
y_pred=ConfusionMatrix.printConvstats(prediction,outpred,logfile,y_test)



print(model.summary())
text=open(logfile,'a')
model.summary(print_fn=lambda x: text.write(x+'\n'))
text.close()

cnfout=os.path.join(output_directory,'CnfM'+date+'_'+str(trial_num)+'.png')
ConfusionMatrix(y_test,y_pred,cnfout)


