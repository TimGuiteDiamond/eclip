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


from eclip.utils.datamanip import inputTrainingImages
from eclip.utils.visu import ConfusionMatrix, fitplot, plot_test_results
from eclip.utils.modelmanip import mapModel, expjson, expyaml


################################################################################################

def main():
  '''
  |
  
  Main function of learn. learn.py creates a model, compiles it and fits it using a set of parameters
  and keras functions. 

  Other modules called in the package: 

   * inputTrainingImages from eclip.utils.datamanip
   * ConfusionMatrix and fitplot from eclip.utils.visu
   * mapModel from eclip.utils.modelmanip

  Calling learn results in the creation of: 

   * model.json
   * model.yaml
   * model.h5
   * predict------_-.txt
   * log------_-.txt
   * Plot------_-.png
   * CnfM------_-.png

  '''
 
  #: Parameters are set for running
  batchsize = 32 
  epochs=100
  loss_equation='categorical_crossentropy'
  learning_rate=0.0001
  parallel=True
  trial_num=1
  date='020818'
  decay_rt=1e-8
  input_shape = [201,201,3]
  output_directory = '/dls/science/users/ycc62267/eclip/eclip/paratry'

  #: trial_num is increased untill there are no previously saved plots with that trial_number
  while os.path.exists(os.path.join(output_directory,'Plot'+date+'_'+str(trial_num)+'.png')):
    trial_num+=1
  
  logfile = os.path.join(output_directory, 'log'+date+'_'+str(trial_num)+'.txt')
  
  #: A log file is created to record the starting parameters and the results of the learning.
  text=open(logfile,'w')
  text.write('''Running learn.py for parameters: \n batchsize: %s \n epochs : %s \n
  loss: %s \n learning rate: %s \n parallelization: %s \n decay rate: %s \n'''
  %(batchsize,epochs,loss_equation,learning_rate,parallel,decay_rt))
  print('running learn.py')
  text.close()
  
  #: learn calls inputTrainingImages to select image data and change into the correct form.
  x_train, y_train,x_test,y_test=inputTrainingImages(database='/dls/science/users/ycc62267/metrix_db/metrix_db.sqlite',input_shape=input_shape,fractionTrain=0.8)
  
  #Printing the data dimensions to logfile
  text=open(logfile,'a')
  text.write('train: ')
  ones=sum(y_train)
  text.write(str(ones))
  text.write('\n')
  text.write('test: ')
  ones=sum(y_test)
  text.write(str(ones)+'\n')
  
  #: Class weights are used to counter the effect of an imbalanced input dataset
  y_ints=[y.argmax() for y in y_train]
  class_weights=dict(enumerate(class_weight.compute_class_weight('balanced',np.unique(y_ints),y_ints)))
  print('classweights %s'%class_weights)
  text.write('Class weights: %s' %class_weights)
  text.write('input_shape: %s'%input_shape)
  text.close()
  
  #building model
  model = mapModel()
  model.createCustom2(input_shape2=input_shape,logfile=logfile)
  
  
  #optimiser could be opt=keras.optimizers.SGD(lr=0.0001), but should check
  opt = keras.optimizers.Adam(lr=learning_rate,decay=decay_rt)
  
  #: learn can be run on a 4 GPU cluster by setting parallel to True
  if parallel:
    model=multi_gpu_model(model,gpus=4)
  else:
    model=model

  #compiling the model
  model.compile(loss = loss_equation,
                optimizer = opt,
                metrics = ['accuracy'])
  print('model compiled')
  
  
  
  #fit the model on the training data- this is the bit that trains the model -
  #look at the need for augmenting the data- we have a lot so it may not be
  #necessary
  history=model.fit(x_train,y_train,class_weight=class_weights,batch_size =batchsize, epochs=epochs, verbose=1,validation_split=(0.33),callbacks=[])
  
  #: The validation loss and training loss is plotted against epochs and likewise with the validation accuracy and training accuracy
  outputplot=os.path.join(output_directory,'Plot'+date+'_'+str(trial_num)+'.png')
  fitplot(history.history['loss'],history.history['val_loss'],history.history['acc'],history.history['val_acc'],outputplot)
  
  #: Predicting the score of the test data
  prediction=model.predict(x_test,batch_size=batchsize,verbose=1)

  #:saveing model as .json file
  outfilejson= os.path.join(output_directory,'model.json')
  expjson(outfilejson,model)

  #:saving model as .yaml file 
  outfileyaml = os.path.join(output_directory,'model.yaml')
  expyaml(outfileyaml,model)

  #:saving weigths to .h5 file
  weights_out = os.path.join(output_directory,'model.h5')
  model.save_weights(weights_out)
  
  #Evaluating the success of the model
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
  
  
  #writing predictions
  outpred=os.path.join(output_directory,'predict'+date+'_'+str(trial_num)+'.txt')
  y_pred=ConfusionMatrix.printConvstats(prediction,outpred,logfile,y_test)
   
  print(model.summary())
  text=open(logfile,'a')
  model.summary(print_fn=lambda x: text.write(x+'\n'))
  text.close()
  
  #: A confusion matrix is plotted to visually assess the successes of the model
  cnfout=os.path.join(output_directory,'CnfM'+date+'_'+str(trial_num)+'.png')
  ConfusionMatrix(y_test,y_pred,cnfout)


  #:A plot of predictions and true values of test set
  splotout=os.path.join(output_directory,'Compplot'+date+'_'+str(trial_num)+'.png')
  plot_test_results(y_test,y_pred,splotout)
  
if __name__=="__main__":


  main()
